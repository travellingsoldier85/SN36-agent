"""Improved SN36 web automation agent — v2.

Key changes from v1:
- Dramatically reduced prompt size (fewer tokens = lower cost = higher reward)
- Removed internal tool-call mechanism (saves LLM calls per step)
- Removed "done" action (evaluator auto-detects task success)
- Better candidate selection with tighter limits
- Improved loop/stuck detection with smarter recovery
- Cleaner browser state representation
- Lower max_tokens for LLM output (only need short JSON)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import os
import re
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit, urljoin, urlparse
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException

os.environ.setdefault("LLM_PROVIDER", "openai")

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url

try:
    from autoppia_iwa.src.web_agents.classes import IWebAgent
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401
    _AUTOPPIA_IWA_IMPORT_OK = True
except Exception:
    IWebAgent = object
    Task = Any
    BaseAction = Any
    _AUTOPPIA_IWA_IMPORT_OK = False

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


app = FastAPI(title="Autoppia Web Agent API")
logger = logging.getLogger("autoppia_operator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ─── Utilities ───────────────────────────────────────────────────────────

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _normalize_demo_url(raw_url: str | None) -> str:
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized
    try:
        if "://" not in normalized:
            if not normalized.startswith("/"):
                if "." in normalized or ":" in normalized:
                    parsed = urlsplit(f"http://{normalized}")
                    return urlunsplit(("http", "localhost", parsed.path or "/", parsed.query, parsed.fragment))
                normalized = f"/{normalized}"
            return f"http://localhost{normalized}"
        parsed = urlsplit(normalized)
        return urlunsplit(("http", "localhost", parsed.path or "/", parsed.query, parsed.fragment))
    except Exception:
        return "http://localhost/"


def _is_navigate_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower() in {"navigateaction", "navigate"}


def _sanitize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    if _is_navigate_action_type(payload.get("type")):
        payload["url"] = _normalize_demo_url(payload.get("url"))
    return payload


def _preserve_seed_url(target_url: str, current_url: str) -> str:
    try:
        cur = urlparse(current_url or "")
        tgt = urlparse(target_url or "")
        cur_seed = (parse_qs(cur.query).get("seed") or [None])[0]
        if not cur_seed:
            return target_url
        q = parse_qs(tgt.query)
        if (q.get("seed") or [None])[0] == str(cur_seed):
            return target_url
        q["seed"] = [str(cur_seed)]
        new_q = urlencode(q, doseq=True)
        fixed = tgt._replace(query=new_q)
        if not fixed.scheme and not fixed.netloc:
            return f"{fixed.path}{'?' + fixed.query if fixed.query else ''}{'#' + fixed.fragment if fixed.fragment else ''}"
        return fixed.geturl()
    except Exception:
        return target_url


def _resolve_url(url: str, base_url: str) -> str:
    try:
        u = str(url or "").strip()
        b = str(base_url or "").strip()
        if not u:
            return ""
        return urljoin(b, u) if b else u
    except Exception:
        return str(url or "").strip()


def _same_path_query(a: str, b: str) -> bool:
    try:
        pa, pb = urlparse(a), urlparse(b)
        return (pa.path or "/") == (pb.path or "/") and pa.query == pb.query
    except Exception:
        return a == b


# ─── Per-task state ──────────────────────────────────────────────────────

_TASK_STATE: dict[str, dict[str, object]] = {}


# ─── Selectors ───────────────────────────────────────────────────────────

def _sel_attr(attribute: str, value: str) -> Dict[str, Any]:
    return {"type": "attributeValueSelector", "attribute": attribute, "value": value, "case_sensitive": False}


def _sel_text(value: str) -> Dict[str, Any]:
    return {"type": "tagContainsSelector", "value": value, "case_sensitive": False}


def _sel_custom(value: str) -> Dict[str, Any]:
    return {"type": "attributeValueSelector", "attribute": "custom", "value": value, "case_sensitive": False}


def _build_selector(tag: str, attrs: Dict[str, str], *, text: str) -> Dict[str, Any]:
    if attrs.get("id"):
        return _sel_attr("id", attrs["id"])
    if attrs.get("data-testid"):
        return _sel_attr("data-testid", attrs["data-testid"])
    if tag == "a" and attrs.get("href") and not attrs["href"].lower().startswith("javascript:"):
        return _sel_attr("href", attrs["href"])
    if attrs.get("aria-label"):
        return _sel_attr("aria-label", attrs["aria-label"])
    if attrs.get("name"):
        return _sel_attr("name", attrs["name"])
    if attrs.get("placeholder"):
        return _sel_attr("placeholder", attrs["placeholder"])
    if attrs.get("title"):
        return _sel_attr("title", attrs["title"])
    if text and tag in {"button", "a"}:
        return _sel_text(text)
    return _sel_custom(tag)


def _selector_repr(selector: Dict[str, Any]) -> str:
    t = selector.get("type")
    a = selector.get("attribute")
    v = selector.get("value")
    if t == "attributeValueSelector":
        vv = str(v)[:80]
        return f"attr[{a}]={vv}"
    if t == "tagContainsSelector":
        return f"text~={v}"
    return str(selector)


# ─── Candidate ───────────────────────────────────────────────────────────

class _Candidate:
    __slots__ = ("selector", "text_selector", "text", "tag", "attrs", "context", "group")

    def __init__(self, selector, text, tag, attrs, *, text_selector=None, context="", group=""):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context
        self.group = group

    def click_selector(self) -> Dict[str, Any]:
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr in {"id", "href", "data-testid", "name", "aria-label", "placeholder", "title"}:
                return self.selector
        for a in ("id", "data-testid", "href", "aria-label", "name", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)
        try:
            t = (self.text or '').strip()
            if t and self.tag in {'button', 'a'}:
                return _sel_custom(f"{self.tag}:has-text({json.dumps(t)})")
        except Exception:
            pass
        if self.text_selector:
            return self.text_selector
        return self.selector

    def type_selector(self) -> Dict[str, Any]:
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr and attr != "class":
                return self.selector
        for a in ("id", "data-testid", "name", "aria-label", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)
        return _sel_custom(self.tag)


# ─── Candidate extraction ────────────────────────────────────────────────

def _attrs_to_str_map(attrs) -> Dict[str, str]:
    out = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = " ".join(str(x) for x in v if x is not None).strip()
        else:
            out[k] = str(v)
    return out


def _is_hidden(attr_map: Dict[str, str]) -> bool:
    if attr_map.get("hidden") is not None:
        return True
    if str(attr_map.get("aria-hidden") or "").lower() == "true":
        return True
    style = str(attr_map.get("style") or "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    classes = str(attr_map.get("class") or "").lower()
    if any(tok in classes for tok in ("hidden", "sr-only", "invisible")):
        return True
    return False


def _extract_label(soup, el, attr_map: Dict[str, str]) -> str:
    tag = str(getattr(el, "name", "") or "")
    if tag in {"a", "button"}:
        t = _norm_ws(el.get_text(" ", strip=True))
        if t:
            return t[:120]
    for key in ("aria-label", "placeholder", "title"):
        if attr_map.get(key):
            return _norm_ws(attr_map[key])[:120]
    if attr_map.get("id"):
        lab = soup.find("label", attrs={"for": attr_map["id"]})
        if lab:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]
    parent_label = el.find_parent("label")
    if parent_label:
        t = _norm_ws(parent_label.get_text(" ", strip=True))
        if t:
            return t[:120]
    return ""


def _get_context(el) -> str:
    """Get short context from nearest container."""
    try:
        cur = el
        for _ in range(6):
            cur = cur.parent
            if cur is None:
                break
            tag = str(getattr(cur, "name", "") or "")
            if tag not in {"li", "tr", "article", "section", "div", "td"}:
                continue
            txt = _norm_ws(cur.get_text(" ", strip=True))
            if 30 <= len(txt) <= 500:
                return txt[:150]
    except Exception:
        pass
    return ""


def _extract_candidates_bs4(html: str, *, max_candidates: int) -> List[_Candidate]:
    soup = BeautifulSoup(html, "lxml")
    selectors = ["button", "a[href]", "input", "textarea", "select", "[role='button']", "[role='link']"]
    els = []
    for sel in selectors:
        els.extend(soup.select(sel))

    seen: set[tuple[str, str, str]] = set()
    out: List[_Candidate] = []

    for el in els:
        tag = str(getattr(el, "name", "") or "")
        attr_map = _attrs_to_str_map(getattr(el, "attrs", {}) or {})

        if tag == "input" and attr_map.get("type", "").lower() == "hidden":
            continue
        if attr_map.get("disabled") is not None or attr_map.get("aria-disabled", "").lower() == "true":
            continue
        if _is_hidden(attr_map):
            continue

        label = _extract_label(soup, el, attr_map)

        # Group for display
        group = 'PAGE'
        try:
            if el.find_parent('nav'):
                group = 'NAV'
            elif el.find_parent('form'):
                group = 'FORM'
        except Exception:
            pass

        context = _get_context(el)

        # Select options for <select> elements
        if tag == "select":
            opts = []
            try:
                for o in el.find_all("option")[:8]:
                    t = o.get_text(" ", strip=True)
                    if t:
                        opts.append(t)
            except Exception:
                pass
            if opts:
                label = (label or "select") + f" options=[{', '.join(opts[:6])}]"
                label = label[:180]

        primary = _build_selector(tag, attr_map, text=label)

        text_sel = None
        if tag in {"a", "button"} and label:
            text_sel = _sel_text(label)

        sig = (str(primary.get("type") or ""), str(primary.get("attribute") or ""), str(primary.get("value") or ""))
        if sig in seen:
            continue
        seen.add(sig)

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context, group=group))
        if len(out) >= max_candidates:
            break

    return out


class _FallbackExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text = []
        self._tag = None
        self._attrs = {}
        self.candidates = []

    def handle_starttag(self, tag, attrs):
        attr_map = {k: (v or "") for k, v in attrs}
        self._tag = tag
        self._attrs = attr_map
        if tag in {"button", "a", "input", "textarea", "select"} or attr_map.get("role") in {"button", "link"}:
            label = attr_map.get("aria-label") or attr_map.get("placeholder") or attr_map.get("title") or ""
            selector = _build_selector(tag, attr_map, text=label)
            self.candidates.append(_Candidate(selector, label, tag, attr_map))

    def handle_data(self, data):
        if self._tag in {"button", "a"} and data.strip():
            self._text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == self._tag and self._text and self.candidates:
            text = " ".join(self._text)[:120]
            c = self.candidates[-1]
            c.text = text or c.text
            if c.tag in {"button", "a"} and c.text:
                c.text_selector = _sel_text(c.text)
        self._text = []


def _extract_candidates(html: str, max_candidates: int = 40) -> List[_Candidate]:
    if not html:
        return []
    if BeautifulSoup is not None:
        try:
            return _extract_candidates_bs4(html, max_candidates=max_candidates)
        except Exception:
            pass
    parser = _FallbackExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.candidates[:max_candidates]


# ─── Candidate selection for LLM ─────────────────────────────────────────

def _select_for_llm(task: str, candidates: List[_Candidate], url: str, max_total: int = 35) -> List[_Candidate]:
    """Pick a compact, relevant candidate set. Prioritize form controls, then buttons, then links."""
    controls, buttons, links, others = [], [], [], []

    for c in candidates:
        # Skip self-links
        try:
            if c.tag == "a":
                href = (c.attrs or {}).get("href", "")
                if href:
                    ph, pc = urlparse(href), urlparse(url or "")
                    if ph.path and pc.path and ph.path == pc.path:
                        continue
        except Exception:
            pass

        if c.tag in {"input", "textarea", "select"}:
            controls.append(c)
        elif c.tag == "button":
            buttons.append(c)
        elif c.tag == "a":
            links.append(c)
        else:
            others.append(c)

    picked = []
    seen = set()

    def add(arr, limit):
        for c in arr:
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:60]}"
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(c)
            if len(picked) >= limit:
                return

    add(controls, max_total)
    add(buttons, max_total)
    add(links, max_total)
    add(others, max_total)

    return picked[:max_total]


# ─── Browser state formatting ────────────────────────────────────────────

def _format_browser_state(candidates: List[_Candidate]) -> str:
    """Compact numbered list of interactive elements."""
    lines = []
    for i, c in enumerate(candidates):
        label = (c.text or '').strip() or (c.attrs or {}).get('placeholder', '') or (c.attrs or {}).get('aria-label', '')
        label = str(label).strip()

        # Key attributes
        bits = []
        for k in ('id', 'name', 'type', 'placeholder', 'href', 'role', 'value'):
            v = (c.attrs or {}).get(k)
            if v:
                vv = str(v)[:50]
                bits.append(f"{k}={vv}")
        attrs_str = (' ' + ' '.join(bits)) if bits else ''

        # Short context for disambiguating repeated buttons
        ctx = ''
        if c.tag in {'a', 'button'} and c.context:
            ctx = ' :: ' + c.context[:80]

        lines.append(f"[{i}]<{c.tag}>{label}</{c.tag}>{attrs_str}{ctx}")
    return "\n".join(lines)


# ─── Page summary ────────────────────────────────────────────────────────

def _page_summary(html: str, limit: int = 600) -> str:
    """Extract title + headings + visible text snippet."""
    if not html:
        return ""
    if BeautifulSoup is None:
        return _norm_ws(re.sub(r"<[^>]+>", " ", html))[:limit]

    try:
        soup = BeautifulSoup(html, "lxml")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()

        parts = []
        if soup.title:
            parts.append(f"TITLE: {_norm_ws(soup.title.get_text(' ', strip=True))[:100]}")

        heads = []
        for h in soup.find_all(["h1", "h2", "h3"], limit=8):
            t = _norm_ws(h.get_text(" ", strip=True))
            if t and t not in heads:
                heads.append(t[:80])
        if heads:
            parts.append("HEADINGS: " + " | ".join(heads[:6]))

        text = _norm_ws(soup.get_text(" ", strip=True))
        if text:
            parts.append(f"TEXT: {text[:300]}")

        return "\n".join(parts)[:limit]
    except Exception:
        return ""


# ─── LLM decision ────────────────────────────────────────────────────────

def _parse_llm_json(content: str) -> Dict[str, Any]:
    raw = str(content or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        s = raw
        if s.startswith("```"):
            s = re.sub(r"^```\w*\n?", "", s)
            s = re.sub(r"\n?```$", "", s)
            s = s.strip()
        start = s.find("{")
        end = s.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        raise ValueError(f"Non-JSON LLM output: {raw[:200]}")


def _history_summary(history: List[Dict[str, Any]] | None) -> str:
    if not history:
        return ""
    lines = []
    for h in (history or [])[-5:]:
        ok = 'OK' if h.get('exec_ok', True) else 'FAIL'
        lines.append(f"  {h.get('step','?')}. {h.get('action','')} text={h.get('text','')} [{ok}]")
    return "\n".join(lines)


def _detect_loop(history: List[Dict[str, Any]] | None) -> str:
    if not history or len(history) < 3:
        return ""
    last3 = history[-3:]
    actions = [(h.get("action", ""), h.get("text", "")) for h in last3]
    if len(set(str(a) for a in actions)) == 1:
        return " WARNING: You are stuck in a loop repeating the same action. You MUST choose a completely different action or scroll."
    return ""


def _llm_decide(
    *,
    task_id: str,
    task: str,
    step_index: int,
    url: str,
    candidates: List[_Candidate],
    page_info: str,
    html_snapshot: str,
    history: List[Dict[str, Any]] | None,
    model_override: str = "",
) -> Dict[str, Any]:

    n_candidates = len(candidates)
    browser_state = _format_browser_state(candidates)

    system_msg = (
        "You are a web automation agent. Given a task and the current page state, choose ONE action.\n"
        "Return ONLY a JSON object with keys: action, candidate_id, text, url\n"
        "Actions: click, type, select, navigate, scroll_down, scroll_up\n"
        "Rules:\n"
        "- click/type/select: candidate_id must be valid index [0," + str(n_candidates - 1) + "]\n"
        "- type/select: text must be non-empty\n"
        "- navigate: url must be full URL. Preserve query params (especially seed=)\n"
        "- For form tasks: fill fields then click submit\n"
        "- For navigation tasks: click the right link or navigate to target URL\n"
        "- Never output 'done' — the system detects completion automatically\n"
        "- If stuck, try scroll_down to reveal more content"
    )

    loop_warning = _detect_loop(history)
    history_text = _history_summary(history)

    user_msg = (
        f"TASK: {task}\n"
        f"STEP: {step_index}\n"
        f"URL: {url}\n\n"
        f"PAGE:\n{page_info}\n\n"
        f"ELEMENTS:\n{browser_state}\n"
    )

    if history_text:
        user_msg += f"\nHISTORY:\n{history_text}\n"

    if loop_warning:
        user_msg += f"\n{loop_warning}\n"

    user_msg += "\nReturn JSON: {action, candidate_id, text, url}"

    model = str(model_override or os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "200"))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    def _call() -> Dict[str, Any]:
        resp = openai_chat_completions(
            task_id=task_id,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]
        return _parse_llm_json(content)

    def _valid(obj: Dict[str, Any]) -> bool:
        a = (obj.get("action") or "").lower()
        if a not in {"click", "type", "select", "navigate", "scroll_down", "scroll_up"}:
            return False
        if a == "navigate":
            u = obj.get("url")
            return isinstance(u, str) and bool(u.strip())
        if a in {"click", "type", "select"}:
            cid = obj.get("candidate_id")
            if isinstance(cid, str) and cid.isdigit():
                cid = int(cid)
            if not isinstance(cid, int) or not (0 <= cid < n_candidates):
                return False
            if a in {"type", "select"}:
                t = obj.get("text")
                if not isinstance(t, str) or not t.strip():
                    return False
        return True

    # Try up to 2 times
    for attempt in range(2):
        try:
            obj = _call()
            if _valid(obj):
                return obj
        except Exception:
            pass

        # On retry, add a correction message
        if attempt == 0:
            messages.append({"role": "assistant", "content": '{"action":"invalid"}'})
            messages.append({"role": "user", "content": (
                f"Invalid response. Return valid JSON. "
                f"candidate_id must be integer in [0,{n_candidates - 1}]. "
                f"type/select need non-empty text. "
                f"If unsure, use scroll_down."
            )})

    # Final fallback
    return {"action": "scroll_down"}


# ─── Main Agent ──────────────────────────────────────────────────────────

class ApifiedWebAgent(IWebAgent):
    def __init__(self, id: str = "1", name: str = "AutoppiaOperator") -> None:
        self.id = str(id)
        self.name = str(name)

    async def act(self, *, task: Task, snapshot_html: str, screenshot=None, url: str, step_index: int, history=None) -> list:
        task_id = str(getattr(task, "id", "") or "")
        prompt = str(getattr(task, "prompt", "") or "")
        create_action_fn = getattr(BaseAction, "create_action", None)

        payload = {
            "task_id": task_id,
            "prompt": prompt,
            "snapshot_html": snapshot_html,
            "url": url,
            "step_index": int(step_index),
            "history": history or [],
        }
        resp = await self.act_from_payload(payload)
        actions = resp.get("actions") if isinstance(resp, dict) else []
        out = []
        for a in actions if isinstance(actions, list) else []:
            if not isinstance(a, dict):
                continue
            try:
                ac = create_action_fn(a) if callable(create_action_fn) else None
                if ac is not None:
                    out.append(ac)
            except Exception:
                continue
        return out

    async def act_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(payload.get("task_id") or "")
        task = str(payload.get("prompt") or payload.get("task_prompt") or "")
        model_override = str(payload.get("model") or "").strip()
        url = _normalize_demo_url(str(payload.get("url") or ""))
        step_index = int(payload.get("step_index") or 0)
        html = payload.get("snapshot_html") or ""
        history = payload.get("history") if isinstance(payload.get("history"), list) else None

        def _resp(actions):
            return {"actions": [_sanitize_action_payload(a) for a in actions if isinstance(a, dict)]}

        # Extract candidates
        candidates_all = _extract_candidates(html, max_candidates=50)
        candidates = _select_for_llm(task, candidates_all, url, max_total=35)

        if task_id == "check":
            if candidates:
                return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}])
            return _resp([{"type": "WaitAction", "time_seconds": 0.1}])

        # Get effective URL (track navigations)
        st = _TASK_STATE.get(task_id)
        if isinstance(st, dict):
            eu = str(st.get("effective_url") or "").strip()
            if eu:
                url = eu

        page_info = _page_summary(html)

        try:
            base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            if not os.getenv("OPENAI_API_KEY") and not is_sandbox_gateway_base_url(base_url):
                raise RuntimeError("OPENAI_API_KEY not set")

            decision = _llm_decide(
                task_id=task_id,
                task=task,
                step_index=step_index,
                url=url,
                candidates=candidates,
                page_info=page_info,
                html_snapshot=html,
                history=history,
                model_override=model_override,
            )
        except Exception:
            return _resp([{"type": "ScrollAction", "down": True, "up": False}])

        action = (decision.get("action") or "").lower()
        cid = decision.get("candidate_id")
        text = decision.get("text")
        if isinstance(cid, str) and cid.isdigit():
            cid = int(cid)

        # Initialize task state
        if task_id and task_id not in _TASK_STATE:
            _TASK_STATE[task_id] = {}
        tstate = _TASK_STATE.get(task_id, {})

        if action == "navigate":
            nav_url = str(decision.get("url") or "").strip()
            if not nav_url:
                return _resp([{"type": "ScrollAction", "down": True, "up": False}])
            nav_url = _resolve_url(nav_url, url)
            nav_url = _preserve_seed_url(nav_url, url)
            if _same_path_query(nav_url, url):
                return _resp([{"type": "ScrollAction", "down": True, "up": False}])
            if task_id:
                tstate["effective_url"] = nav_url
                _TASK_STATE[task_id] = tstate
            return _resp([{"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}])

        elif action in {"scroll_down", "scroll_up"}:
            return _resp([{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}])

        elif action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
            c = candidates[cid]

            if action == "click":
                selector = c.click_selector()
                # Check if clicking an href link — preserve seed and navigate if needed
                if isinstance(selector, dict) and selector.get("attribute") == "href":
                    href = str(selector.get("value") or "")
                    fixed = _preserve_seed_url(href, url)
                    fixed_abs = _resolve_url(fixed, url)
                    if _same_path_query(fixed_abs, url):
                        return _resp([{"type": "ScrollAction", "down": True, "up": False}])
                    if fixed != href:
                        if task_id:
                            tstate["effective_url"] = fixed_abs
                            _TASK_STATE[task_id] = tstate
                        return _resp([{"type": "NavigateAction", "url": fixed_abs, "go_back": False, "go_forward": False}])
                return _resp([{"type": "ClickAction", "selector": selector}])

            elif action == "type":
                if not text:
                    return _resp([{"type": "ScrollAction", "down": True, "up": False}])
                selector = c.type_selector()
                return _resp([{"type": "TypeAction", "selector": selector, "text": str(text)}])

            else:  # select
                if not text:
                    return _resp([{"type": "ScrollAction", "down": True, "up": False}])
                selector = c.type_selector()
                return _resp([{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text), "timeout_ms": 4000}])

        else:
            # Invalid action — scroll to reveal more
            return _resp([{"type": "ScrollAction", "down": True, "up": False}])


# ─── HTTP API ────────────────────────────────────────────────────────────

AutoppiaOperator = ApifiedWebAgent
OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/act")
async def act(payload: Dict[str, Any] = Body(...)):
    raw_resp = await OPERATOR.act_from_payload(payload)
    actions = raw_resp.get("actions") if isinstance(raw_resp, dict) else []
    normalized = []
    for action in actions if isinstance(actions, list) else []:
        try:
            if isinstance(action, dict):
                normalized.append(_sanitize_action_payload(action))
            else:
                normalized.append(_sanitize_action_payload(action.model_dump(exclude_none=True)))
        except Exception:
            continue
    return {"actions": normalized}


@app.post("/step")
async def step(payload: Dict[str, Any] = Body(...)):
    return await act(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
