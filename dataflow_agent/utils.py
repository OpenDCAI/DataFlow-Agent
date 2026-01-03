import json
import re
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Union

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def robust_parse_json(
    text: str,
    *,
    merge_dicts: bool = False,
    strip_double_braces: bool = False,
) -> Union[Dict[str, Any], List[Any]]:
    """
    尽量从 LLM / 日志 / jsonl / Markdown 片段中提取合法 JSON。

    参数
    ----
    text : str
        输入原始文本
    merge_dicts : bool, default False
        提取到多个对象且全部是 dict 时，是否用 dict.update 合并返回
    strip_double_braces : bool, default False
        把 '{{' / '}}' 替换成 '{' / '}'（某些模板语言会加双层花括号）

    返回
    ----
    Dict / List / List[Dict | List]
    """
    s = text.strip()

    s = _remove_markdown_fence(s)
    s = _remove_outer_triple_quotes(s)
    s = _remove_leading_json_word(s)

    if strip_double_braces:
        s = s.replace("{{", "{").replace("}}", "}")

    s = _strip_json_comments(s)

    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

    s = s.replace("\\\\", "\x00DOUBLE_BACKSLASH\x00")
    s = s.replace("\\n", "\x00NEWLINE\x00")
    s = s.replace("\\r", "\x00RETURN\x00")
    s = s.replace("\\t", "\x00TAB\x00")
    s = s.replace('\\"', "\x00QUOTE\x00")
    s = s.replace("\\/", "\x00SLASH\x00")
    s = s.replace("\\b", "\x00BACKSPACE\x00")
    s = s.replace("\\f", "\x00FORMFEED\x00")
    s = s.replace("\\", "\\\\")
    s = s.replace("\x00DOUBLE_BACKSLASH\x00", "\\\\")
    s = s.replace("\x00NEWLINE\x00", "\\n")
    s = s.replace("\x00RETURN\x00", "\\r")
    s = s.replace("\x00TAB\x00", "\\t")
    s = s.replace("\x00QUOTE\x00", '\\"')
    s = s.replace("\x00SLASH\x00", "\\/")
    s = s.replace("\x00BACKSPACE\x00", "\\b")
    s = s.replace("\x00FORMFEED\x00", "\\f")

    try:
        result = json.loads(s)
        return result
    except JSONDecodeError as exc:
        log.debug(f"整体解析失败: {exc}")

    objs = _parse_json_lines(s)
    if objs is not None:
        return _maybe_merge(objs, merge_dicts)

    objs = _extract_json_objects(s)
    if not objs:
        raise ValueError("Unable to locate any valid JSON fragment.")

    return _maybe_merge(objs, merge_dicts)


_outer_fence_pat = re.compile(r"^\\s*```[\\w-]*\\s*([\\s\\S]*?)```\\s*$", re.I)


def _remove_markdown_fence(src: str) -> str:
    match = _outer_fence_pat.match(src)
    if match:
        return match.group(1).strip()
    return src


def _remove_outer_triple_quotes(src: str) -> str:
    if (src.startswith("'''") and src.endswith("'''")) or (
        src.startswith('"""') and src.endswith('"""')
    ):
        return src[3:-3].strip()
    return src


def _remove_leading_json_word(src: str) -> str:
    return src[4:].lstrip() if src.lower().startswith("json") else src


def _strip_json_comments(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    # src = re.sub(r"(?<![:\"'])//.*", "", src)
    src = re.sub(r",\s*([}\]])", r"\1", src)
    return src.strip()


def _parse_json_lines(src: str) -> Union[List[Any], None]:
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return None

    objs: List[Any] = []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except JSONDecodeError:
            return None
    return objs


def _extract_json_objects(src: str) -> List[Any]:
    decoder = JSONDecoder()
    idx, n = 0, len(src)
    objs: List[Any] = []

    while idx < n:
        m = re.search(r"[{\\[]", src[idx:])
        if not m:
            break
        idx += m.start()
        try:
            obj, end = decoder.raw_decode(src, idx)
            tail = src[end:].lstrip()
            if tail and tail[0] not in ",]}>\n\r":
                idx += 1
                continue
            objs.append(obj)
            idx = end
        except JSONDecodeError:
            idx += 1
    return objs


def _maybe_merge(objs: List[Any], merge_dicts: bool) -> Union[Any, List[Any]]:
    if len(objs) == 1:
        return objs[0]
    if merge_dicts and all(isinstance(o, dict) for o in objs):
        merged: Dict[str, Any] = {}
        for o in objs:
            merged.update(o)
        return merged
    return objs
