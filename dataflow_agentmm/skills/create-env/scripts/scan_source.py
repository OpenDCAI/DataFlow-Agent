#!/usr/bin/env python3
"""Secret-safe static inventory for a prospective DataFlow-AgentMM Env source.

The scanner reports paths and structural signals. It never prints source file
contents or values from credential-like files. Results are heuristic and must
be confirmed against authoritative registration and handler code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

MANIFEST_NAMES = {
    "Cargo.toml",
    "Gemfile",
    "go.mod",
    "package.json",
    "pom.xml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}

DOCUMENT_NAMES = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE.md",
    "README",
    "README.md",
    "README.rst",
}

SENSITIVE_NAME_PATTERNS = (
    re.compile(r"^\.env(?:\..+)?$", re.IGNORECASE),
    re.compile(r"(?:credential|secret|token|cookie|api[-_]?key)", re.IGNORECASE),
    re.compile(r"\.(?:key|pem|p12|pfx)$", re.IGNORECASE),
)

LANGUAGES = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".sh": "Shell",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
}

TEXT_SUFFIXES = set(LANGUAGES) | {
    ".json",
    ".jsonc",
    ".md",
    ".rst",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}

SIGNALS: dict[str, tuple[re.Pattern[str], ...]] = {
    "mcp": tuple(re.compile(value, re.IGNORECASE) for value in (
        r"@modelcontextprotocol/sdk",
        r"\bFastMCP\b",
        r"\bModel Context Protocol\b",
        r"\bListToolsRequestSchema\b",
        r"\bCallToolRequestSchema\b",
        r"\btools/list\b",
        r"\bregisterTool\b",
        r"\bserver\.tool\s*\(",
    )),
    "agentmm_env": tuple(re.compile(value) for value in (
        r"\bdataflow_agentmm\.contracts\b",
        r"\bEnvironmentSpec\s*\(",
        r"\bregister_env\s*\(",
        r"class\s+\w*Env\s*\(\s*Env\s*\)",
        r"def\s+tools\s*\(\s*self",
        r"def\s+call\s*\(\s*self",
    )),
    "tool_schema": tuple(re.compile(value, re.IGNORECASE) for value in (
        r"\bToolSpec\s*\(",
        r"\bregisterTool\b",
        r"\bserver\.tool\s*\(",
        r"\binputSchema\b",
        r"\binput_schema\b",
        r"\btools/call\b",
        r"\bcall_tool\b",
    )),
    "process_or_browser": tuple(re.compile(value, re.IGNORECASE) for value in (
        r"\bplaywright\b",
        r"\bpuppeteer\b",
        r"\bselenium\b",
        r"\bsubprocess\b",
        r"\bchild_process\b",
        r"\bexecFile\b",
        r"\bspawn\s*\(",
    )),
}

TOOL_PATTERNS = (
    re.compile(r"ToolSpec\s*\(\s*name\s*=\s*[\"']([A-Za-z0-9_.-]{1,100})[\"']", re.DOTALL),
    re.compile(r"\b_tool\s*\(\s*[\"']([A-Za-z0-9_.-]{1,100})[\"']"),
    re.compile(r"(?:registerTool|server\.tool)\s*\(\s*[\"']([A-Za-z0-9_.-]{1,100})[\"']"),
    re.compile(r"\bname\s*:\s*[\"']([A-Za-z0-9_.-]{1,100})[\"']"),
)


def _is_sensitive(path: Path) -> bool:
    return any(pattern.search(path.name) for pattern in SENSITIVE_NAME_PATTERNS)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _walk(
    root: Path,
    max_files: int,
) -> tuple[list[Path], bool, list[str], list[str]]:
    files: list[Path] = []
    ignored_present: set[str] = set()
    symlinks_skipped: set[str] = set()
    stack = [root]
    truncated = False
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                symlinks_skipped.add(_relative(child, root))
                continue
            if child.is_dir():
                if child.name in IGNORED_DIRECTORIES:
                    ignored_present.add(_relative(child, root))
                else:
                    stack.append(child)
                continue
            if not child.is_file():
                continue
            files.append(child)
            if len(files) >= max_files:
                truncated = True
                stack.clear()
                break
    return files, truncated, sorted(ignored_present), sorted(symlinks_skipped)


def _read_text(path: Path, max_read_bytes: int) -> str | None:
    if _is_sensitive(path):
        return None
    try:
        if path.stat().st_size > max_read_bytes:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _rank_recommended(
    manifests: Iterable[str],
    documents: Iterable[str],
    signal_files: dict[str, list[str]],
) -> list[str]:
    ranked: list[str] = []
    for values in (
        sorted(set(manifests)),
        sorted(set(documents)),
        signal_files.get("mcp", []),
        signal_files.get("agentmm_env", []),
        signal_files.get("tool_schema", []),
    ):
        for value in values:
            if value not in ranked:
                ranked.append(value)
            if len(ranked) >= 25:
                return ranked
    return ranked


def scan(root: Path, *, max_files: int, max_read_bytes: int) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"source must be a directory: {root}")
    files, truncated, ignored_present, symlinks_skipped = _walk(root, max_files)
    language_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    manifests: list[str] = []
    documents: list[str] = []
    sensitive_paths: list[str] = []
    signal_files: dict[str, list[str]] = defaultdict(list)
    signal_counts: Counter[str] = Counter()
    tools: dict[str, set[str]] = defaultdict(set)
    unreadable_text: list[str] = []

    for path in files:
        relative = _relative(path, root)
        suffix = path.suffix.lower() or "[none]"
        suffix_counts[suffix] += 1
        if path.suffix.lower() in LANGUAGES:
            language_counts[LANGUAGES[path.suffix.lower()]] += 1
        if path.name in MANIFEST_NAMES:
            manifests.append(relative)
        if path.name in DOCUMENT_NAMES or path.name.lower().startswith("readme"):
            documents.append(relative)
        if _is_sensitive(path):
            sensitive_paths.append(relative)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in MANIFEST_NAMES:
            continue
        text = _read_text(path, max_read_bytes)
        if text is None:
            unreadable_text.append(relative)
            continue
        matched_tool_schema = False
        for signal_name, patterns in SIGNALS.items():
            count = sum(len(pattern.findall(text)) for pattern in patterns)
            if count:
                signal_counts[signal_name] += count
                if len(signal_files[signal_name]) < 30:
                    signal_files[signal_name].append(relative)
                if signal_name == "tool_schema":
                    matched_tool_schema = True
        if matched_tool_schema:
            if path.suffix.lower() in {".json", ".jsonc"}:
                try:
                    document = json.loads(text)
                except json.JSONDecodeError:
                    document = None
                if isinstance(document, dict) and isinstance(document.get("tools"), list):
                    for item in document["tools"]:
                        if isinstance(item, dict):
                            name = item.get("name")
                            if isinstance(name, str) and re.fullmatch(
                                r"[A-Za-z0-9_.-]{1,100}", name
                            ):
                                tools[name].add(relative)
            for pattern in TOOL_PATTERNS:
                for name in pattern.findall(text):
                    if name.lower() not in {"string", "object", "array", "number"}:
                        tools[name].add(relative)

    if signal_counts["agentmm_env"]:
        source_kind = "agentmm-env"
    elif signal_counts["mcp"]:
        source_kind = "mcp"
    elif manifests:
        source_kind = "application-or-library"
    elif files:
        source_kind = "source-tree"
    else:
        source_kind = "empty"

    tool_candidates = [
        {"name": name, "files": sorted(paths)[:8]}
        for name, paths in sorted(tools.items())
    ]
    return {
        "root": str(root),
        "source_kind": source_kind,
        "scanned_files": len(files),
        "truncated": truncated,
        "languages": dict(language_counts.most_common()),
        "top_suffixes": dict(suffix_counts.most_common(20)),
        "manifests": sorted(set(manifests)),
        "documents": sorted(set(documents)),
        "signals": {
            name: {
                "matches": signal_counts[name],
                "files": sorted(set(signal_files[name])),
            }
            for name in SIGNALS
        },
        "tool_candidates": tool_candidates,
        "sensitive_filenames": sorted(set(sensitive_paths)),
        "large_binary_or_unreadable_text": sorted(set(unreadable_text))[:50],
        "ignored_directories_present": ignored_present,
        "symlinks_skipped": symlinks_skipped,
        "recommended_reads": _rank_recommended(manifests, documents, signal_files),
        "warnings": [
            "Heuristic inventory only; confirm tools against registration and handlers.",
            "Sensitive filenames are listed, but their contents were not read.",
            "Ignored/generated directories were not scanned.",
        ],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Source scan",
        "",
        f"- Root: `{report['root']}`",
        f"- Classified as: `{report['source_kind']}`",
        f"- Files scanned: {report['scanned_files']}",
        f"- Truncated: {str(report['truncated']).lower()}",
        "",
    ]
    for title, key in (
        ("Languages", "languages"),
        ("Manifests", "manifests"),
        ("Documentation", "documents"),
        ("Recommended reads", "recommended_reads"),
        ("Sensitive filenames (contents not read)", "sensitive_filenames"),
        ("Ignored directories present", "ignored_directories_present"),
        ("Symlinks skipped", "symlinks_skipped"),
    ):
        lines.extend((f"## {title}", ""))
        value = report[key]
        if isinstance(value, dict):
            lines.extend(f"- `{name}`: {count}" for name, count in value.items())
        elif isinstance(value, list):
            lines.extend(f"- `{item}`" for item in value)
        if not value:
            lines.append("- None detected")
        lines.append("")

    lines.extend(("## Structural signals", ""))
    for name, value in report["signals"].items():  # type: ignore[union-attr]
        files = value["files"]
        lines.append(f"- `{name}`: {value['matches']} match(es) in {len(files)} file(s)")
    lines.extend(("", "## Candidate tool names", ""))
    candidates = report["tool_candidates"]
    if candidates:
        for candidate in candidates:  # type: ignore[assignment]
            joined = ", ".join(f"`{item}`" for item in candidate["files"])
            lines.append(f"- `{candidate['name']}` — {joined}")
    else:
        lines.append("- None detected")
    lines.extend(("", "## Warnings", ""))
    lines.extend(f"- {item}" for item in report["warnings"])  # type: ignore[union-attr]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Statically inventory a source tree before building an DataFlow-AgentMM Env."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--max-read-bytes", type=int, default=512_000)
    args = parser.parse_args(argv)
    if args.max_files < 1 or args.max_read_bytes < 1:
        parser.error("scan limits must be positive")
    try:
        report = scan(
            args.source,
            max_files=args.max_files,
            max_read_bytes=args.max_read_bytes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "json":
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
