"""File tools — read, write, surgical edit, list, glob, grep."""

from __future__ import annotations

import fnmatch
import os
import re

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".idea", ".vscode", "dist", "build", ".next", "vendor"}


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path or "."))


def register(r) -> None:

    @r.register("read_file", "Read a text file, optionally a line range",
                {"path": "string: file path",
                 "?start_line": "integer: 1-based first line",
                 "?end_line": "integer: last line (inclusive)"})
    def read_file(ctx, path: str, start_line: int = 0, end_line: int = 0) -> str:
        path = _expand(path)
        if not os.path.isfile(path):
            return f"Not a file: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        s = max(1, int(start_line) or 1)
        e = min(total, int(end_line) or total)
        body = "".join(f"{i:>5}| {line}" for i, line in
                       enumerate(lines[s - 1:e], start=s))
        if len(body) > 24000:
            body = body[:24000] + f"\n...[truncated; file has {total} lines, use line ranges]"
        return f"{path} (lines {s}-{e} of {total}):\n{body}"

    @r.register("write_file", "Create or overwrite a text file",
                {"path": "string: file path", "content": "string: full file content"})
    def write_file(ctx, path: str, content: str = "") -> str:
        path = _expand(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        existed = os.path.exists(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        verb = "Overwrote" if existed else "Wrote"
        return f"{verb} {path} ({len(content)} chars)"

    @r.register("edit_file", "Replace an exact text snippet in a file",
                {"path": "string: file path",
                 "find": "string: exact text to find",
                 "replace": "string: replacement text",
                 "?replace_all": "boolean: replace every occurrence"})
    def edit_file(ctx, path: str, find: str, replace: str = "",
                  replace_all: bool = False) -> str:
        path = _expand(path)
        if not os.path.isfile(path):
            return f"Not a file: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        count = text.count(find)
        if count == 0:
            return "Snippet not found — read the file and match it exactly (whitespace matters)."
        if count > 1 and not replace_all:
            return f"Snippet occurs {count} times; provide more context or set replace_all."
        new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        return f"Edited {path}: {count if replace_all else 1} replacement(s)."

    @r.register("list_dir", "List a directory (entries with sizes)",
                {"?path": "string: directory, default cwd"})
    def list_dir(ctx, path: str = ".") -> str:
        path = _expand(path)
        if not os.path.isdir(path):
            return f"Not a directory: {path}"
        entries = sorted(os.listdir(path))
        out = []
        for e in entries[:200]:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                out.append(f"  {e}/")
            else:
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                human = (f"{size/1048576:.1f}M" if size > 1048576
                         else f"{size/1024:.0f}K" if size > 1024 else f"{size}B")
                out.append(f"  {e}  ({human})")
        more = f"\n  ... +{len(entries)-200} more" if len(entries) > 200 else ""
        return f"{path}:\n" + "\n".join(out) + more

    @r.register("glob_search", "Find files by glob pattern, recursive",
                {"pattern": "string: e.g. *.py or test_*",
                 "?root": "string: directory to search, default cwd"})
    def glob_search(ctx, pattern: str, root: str = ".") -> str:
        root = _expand(root)
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if fnmatch.fnmatch(fn, pattern):
                    hits.append(os.path.relpath(os.path.join(dirpath, fn), root))
                    if len(hits) >= 200:
                        return f"{root} ({pattern}), first 200:\n" + "\n".join(hits)
        return (f"{root} ({pattern}), {len(hits)} match(es):\n" + "\n".join(hits)
                if hits else f"No files matching {pattern} under {root}")

    @r.register("grep_search", "Regex search inside files",
                {"pattern": "string: regular expression",
                 "?root": "string: directory, default cwd",
                 "?glob": "string: filename filter e.g. *.py"})
    def grep_search(ctx, pattern: str, root: str = ".", glob: str = "*") -> str:
        root = _expand(root)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Bad regex: {e}"
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if not fnmatch.fnmatch(fn, glob):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(full) > 2_000_000:
                        continue
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                rel = os.path.relpath(full, root)
                                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                                if len(hits) >= 100:
                                    return f"First 100 matches:\n" + "\n".join(hits)
                except OSError:
                    continue
        return ("\n".join(hits) if hits
                else f"No matches for /{pattern}/ in {root} ({glob})")
