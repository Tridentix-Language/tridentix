#!/usr/bin/env python3
"""Collapse consecutive //! and /// blocks to at most two lines. Leaves code unchanged."""
from pathlib import Path

ROOTS = [
    Path("compiler"),
    Path("runtime"),
]


def collapse(lines):
    out = []
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i]
        stripped = s.lstrip()
        if stripped.startswith("//!") or stripped.startswith("///"):
            prefix = "//!" if stripped.startswith("//!") else "///"
            indent = s[: len(s) - len(stripped)]
            block = []
            while i < n:
                cur = lines[i]
                cs = cur.lstrip()
                if cs.startswith(prefix):
                    block.append(cur)
                    i += 1
                else:
                    break
            if len(block) <= 2:
                out.extend(block)
            else:
                kept = 0
                for b in block:
                    body = b.lstrip()[len(prefix) :].strip()
                    if body:
                        out.append(f"{indent}{prefix} {body}\n" if not b.endswith("\n") else f"{indent}{prefix} {body}\n")
                        kept += 1
                        if kept == 2:
                            break
                if kept == 0:
                    out.append(f"{indent}{prefix}\n")
            continue
        out.append(s if s.endswith("\n") or s == "" else s + "\n")
        if not lines[i].endswith("\n") and i == n - 1:
            pass
        i += 1
    return out


def main():
    for root in ROOTS:
        for path in root.rglob("*.rs"):
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(keepends=True)
            new = collapse(lines)
            path.write_text("".join(new), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
