"""Read `.env` into the environment, without a dependency.

The documentation tells users to put credentials in `.env`; this makes that
true for every command. Values already in the environment win, nothing read
here is ever printed, and only plain `NAME=value` lines are understood.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def load_env_file(path: str | Path = ".env") -> list[str]:
    """Set unset variables from `path`; return the names set (never values).

    Forgiving about the two things Windows Notepad does to a file named
    `.env`: it may save it as `.env.txt` (with the extension hidden), and it
    may start it with a byte-order mark, which would otherwise hide the first
    line -- usually the only one, the key.
    """
    file = Path(path)
    if not file.exists() and file.name == ".env":
        notepad = file.with_name(".env.txt")
        if notepad.exists():
            file = notepad
    try:
        text = file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return []
    loaded: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded
