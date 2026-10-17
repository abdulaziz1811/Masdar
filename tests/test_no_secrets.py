"""No credential is ever committed: every shipped file is scanned for one.

The project is handed from person to person, and a key pasted into a
config file or a doc "just to try" is the usual way one leaks. Keys live in
the environment (`.env`, gitignored, or the host's settings); the
repository holds only the *names* of the variables. This scans the files
that ship -- code, configuration, docs, deployment files -- for the shapes
the services in use give their keys.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

KEY_SHAPES = {
    "Google / Gemini": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Anthropic": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "Firecrawl": re.compile(r"\bfc-[0-9a-f]{32}\b"),
    "Groq": re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
    "GitHub": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "a key assigned in place": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{24,}"
    ),
}
# Tests build fake keys on purpose, to check they are refused or not logged.
SKIPPED = ("tests/",)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".js", ".css", ".html", ".toml",
                 ".txt", ".sh", ".bat", ".example", ""}


def shipped_files() -> list[Path]:
    try:
        listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    files = []
    for name in listed:
        path = ROOT / name
        if name.startswith(SKIPPED) or not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in ("Dockerfile", ".env.example"):
            files.append(path)
    return files


def test_no_file_holds_a_key():
    found = []
    for path in shipped_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for service, shape in KEY_SHAPES.items():
            for match in shape.finditer(text):
                found.append(f"{path.relative_to(ROOT)}: {service} ({match.group()[:12]}…)")
    assert not found, "credential-shaped text in shipped files:\n" + "\n".join(found)


def test_the_scan_would_see_one(tmp_path):
    # A scan that cannot fail proves nothing.
    sample = ("GEMINI_API_KEY=AIza" + "x" * 35 + "\napikey: " + "A1b2C3d4" * 4
              + "\nGROQ: gsk_" + "Ab3" * 18)
    hits = [s for s, shape in KEY_SHAPES.items() if shape.search(sample)]
    assert {"Google / Gemini", "a key assigned in place", "Groq"} <= set(hits)


def test_the_example_env_file_names_variables_and_holds_no_values():
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        assert not value.strip(), f"{name} has a value in .env.example"
