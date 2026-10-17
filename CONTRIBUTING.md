# Contributing to Masdar

Start with the [developer guide](docs/DEVELOPER.md) (Arabic): a map of the code, one question's
path through it, and the rules no change may break. This page covers what every change needs
before and after.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
make install            # the package with the test and lint tools
pre-commit install      # optional: checks before every commit (see .pre-commit-config.yaml)
```

## Before every push

```bash
make check              # ruff, then the full test suite, offline (about two minutes)
```

This is what GitHub Actions runs on every push and pull request (`.github/workflows/ci.yml`).
If you change how questions are understood or ranked, run `make battery` before and after the
change and compare the output.

## What a change is expected to carry

- **A test for every new behaviour or fixed bug**, named after the behaviour
  (`test_a_description_mentioning_it_does_not_name_it`). When it comes from a real wrong answer,
  quote the question and the answer in a comment. Tests never use the network; sources are
  recorded responses under `tests/fixtures/`.
- **No claim stronger than its evidence.** Any change touching an availability verdict must hold
  to the rules in section 4 of the developer guide.
- **Named weights.** Ranking weights are named constants at the top of `pipeline/resolve.py`,
  each with a comment giving the case that called for it, never bare numbers inside functions.
- **The surrounding style.** Comments explain *why*, not *what*. User-facing text is Arabic.
- **Commit messages** in English: a first line saying what changed for the user ("Answer …",
  "Stop …", "Find …"), then a paragraph on why when it is not obvious.

## Secrets

Keys belong in the environment or in `.env` (git-ignored), never in code, configuration,
documentation, tests or commit messages. `tests/test_no_secrets.py` fails if anything shaped like
a key appears in a committed file. If a key is committed by mistake, follow
[SECURITY.md](SECURITY.md).

## Reporting a wrong answer

Open an issue from the *Wrong answer* template with the question as typed, the answer as shown
(verdict and source), and what it should have been. These are the most useful reports: every rule
in the code came from an answer like that.
