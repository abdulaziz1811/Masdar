# Changelog

Notable changes for users, newest first. Details are in the commit messages; decisions and their
reasons in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Unreleased

### Understanding questions
- **Free models that cover for each other:** Gemini Flash, then Flash-Lite (same key), then
  `openai/gpt-oss-120b` on Groq (`GROQ_API_KEY`), then Claude. A model that is busy or out of
  quota is rested and the next is asked, within 12 seconds per question (`nlu/chain.py`).
- **The model assists, it does not replace:** it is also consulted when the rules leave words of
  a question unread, and adds publisher-style wordings that widen the search, without ever
  deciding that a result is about the question (`MASDAR_LLM_ROLE`).
- **Alternative phrasings** from the model are offered as one-click suggestions when nothing is
  found.
- **Relative periods:** «آخر 3 سنوات», «العام الماضي», «قبل سنتين».
- **New everyday vocabulary:** pharmacies and nursing, tourists and accommodation, wheat and
  holdings, remote work and job seekers, scholarship students and graduates, marriage contracts,
  traffic accidents.
- A topic read by a model needs one of its phrases in the result.

### Repository
- English README; the full Arabic guide moved to `docs/README.ar.md`.
- `CONTRIBUTING.md`, `SECURITY.md`, issue and pull-request templates, `.editorconfig` and
  `.pre-commit-config.yaml`.

## 2026-10-17
- No «available» answer from a result whose description mentions the subject but whose title
  does not.
- «حسب المناطق» answered with a national total is *partial*; a year not yet begun is
  *not available*.
- GCC-Stat added as a regional source.
- Developer guide, and CI on GitHub (ruff and the test suite).

## 2026-10-03 – 2026-10-04
- GASTAT bulletin workbooks read as a statistician would, fixing seven faults that reported
  published data as «غير متوفرة».

## 2026-09-27 – 2026-09-30
- GASTAT (statistics API, database and website search), the National Open Data Platform,
  KAPSARC, the World Bank, and live ministry indicators.
- Official chat interface, hosting on Render or Cloud Run, and the presenter guide.
- Optional understanding by a language model (Claude, then Gemini) for questions outside the
  lexicon.
- Excel export with a provenance sheet, the six-verdict logic, and saved copies when a source is
  down.
