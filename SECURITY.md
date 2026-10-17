# Security

## Credentials

Masdar reads its credentials from the environment only, by name:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini, to help read questions (optional) |
| `GROQ_API_KEY` | Groq, to help read questions (optional) |
| `ANTHROPIC_API_KEY` | Claude, to help read questions (optional) |
| `GASTAT_API_KEY` / `GASTAT_API_SECRET` | The GASTAT developer portal |
| `FIRECRAWL_API_KEY` | A Saudi exit for the National Open Data Platform from a server abroad |
| `MASDAR_ACCESS_CODE` | The access code of a hosted page |

- They go in `.env` (git-ignored; `.env.example` lists the names with no values) or in the host's
  settings.
- No key is ever printed in a log or an error message. Logs name the variables only.
- Model keys travel in request headers, never in URLs, so they cannot appear in a log that quotes
  an address.
- `FirecrawlTransport` refuses any request that carries a key, so no source credential passes
  through a third party.
- `tests/test_no_secrets.py` scans every committed file for credential-shaped text.

## What leaves the machine

- **To the official sources:** search words and file requests.
- **To a language model** (if a key is configured): the question text only, and only when the
  rules did not read it completely (`MASDAR_LLM_ROLE`). No file, key or user data is sent. On free
  tiers, the provider may use the text it receives to improve its products.

## If a key is committed by mistake

1. **Revoke it at once** in the provider's console and create a new one. Deleting it from the
   file is not enough: it remains in the git history and in every clone.
2. Put the new key in `.env` or the host's settings.
3. Run `python -m pytest tests/test_no_secrets.py`.

## Reporting a vulnerability

Please do not open a public issue. Contact the repository owner directly through GitHub with a
description and steps to reproduce.
