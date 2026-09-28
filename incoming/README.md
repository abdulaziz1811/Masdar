# Upload the portal specs here

Upload the `swagger*.json` files downloaded from the GASTAT developer portal
into this folder (Add file → Upload files). They are imported into
`masdar/config/specs/gastat/` with `masdar import-spec incoming/*.json`, which
validates each file, refuses any that embeds a credential, names it after its
API, and replaces any hand-reconstructed spec it supersedes. This folder is
then emptied again.

A specification says *where* an API key goes, never what it is. Keys live in
`.env` on your own machine and never in this repository.
