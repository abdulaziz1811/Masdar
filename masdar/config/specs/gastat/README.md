# GASTAT API specifications

Drop the OpenAPI files downloaded from the GASTAT developer portal here, one
per API product. Every `/v1/stats/{dataset_id}` path in them becomes a
searchable dataset of the `gastat_cdata` source: titles, dimensions, time
axis, measures and topics are all read from the file.

How to get a file: open the API's page in the portal, open the page menu,
choose **تنزيل المواصفات** (Download specification), JSON.

Prefer `masdar import-spec FILE...` over copying by hand: it validates the
file, refuses one that embeds a credential, and names it after its API.

A specification describes *where* a key goes (a security scheme naming the
`apikey` header). It must never contain a key. Keys live in `.env`.
