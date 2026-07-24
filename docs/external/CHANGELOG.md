# Changelog & Versioning Policy

## Versioning policy

Two independent version signals exist on this API — they change for
different reasons and shouldn't be confused:

- **API contract version** — the `/v1` path prefix on the prediction
  endpoints. A breaking change to the request/response contract (e.g.
  removing a field, changing a status code's meaning) would ship as a new
  `/v2` prefix alongside `/v1`, not a silent change to `/v1`'s behavior.
  `/v1` is considered stable: additive changes (new optional fields, new
  endpoints) may happen without a version bump; removals or behavior
  changes will not happen under the same prefix.
- **Build version** — `GET /version` reports the package version
  (semantic versioning: `MAJOR.MINOR.PATCH`) and the exact git commit the
  running deployment was built from. This changes on every release and is
  the right thing to quote when reporting an issue — it identifies exactly
  what code is running, independent of the API contract version above.

## History

### 0.1.0 — initial release

- `POST /v1/predict` — single-image digit classification.
- `POST /v1/predict/batch` — batched digit classification (up to 32
  images per request).
- `GET /health`, `GET /ready`, `GET /version` — status endpoints.
- Optional `X-API-Key` authentication on the prediction endpoints
  (deployment-controlled, off by default).
