# Authentication

Authentication on `/v1/predict*` is **optional and controlled by whoever
operates this deployment** — it is not a fixed part of the API contract.
Ask your integration contact whether it's enabled for the deployment you're
using.

## If it's enabled

Send an API key in the `X-API-Key` header on every request to `/v1/predict`
or `/v1/predict/batch`:

```bash
curl -X POST https://<your-deployment-host>/v1/predict \
  -H "X-API-Key: <your-key>" \
  -F "file=@digit.png"
```

| Situation | Response |
|---|---|
| Header missing, a key is required | `401 Unauthorized` |
| Header present but wrong | `401 Unauthorized` |
| Header present and correct | request proceeds normally |

A `401` response body looks like:

```json
{"detail": "Invalid API key"}
```

## If it's not enabled

Omit the header entirely — requests are accepted unauthenticated. This is
the default for a fresh deployment of this template, so don't assume a key
is required unless you've been told one is.

## Key rotation

Key values and rotation policy are entirely up to the operator of your
deployment — there's no self-service key management endpoint in this API.
Contact your integration owner to obtain, rotate, or revoke a key.

## Other endpoints

`GET /health`, `GET /ready`, and `GET /version` are unauthenticated status
endpoints, always open regardless of the `/v1/predict` key setting. A small
number of other operational endpoints (model reload, metrics) exist on this
service but are intended for the platform operator, not API integrators —
they aren't documented here and typically aren't reachable by third
parties.
