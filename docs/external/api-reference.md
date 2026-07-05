# API Reference

All endpoints are served from the base URL of your deployment. There is no
separate staging/production hostname convention enforced by the API itself
— that's a per-deployment concern.

Every response includes an `X-Request-ID` header (a UUID, or the value you
sent in your own `X-Request-ID` request header, echoed back). Include it
when contacting support about a specific request.

Interactive schema (if enabled for your deployment — see
[Authentication](./authentication.md) for why it might not be):
`GET /docs` (Swagger UI), `GET /redoc`, `GET /openapi.json`.

---

## `POST /v1/predict`

Classify a single handwritten digit image.

**Auth:** `X-API-Key` header, if required by your deployment.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | One image. Content-Type must be `image/png`, `image/jpeg`, or `image/jpg`. Max 5MB. |

**Response:** `200 OK`

```json
{
  "predicted_digit": 7,
  "confidence": 0.9842,
  "probabilities": [0.0001, 0.0002, 0.0004, 0.0011, 0.0003,
                     0.0009, 0.0002, 0.9842, 0.0018, 0.0008]
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_digit` | integer, 0–9 | The model's top prediction |
| `confidence` | float, 0.0–1.0 | Probability assigned to `predicted_digit` |
| `probabilities` | array of 10 floats | Full distribution, index = digit |

**Error responses:** see [Errors](./errors.md) — this endpoint can return
`400`, `401`, `413`, `415`, `422`, `503`.

---

## `POST /v1/predict/batch`

Classify multiple handwritten digit images in a single request. This runs
as one batched inference call server-side, not a loop over single
predictions — meaningfully more efficient than calling `/v1/predict`
repeatedly for the same set of images.

**Auth:** `X-API-Key` header, if required by your deployment.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `files` | file (repeated) | yes | 1–32 images. Same content-type/size rules as `/v1/predict`, applied per file. |

Send multiple files under the same `files` field name (standard
multipart repetition — see the curl/Python examples in
[Getting Started](./getting-started.md#3-classify-a-batch)).

**Response:** `200 OK`

```json
{
  "predictions": [
    {"predicted_digit": 7, "confidence": 0.9842, "probabilities": [...]},
    {"predicted_digit": 3, "confidence": 0.8811, "probabilities": [...]}
  ]
}
```

`predictions` is the same shape as a single `/v1/predict` response, one
entry per uploaded file, **in the same order you uploaded them**.

**Error responses:** see [Errors](./errors.md) — this endpoint can return
`400`, `401`, `413`, `415`, `422`, `503`. A `400` here can also mean "more
than 32 files were sent."

---

## `GET /health`

Liveness check: is the process up and responding at all.

**Auth:** none.

**Response:** always `200 OK`

```json
{"status": "ok", "model_loaded": true}
```

`model_loaded: false` means the process is healthy but has no trained
model yet — it will still respond to `/v1/predict`, just with predictions
from an untrained model. Use this endpoint for uptime checks; use
`GET /ready` (below) to check whether it's actually ready to serve useful
predictions.

## `GET /ready`

Readiness check: is this instance able to serve *useful* predictions right
now.

**Auth:** none.

**Response:**

- `200 OK` — `{"ready": true}`
- `503 Service Unavailable` — no model checkpoint is loaded yet

## `GET /version`

Reports what build is running.

**Auth:** none.

**Response:** `200 OK`

```json
{"version": "0.1.0", "git_sha": "75d14d3"}
```

`git_sha` is `"unknown"` for a deployment built without commit info
attached — that's an operational detail of how it was built, not a sign of
a problem.

---

## Limits summary

| Limit | Value |
|---|---|
| Max upload size | 5 MB per file |
| Max batch size | 32 files per `/v1/predict/batch` request |
| Allowed content types | `image/png`, `image/jpeg`, `image/jpg` |
| Concurrency | Bounded per-deployment; exceeding it returns `503` with a `Retry-After` header rather than queuing indefinitely — see [Errors](./errors.md#503-service-unavailable) |

These are the current defaults for this template; your specific deployment
may have adjusted them. If a limit here doesn't match what you're seeing,
trust the deployment's actual behavior and check with your integration
contact.
