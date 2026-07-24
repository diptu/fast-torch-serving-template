# Errors

Error responses are JSON with at least a `detail` field describing what
went wrong:

```json
{"detail": "Unsupported content type: text/plain"}
```

Every response — success or error — carries an `X-Request-ID` header.
Include it when reporting an issue.

## Status codes

### `400` Bad Request

The request was well-formed but the content was invalid. On this API, that
means either:
- an uploaded file isn't a decodable image, or
- (batch endpoint only) more than 32 files were sent in one request.

**Action:** fix the input and retry — retrying the same request unchanged
will fail the same way.

### `401` Unauthorized

Only possible if your deployment requires an API key (see
[Authentication](./authentication.md)). The `X-API-Key` header was missing
or didn't match.

**Action:** check the header is present and correct.

### `413` Content Too Large

The uploaded file exceeded the 5MB per-file limit.

**Action:** downscale/recompress the image before uploading. A single
handwritten digit doesn't need to be large — a few hundred KB is typical.

### `415` Unsupported Media Type

The upload's `Content-Type` wasn't `image/png`, `image/jpeg`, or
`image/jpg`.

**Action:** confirm the client is setting the correct content type for the
file field (most HTTP libraries infer this from the file extension/mimetype
automatically — double check if you're constructing the multipart body by
hand).

### `422` Unprocessable Entity

Standard FastAPI request-validation failure — the request didn't match the
expected shape at all (e.g. the `file`/`files` field was missing entirely,
or an empty file list was sent to the batch endpoint). The response body
follows FastAPI's standard validation-error format, with a `detail` array
describing which field(s) failed and why.

**Action:** check the field name and that at least one file is attached.

### `503` Service Unavailable

The service is at capacity — too many predictions are already in flight.
This response includes a `Retry-After` header (seconds).

```json
{"detail": "Server is at capacity, please retry shortly"}
```

**Action:** back off for the duration in `Retry-After` and retry. This is
an expected, transient condition under load, not a bug — a well-behaved
client should retry automatically (with backoff) rather than surface this
as a hard failure to an end user.

`GET /ready` can also return `503`, with a different meaning: it means no
model checkpoint is loaded, not "at capacity." See
[the API reference](./api-reference.md#get-ready).

### `500` Internal Server Error

An unexpected failure. The response body includes a `request_id`:

```json
{"detail": "Internal server error", "request_id": "b3f1c2..."}
```

**Action:** retry once; if it persists, report it to your integration
contact along with the `request_id` (or the `X-Request-ID` response
header, which carries the same value) so they can find the corresponding
server-side log entry.
