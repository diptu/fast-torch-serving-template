# Getting Started

## 1. Confirm the API is reachable

```bash
curl https://<your-deployment-host>/health
```

```json
{"status": "ok", "model_loaded": true}
```

If `model_loaded` is `false`, the service is up but hasn't been given a
trained model yet — predictions will still return a response, just not a
meaningfully accurate one. Ask whoever operates this deployment to confirm
a model has been trained/loaded.

## 2. Classify a digit image

Send a single handwritten-digit image (PNG or JPEG, square works best) as a
`multipart/form-data` upload:

```bash
curl -X POST https://<your-deployment-host>/v1/predict \
  -F "file=@digit.png"
```

```json
{
  "predicted_digit": 7,
  "confidence": 0.9842,
  "probabilities": [0.0001, 0.0003, ..., 0.9842, ...]
}
```

`probabilities` is always length 10, index `i` is the model's probability
for digit `i` — `probabilities[predicted_digit] == confidence`.

## 3. Classify a batch

Send multiple images in one request (up to 32 — see
[API Reference](./api-reference.md#post-v1predictbatch)):

```bash
curl -X POST https://<your-deployment-host>/v1/predict/batch \
  -F "files=@digit1.png" \
  -F "files=@digit2.png"
```

```json
{
  "predictions": [
    {"predicted_digit": 7, "confidence": 0.9842, "probabilities": [...]},
    {"predicted_digit": 3, "confidence": 0.8811, "probabilities": [...]}
  ]
}
```

`predictions` is in the same order as the uploaded files.

## Python example

```python
import requests

with open("digit.png", "rb") as f:
    resp = requests.post(
        "https://<your-deployment-host>/v1/predict",
        files={"file": f},
        # Only needed if this deployment requires it — see authentication.md
        # headers={"X-API-Key": "your-key"},
        timeout=10,
    )
resp.raise_for_status()
print(resp.json())
```

## Next steps

- If requests start returning `401`, see [Authentication](./authentication.md).
- If requests return anything other than `200`, see [Errors](./errors.md) —
  in particular, a `503` with a `Retry-After` header means "retry shortly,"
  not "something is broken."
- For every field, limit, and status code in detail, see the
  [API Reference](./api-reference.md).
