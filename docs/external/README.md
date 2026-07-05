# Fast Torch Serving API — External Documentation

Documentation for third parties integrating with this API: an MNIST
handwritten-digit classification service built on FastAPI + PyTorch.

This is the **external-facing** doc set — it covers only the public HTTP
API contract (endpoints, auth, request/response formats, errors). It does
not describe internal implementation, deployment, or operations; those live
in the project's main `README.md`, `CLAUDE.md`, and `TODO.md`, which are not
intended for distribution outside the team running this service.

## Contents

| Doc | What it covers |
|---|---|
| [Getting Started](./getting-started.md) | Minimal working request in under a minute (curl + Python) |
| [Authentication](./authentication.md) | The `X-API-Key` header, and when it's actually required |
| [API Reference](./api-reference.md) | Every endpoint: request/response schemas, examples, limits |
| [Errors](./errors.md) | Every HTTP status code the API returns and what it means |
| [Changelog](./CHANGELOG.md) | Version history and the versioning policy |

## At a glance

- **Base URL**: provided by whoever operates this deployment — there is no
  fixed public host, since this is a self-hosted template.
- **Versioning**: the prediction API is path-versioned at `/v1/...`. The
  running deployment's build version and commit are available at
  `GET /version`.
- **Format**: requests are `multipart/form-data` (image uploads); responses
  are JSON.
- **Auth**: optional, per-deployment. See [Authentication](./authentication.md).
- **Support**: every response carries an `X-Request-ID` header — include it
  when reporting an issue to whoever operates this API, so they can find the
  matching server-side log entry.
