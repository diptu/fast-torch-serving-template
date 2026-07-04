
<div align="center">

# 🚀 Fast-torch-serving-template

*A production-ready template for building scalable Machine Learning APIs with FastAPI, PyTorch, Docker, and modern Python tooling.*

<p>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

**Build ML APIs—not boilerplate.**

</div>

---

# ✨ Features

- ⚡ FastAPI backend
- 🤖 Native PyTorch integration
- 🧠 Ready for Deep Learning inference
- 📦 Dependency management using **uv**
- 🐳 Docker & Docker Compose support
- ✅ Pytest configured
- 🔍 Ruff + formatting
- 📊 Health check endpoint
- 📚 Automatic OpenAPI documentation
- 🔐 Environment-based configuration
- 📝 Structured logging
- 🚀 Production-ready project structure
- 🔄 GitHub Actions ready
- 📈 Easily extendable for training or inference services

---

# 📂 Project Structure

```text
.
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── dependencies.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── ml/
│   │   ├── models/
│   │   ├── inference/
│   │   ├── datasets/
│   │   └── utils/
│   │
│   ├── utils/
│   └── main.py
│
├── tests/
│
├── docker/
│
├── scripts/
│
├── .github/
│   └── workflows/
│
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| API | FastAPI |
| ML Framework | PyTorch |
| Validation | Pydantic v2 |
| ASGI Server | Uvicorn |
| Package Manager | uv |
| Testing | Pytest |
| Linting | Ruff |
| Containerization | Docker |
| Documentation | OpenAPI / Swagger |
| Configuration | pydantic-settings |

---

# 🚀 Quick Start

## Clone

```bash
git clone https://github.com/yourusername/fastapi-pytorch-starter.git

cd fastapi-pytorch-starter
```

---

## Install Dependencies

```bash
uv sync
```

---

## Activate Environment

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

---

## Run Development Server

```bash
uv run uvicorn app.main:app --reload
```

Application

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# 🐳 Docker

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Run in background

```bash
docker compose up -d
```

Stop

```bash
docker compose down
```

---

# 🧪 Testing

```bash
pytest
```

or

```bash
uv run pytest
```

---

# 🎨 Code Quality

Lint

```bash
ruff check .
```

Format

```bash
ruff format .
```

---

# ⚙️ Environment Variables

Create

```bash
cp .env.example .env
```

Example

```env
APP_NAME=FastAPI Starter
APP_ENV=development
DEBUG=true

HOST=0.0.0.0
PORT=8000

LOG_LEVEL=INFO

MODEL_PATH=models/model.pt
```

---

# 🧠 Machine Learning Workflow

```text
Request
   │
   ▼
FastAPI Endpoint
   │
   ▼
Validation
   │
   ▼
Service Layer
   │
   ▼
PyTorch Model
   │
   ▼
Inference
   │
   ▼
Response
```

---

# 📈 Roadmap

- [ ] GPU support
- [ ] ONNX Runtime
- [ ] TensorRT
- [ ] MLflow integration
- [ ] Celery workers
- [ ] Redis cache
- [ ] Prometheus metrics
- [ ] OpenTelemetry
- [ ] Kubernetes deployment
- [ ] Model versioning
- [ ] JWT Authentication
- [ ] Background tasks

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit

```bash
git commit -m "Add awesome feature"
```

4. Push

```bash
git push origin feature/my-feature
```

5. Open a Pull Request

---

# 📜 License

Distributed under the MIT License.

See `LICENSE` for more information.

---

<div align="center">

Made with ❤️ for the Machine Learning & FastAPI community.

If this project helped you, consider giving it a ⭐

</div>
