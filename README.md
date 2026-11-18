# Scoliosis AI Diagnostic System

[![CI/CD Pipeline](https://github.com/alondranahomizepeda-code/Medical-IA-System/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/alondranahomizepeda-code/Medical-IA-System/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-102%20passed-brightgreen.svg)]()
[![Test Coverage](https://img.shields.io/badge/coverage-%E2%89%A579%25-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/accuracy-88.12%25-success.svg)]()
[![Version](https://img.shields.io/badge/version-3.2.0-informational.svg)]()

Intelligent clinical decision support system (CDSS) for scoliosis detection, spinal curvature classification, and affected vertebral level localization from coronal spinal radiographs (X-rays). Developed for the **Software Engineering II** course (Team 3).

> **Clinical safety notice:** this system is a *second-opinion decision-support tool*. Every prediction starts as `pending_review` and must be confirmed or corrected by a licensed radiologist. It does not replace professional medical judgment.

---

## Key Features

- **Multi-Class Pathological Classification:**
  1. `NO ESCOLIOSIS` (Normal): physiologically aligned spine.
  2. `ESCOLIOSIS GRAVE (EN C)`: single structural curvature with identified apex.
  3. `ESCOLIOSIS SEVERA (EN S)`: double structural curvature with thoracic and lumbar inflection points.
- **Anatomical Vertebral Localization:** automated tracking of spinal landmarks (T1-T12, L1-L5), apex vertebra and affected vertebral range (e.g. `T7 - T11`).
- **Deep Transfer Learning:** **MobileNetV2** (ImageNet) fine-tuned for spinal radiography, with in-graph `Rescaling(2.0, offset=-1.0)`, a 7x7 spatial curvature head and inverse-frequency class weights.
- **Human-in-the-Loop Safety:** `requires_review` flag when confidence < 60% or when the CNN and the geometric estimator disagree; radiologists can **Confirm** or **Correct** every result.
- **Security & Traceability:**
  - `bcrypt` password hashing, JWT `HS256`, and two-factor authentication (6-digit codes stored as `HMAC-SHA256` digests with 5-minute expiry and 5-attempt lockout).
  - Server-side Role-Based Access Control (`technician`, `radiologist`, `ml_engineer`, `admin`) returning `HTTP 403` on unauthorized calls.
  - DICOM PHI anonymization and an immutable append-only audit log (`America/Mexico_City` timezone).
- **Fail-Fast Input Validation:** rejects non-radiographic images, color photos, empty/corrupt files, oversized uploads (`HTTP 413`) and unsupported formats (`HTTP 422`); supports JPG, PNG, TIFF and 16-bit DICOM.
- **Model Registry & Quality Gate:** only approved model versions (macro-F1 >= 0.70, per-class sensitivity >= 0.60) can serve traffic; otherwise `/ready` returns `HTTP 503`.
- **Bilingual (EN/ES)** web portal and PDF clinical reports, light/dark mode.
- **DevOps:** 5-stage GitHub Actions pipeline (lint/SAST, tests with coverage gate, Docker build + Trivy, Kubernetes validation, staging deploy), non-root multi-stage `Dockerfile`, and 12 Kubernetes manifests (HPA 2-6 replicas, PDB, PostgreSQL, Ingress).

---

## Model Evaluation (`scoliosis_classifier v1.1.0`)

Evaluated on the independent 101-radiograph held-out test split (33 Normal, 45 C-curve, 23 S-curve):

| Metric | `v1.0.0` (rejected) | `v1.1.0` (approved, active) |
| :--- | :---: | :---: |
| **Overall accuracy** | 44.55% | **88.12%** (89/101) |
| **Balanced accuracy** | 33.33% | **87.20%** |
| **Macro F1-score** | 20.55% | **88.07%** |
| Sensitivity - Normal | 0.00% | 87.88% (29/33) |
| Sensitivity - C-curve | 100.00% | 91.11% (41/45) |
| Sensitivity - S-curve | 0.00% | 82.61% (19/23) |
| Quality gate | `approved: false` | `approved: true` |

Confusion matrix (`v1.1.0`): `[[29, 4, 0], [2, 41, 2], [0, 4, 19]]` - no Normal case was classified as S-curve and vice versa.

Artifacts: [`data/confusion_matrix.png`](data/confusion_matrix.png) and [`data/models/scoliosis_classifier_v1.1.0/evaluation_results.json`](data/models/scoliosis_classifier_v1.1.0/evaluation_results.json).

---

## Project Structure

```text
.
|-- .github/workflows/ci-cd.yml   # 5-stage CI/CD pipeline
|-- ai/                           # Transfer learning, training, evaluation, quality-gate metrics
|-- api/                          # FastAPI routes (RBAC, 2FA, upload, classify, review, reports, audit)
|-- config/                       # Centralized settings
|-- data/                         # Model versions, confusion matrix, benchmarks
|-- domain/                       # Entities and abstract contracts (DIP)
|-- frontend/                     # Clinical web portal (HTML/CSS/JS, EN/ES)
|-- infrastructure/               # Repositories, storage adapter, model registry
|-- k8s/                          # 12 Kubernetes manifests (kustomize)
|-- pipeline/                     # Fail-fast preprocessing stages
|-- services/                     # Auth, image mgmt, inference (Strategy/Decorator), clinical results, audit
|-- tests/                        # 102 automated tests (unit, integration, profiling, system)
|-- utils/                        # Logging and exception hierarchy
|-- Dockerfile / docker-compose.yml
|-- requirements.txt
`-- main.py                       # Composition root / entrypoint
```

---

## Getting Started

```bash
git clone https://github.com/alondranahomizepeda-code/Medical-IA-System.git
cd Medical-IA-System
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --port 8000
```

- Web portal: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Liveness / readiness: `http://localhost:8000/health` and `http://localhost:8000/ready`

Default demo users (password `password123`): `radiologist`, `technician`, `admin`, `ml_engineer`. The 2FA code is printed in the server console after login.

### Tests, evaluation and benchmark

```bash
pytest -v --cov=. --cov-fail-under=75 tests/      # 102 tests
python ai/evaluate_model.py                       # recompute metrics on the test split
python ai/plot_confusion_matrix.py                # confusion matrix chart
python tests/profiling/benchmark_performance.py   # latency / throughput benchmark
```

Mean end-to-end CPU latency is ~153 ms per radiograph (RNF-01 target: < 3 s).

---

## Containerization & Deployment

```bash
docker compose up --build -d      # API on http://localhost:8000
docker compose down
```

```bash
kubectl apply -k k8s/             # namespace: medical-system
kubectl get pods,hpa -n medical-system
```

---

## Authors & Academic Credits

- **Course:** Software Engineering II
- **Academic Term:** Fifth Semester, 2026B
- **Team 3:** Aguilar, Hernandez, Gutierrez, Salinas, Zepeda
