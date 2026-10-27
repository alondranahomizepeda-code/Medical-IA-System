# Scoliosis AI Diagnostic System

[![CI/CD Pipeline](https://github.com/alondranahomizepeda-code/scoliosis-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/alondranahomizepeda-code/scoliosis-project/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Test Coverage](https://img.shields.io/badge/coverage-76%25-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/accuracy-85.15%25-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Intelligent clinical decision support system for scoliosis detection, spinal curvature classification, and affected vertebral level localization from coronal spinal radiographs (X-rays). Developed for the **Software Engineering II** course (Team 3).

---

## Key Features

- **Multi-Class Pathological Classification:**
  1. `NO SCOLIOSIS`: Physiologically aligned spine with normal plumb-line tolerance ($<0.042$ coronal deviation).
  2. `SEVERE SCOLIOSIS (C-CURVE)`: Single structural scoliosis curvature with identified apex and coronal convexity.
  3. `SEVERE SCOLIOSIS (S-CURVE)`: Double structural curvature with dual thoracic and lumbar inflection points.
- **Anatomical Vertebral Localization:**
  - Automated tracking of spinal landmarks ($T1-T12$ and $L1-L5$).
  - Apex vertebra identification and affected vertebral range span (e.g., $T7 - T11$, $L1 - L4$).
- **Deep Transfer Learning Architecture:**
  - Convolutional backbone powered by **MobileNetV2** pre-trained on ImageNet and fine-tuned for spinal radiography.
  - Hybrid clinical consensus fusing deep visual features with anatomical plumb-line deviation analysis.
- **Enterprise Security & Traceability:**
  - Multi-factor authentication (2FA) via time-limited one-time tokens.
  - Role-Based Access Control (RBAC) separating Radiologic Technicians, Attending Physicians, and System Administrators.
  - Immutable audit logging with Mexico City clinical timezone synchronization (`America/Mexico_City`).
- **Comprehensive Software Verification (Milestone 3):**
  - **30 automated tests** covering unit, integration, memory profiling (`tracemalloc`), and end-to-end clinical workflow.
  - **76% code coverage** certified across business domains, pipelines, and services.
  - Zero memory leaks detected across 30 sustained diagnostic cycles.
- **DevOps & Cloud-Native Deployment:**
  - Automated 5-stage **GitHub Actions CI/CD pipeline** (Linting, Unit Testing, Docker Multi-Stage Build, Trivy Security Scan, Kubernetes Validation).
  - Production multi-stage `Dockerfile` with non-root security context (`appuser`).
  - Declarative **Kubernetes manifests** (`k8s/`) featuring Horizontal Pod Autoscaling (HPA) from 2 to 10 replicas.

---

## Model Evaluation & Performance Metrics

Evaluated on the independent 101-image held-out test split from the Roboflow Scoliosis Dataset:

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Overall Accuracy** | **85.15%** | 86 of 101 test radiographs correctly diagnosed. |
| **Macro Precision** | **87.22%** | High diagnostic confidence across all pathological categories. |
| **Macro Recall (Sensitivity)** | **85.15%** | Robust detection rate for both healthy and scoliosis patients. |
| **Macro F1-Score** | **85.98%** | Balanced harmonic mean between precision and sensitivity. |
| **S-Curve Precision** | **100.00%** | Zero false alarms between C-curve and S-curve morphologies. |
| **Average E2E Latency** | **581.17 ms** | Sub-second response time compliant with clinical screening standards. |

The detailed quantitative confusion matrix chart is available at [`Codigo/data/confusion_matrix.png`](Codigo/data/confusion_matrix.png) and full evaluation metrics are tracked in [`Codigo/data/models/scoliosis_classifier_v1.0.0/evaluation_results.json`](Codigo/data/models/scoliosis_classifier_v1.0.0/evaluation_results.json).

---

## Project Structure

```text
scoliosis-project/
|-- .github/
|   `-- workflows/
|       `-- ci-cd.yml             # 5-stage GitHub Actions CI/CD Pipeline
|-- Codigo/
|   |-- ai/                       # AI model evaluation, visualization & transfer learning
|   |-- api/                      # FastAPI HTTP endpoints and routers
|   |-- config/                   # Centralized application settings
|   |-- data/                     # Model weights (model.keras), confusion matrix & benchmarks
|   |-- domain/                   # Enterprise interfaces and architectural contracts
|   |-- frontend/                 # Clinical web portal (HTML/CSS/JS)
|   |-- infrastructure/           # SQLite repositories, storage, and model registry
|   |-- k8s/                      # Declarative Kubernetes manifests (HPA, Deployment, Service)
|   |-- pipeline/                 # Image preprocessing pipeline (Fail-Fast quality stages)
|   |-- services/                 # Business logic, inference decorators, and clinical results
|   |-- tests/                    # 30-test automated suite (unit, integration, profiling, e2e)
|   |-- utils/                    # Structured logging and custom exception handlers
|   |-- Dockerfile                # Multi-stage production container definition
|   |-- docker-compose.yml        # Local orchestration service definition
|   |-- requirements.txt          # Python dependencies
|   `-- main.py                   # Application entrypoint
|-- CHANGELOG.md                  # Milestone release history
|-- README.md                     # Project documentation
`-- .gitignore                    # Version control ignore rules
```

---

## Getting Started

### 1. Prerequisites

- Python 3.11 or 3.12
- Git
- (Optional) Docker & Docker Compose for containerized execution

### 2. Local Setup

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/alondranahomizepeda-code/scoliosis-project.git
cd scoliosis-project/Codigo
```

Create and activate a virtual environment:

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install production and testing dependencies:

```bash
pip install -r requirements.txt
```

---

## Execution and Testing Guide

### 1. Run Automated Test Suite & Code Coverage

Execute the 30 automated tests and generate the coverage report:

```bash
cd Codigo
pytest -v --cov=services --cov=pipeline --cov=domain --cov-report=term tests/
```

### 2. Run Model Evaluation & Confusion Matrix Visualization

To recalculate the evaluation metrics across the test dataset:

```bash
cd Codigo
python ai/evaluate_model.py
```

To display the interactive multi-class confusion matrix and clinical metric charts:

```bash
cd Codigo
python ai/plot_confusion_matrix.py
```

### 3. Run Performance & Scalability Benchmark

Benchmark latency breakdown and throughput under concurrent simulation:

```bash
cd Codigo
python tests/profiling/benchmark_performance.py
```

### 4. Start the Application Locally

Launch the FastAPI application server:

```bash
cd Codigo
uvicorn main:app --reload --port 8000
```

Access the interactive API documentation (Swagger UI):
- **Swagger Documentation:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/health`
- **Clinical Frontend Interface:** `http://localhost:8000/`

---

## Containerization & Deployment

### Run with Docker Compose

Build and launch the containerized application with persistent volumes:

```bash
cd Codigo
docker compose up --build -d
```

Check running container status:

```bash
docker compose ps
docker compose logs -f
```

Stop the containerized service:

```bash
docker compose down
```

### Deploy to Kubernetes

Apply the declarative manifests to your cluster:

```bash
kubectl apply -f Codigo/k8s/namespace.yaml
kubectl apply -f Codigo/k8s/
```

Verify pod status and Horizontal Pod Autoscaler:

```bash
kubectl get pods -n scoliosis-system
kubectl get hpa -n scoliosis-system
```

---

## Authors & Academic Credits

- **Course:** Software Engineering II (Ingeniería de Software II)
- **Academic Term:** Quinto Semestre, 2026
- **Team:** Team 3 (Equipo 3)
