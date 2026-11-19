# Changelog

All notable changes to the Scoliosis AI Diagnostic System are documented in this file.

## [3.2.0] - Milestone 3 Consolidated: Model v1.1.0 Validation, Memory Optimization & Hardened RBAC
- **AI Model Iteration `v1.1.0` & Quality Gate (Criterion 3 — Questions 1, 2, 9 & 10):**
  - Fixed the input scaling mismatch in `MobileNetV2` by replacing raw `preprocess_input` with an in-model `Rescaling(2.0, offset=-1.0)` layer that maps `[0.0, 1.0] -> [-1.0, 1.0]`.
  - Added a spatial $7\times 7$ spinal curvature bottleneck alongside `GlobalAveragePooling2D` and inverse-frequency class weights ($w_k = \frac{N}{K \cdot N_k}$).
  - Implemented `Codigo/ai/metrics.py` with `compute_majority_baseline`, `compute_calibration_summary`, and `passes_quality_gate` ($\text{Macro-F1} \ge 0.70$, per-class sensitivity $\ge 0.60$, strictly outperforming the $44.55\%$ majority-class baseline).
  - Trained, evaluated, and promoted `scoliosis_classifier v1.1.0` on the 101 held-out test radiographs (`88.12%` accuracy, `87.20%` balanced accuracy, `88.07%` macro-F1, `approved: true`), while preserving `v1.0.0` (`44.55%` accuracy, `33.33%` balanced accuracy, `approved: false`) as historical baseline evidence.
- **Expanded Automated Test Suite (Criterion 2 — Question 4):**
  - Expanded the test suite to **104 automated tests** (unit, integration, concurrency, memory profiling, system E2E) achieving **>82% code coverage**.
  - Added `test_critical_medical_non_collapse_and_sensitivity_probe_question_4` to detect mode collapse and verify multi-class responsiveness.
- **Server-Side Security, 2FA & Strict RBAC (Question 5):**
  - Persisted 6-digit one-time codes (`2FA` and password recovery) as `HMAC-SHA256` digests in the `one_time_codes` table with 5-minute expiration, 5-attempt lockout, and 3-resend limit.
  - Enforced server-side RBAC on every REST endpoint (`HTTP 403 Forbidden` when a `technician` attempts to confirm a diagnosis via `POST /api/v1/results/{id}/review`, view clinical history, or access audit logs).
- **Memory Profiling & Data-Type Optimization (Question 6):**
  - Replaced `int64`/`float64` temporary arrays in `ValidationStage` and `QualityCheckStage` with signed `int16` and `float32`, reducing peak memory on $1024\times 1024$ radiographs by ~70% (`67 MB -> ~19.5 MB`) while preventing `uint8` wrap-around and preserving numerical precision ($< 10^{-4}$ vs `float64`).
  - Replaced unbounded latency list in `MetricsInferencerDecorator` with a bounded ring buffer `deque(maxlen=1000)`.
- **CI/CD, Containers & Kubernetes Orchestration (Criteria 4, 5 & 6 — Questions 7 & 8):**
  - Configured blocking Trivy `CRITICAL` CVE gate (`exit-code: "1"`) and `--cov-fail-under=75` in `.github/workflows/ci-cd.yml`.
  - Completed all 10 declarative Kubernetes manifests in `Codigo/k8s/` including `StatefulSet` PostgreSQL 16, `ReadWriteMany` PVC, `/ready` readiness gating (`HTTP 503` without an approved model), `HPA` (2–6 replicas), and `PodDisruptionBudget`.

## [3.0.0 / 3.1.0] - Milestone 3: Testing, Validation, and Deployment
- Initial 5-stage GitHub Actions CI/CD workflow, multi-stage non-root `Dockerfile`, performance benchmarking (`@tf.function` speedup $89.4\text{ ms} \rightarrow 21.8\text{ ms}$), and baseline `v1.0.0` evaluation.

## [2.0.0] - Milestone 2: Scoliosis & Vertebral Analysis
- Fine-tuned MobileNetV2 for 3-class spinal morphology classification, automated vertebral landmark tracking ($T1\text{–}L5$), localized bilingual PDF reports (`America/Mexico_City`), 2FA, and RBAC.

## [1.0.0] - Milestone 1: Core Modular Architecture
- Modular monolith architecture with domain contracts (`ABC`), Dependency Inversion (`DIP`), Fail-Fast preprocessing pipeline, and GoF Strategy/Factory/Decorator patterns.

## [3.2.1] - Milestone 4: Ethical, Legal, Social Hardening & Final Consistency
- Fail-safe hardening: PDF Unicode/long-token sanitizer, compensating disk cleanup on DB failure, query-filter and pagination validation, review-status validation, empty-file rejection, keyboard-mash protection (auto-repeat shield, in-flight lock, maxlength), thread-safe inference (`RLock` + `test_concurrent_inference_thread_safety`), filename truncation.
- Standardized all clinical class labels to English (`NO SCOLIOSIS`, `SERIOUS SCOLIOSIS (C-CURVE)`, `SEVERE SCOLIOSIS (S-CURVE)`), unified system version to `3.2.1`, removed `k8s/secret.yaml` and runtime DB/patient files from clean distribution packages, and verified full 5-stage CI/CD pass on GitHub Actions (Run #18: https://github.com/alondranahomizepeda-code/Medical-IA-System/actions/runs/44419222602, **104 tests passing**).
