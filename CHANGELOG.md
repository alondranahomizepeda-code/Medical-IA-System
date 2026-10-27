# Changelog

## [3.0.0] - Milestone 3: Testing, Validation, and Deployment of AI Systems
- **Estrategia Integral de Pruebas (2.0 pts):**
  - Implementación de 30 pruebas automatizadas en `Codigo/tests/` (unitarias, integración, perfilamiento de memoria y sistema E2E).
  - Cobertura de código certificada al 76% sobre 1029 líneas de código en servicios, pipeline y dominio.
  - Perfilamiento de memoria con `tracemalloc` certificando ausencia de fugas de memoria (<500 KB) tras 30 iteraciones continuas.
- **Validación del Modelo de IA y Análisis de Errores (2.0 pts):**
  - Creación y ejecución de `Codigo/ai/evaluate_model.py` sobre 101 imágenes radiológicas del conjunto de prueba (Roboflow dataset).
  - Cálculo de métricas: Exactitud (44.55%), Macro F1 (20.55%), Matriz de Confusión 3x3 y desglose clínico de 56 casos de falla con causas de sesgo por desbalance.
- **Pipeline de CI/CD (1.5 pts):**
  - Flujo de GitHub Actions en `.github/workflows/ci-cd.yml` con 5 etapas: Linting (Flake8/Bandit), Pruebas y Cobertura, Construcción Docker y Escaneo Trivy, Validación de Manifiestos K8s, Despliegue continuo a Staging.
- **Contenedorización y Orquestación (1.5 pts):**
  - `Dockerfile` multi-etapa de producción con usuario no privilegiado (`medicaluser`) y healthcheck.
  - `docker-compose.yml` para despliegue local con volúmenes persistentes.
  - Manifiestos declarativos de Kubernetes en `Codigo/k8s/`: Namespace, ConfigMap, Secret, PVC (20GB), Deployment con RollingUpdates y sondas, Service (ClusterIP) y HPA (Autoescalado horizontal 2 a 10 réplicas).
- **Análisis de Rendimiento y Escalabilidad (1.5 pts):**
  - Script `Codigo/tests/profiling/benchmark_performance.py` midiendo throughput (1.72 req/s) y latencia media (581.17 ms).
  - Identificación del forward pass de la red neuronal como cuello de botella primario (44.10% del tiempo total).
  - Justificación de optimizaciones: Cuantización INT8/TensorRT, colas asíncronas para reportes PDF y autoescalado con HPA.
- **Correcciones del Hito 2 (1.0 pt) y Evidencia Técnica (0.5 pt):**
  - Documento formal exhaustivo en `Documentación/Entrega_Final_Hito_3.md` resolviendo los 7 puntos de retroalimentación del Hito 2.

## [2.0.0] - Scoliosis & Vertebral Analysis
- Reentrenamiento del sistema AI para clasificación de Escoliosis usando Transfer Learning (MobileNetV2).
- Soporte para 3 clases clínicas: NO ESCOLIOSIS, ESCOLIOSIS GRAVE (EN C), ESCOLIOSIS SEVERA (EN S).
- Detección anatómica de vértebras afectadas (ápex y extensión de la curva espinal, e.g., T7 - T11, L1 - L4).
- Reportes médicos PDF y TXT actualizados con hallazgos vertebrales y curvatura.
- Interfaz gráfica actualizada con métricas de escoliosis, badges de severidad y visualización por roles (RBAC).

## [1.0.0] - Milestone 2
- Integración dinámica de modelos AI usando Transfer Learning.
- Reestructuración de patrones de diseño de vuelta a la capa de dominio/servicios.
- Sistema de auditoría, 2FA integrado.
