# Scoliosis AI Diagnostic System

Sistema clínico inteligente para la detección y clasificación de Escoliosis y análisis de curvatura vertebral a partir de radiografías de columna (Rayos X).
Desarrollado para la materia de **Ingeniería de Software II** (Equipo 3).

---

## Características Principales
- **Clasificación en 3 categorías:**
  1. `NO ESCOLIOSIS`: Columna alineada sin desviación patológica.
  2. `ESCOLIOSIS GRAVE (EN C)`: Curvatura simple en forma de C.
  3. `ESCOLIOSIS SEVERA (EN S)`: Doble curvatura patológica en forma de S.
- **Detección de Vértebras Afectadas:** Identificación anatómica de vértebras comprometidas (ej. T7 - T11, L1 - L4) y ápex de la curva.
- **Transfer Learning:** Red neuronal profunda basada en MobileNetV2 con fine-tuning.
- **Seguridad y Trazabilidad:** Autenticación de doble factor (2FA), control de acceso basado en roles (RBAC) y registro inmutable de auditoría.
- **Generación Documental:** Reportes médicos estructurados en PDF (localizados en hora de México `America/Mexico_City`).
- **Pruebas y Validación (Hito 3):** Suite con 30 pruebas automatizadas (unitarias, integración, perfilamiento de memoria y sistema E2E) con 76% de cobertura de código.
- **CI/CD y Contenedorización:** Pipeline de GitHub Actions con 5 etapas, Docker multi-etapa y manifiestos de Kubernetes con autoescalado HPA.

---

## Documentación del Hito 3
El entregable formal completo con la justificación de cada criterio de la rúbrica se encuentra en:
- [Documentación/Entrega_Final_Hito_3.md](file:///c:/Users/Nahomi/Desktop/Quinto%20Semestre/Software%20Engineering%20II/Proyecto/Documentaci%C3%B3n/Entrega_Final_Hito_3.md)

---

## Comandos de Ejecución y Validación

### 1. Activar el Entorno Virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Ejecutar la Suite Completa de Pruebas (30 pruebas + Cobertura)
```powershell
cd Codigo
pytest -v --cov=services --cov=pipeline --cov=domain --cov-report=term tests/
```

### 3. Ejecutar la Validación del Modelo de IA y Matriz de Confusión
```powershell
cd Codigo
python ai/evaluate_model.py
```
*Genera el reporte cuantitativo en `data/models/scoliosis_classifier_v1.0.0/evaluation_results.json`.*

### 4. Ejecutar el Benchmark de Rendimiento y Escalabilidad
```powershell
cd Codigo
python tests/profiling/benchmark_performance.py
```
*Genera las métricas de latencia, throughput y cuellos de botella en `data/performance_benchmark_results.json`.*

### 5. Iniciar la Aplicación en Modo Local
```powershell
cd Codigo
uvicorn main:app --reload --port 8000
```

### 6. Despliegue con Docker Compose
```powershell
cd Codigo
docker-compose up --build -d
```

### 7. Despliegue en Clúster de Kubernetes
```powershell
kubectl apply -f Codigo/k8s/namespace.yaml
kubectl apply -f Codigo/k8s/
```
