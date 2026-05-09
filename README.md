# LLM Optimal Segmentation

Sistema de segmentación óptima de textos para procesamiento con Modelos de Lenguaje Grande (LLMs). El objetivo es dividir un texto en segmentos que respeten las restricciones de ventana de contexto del LLM (tokens mínimos y máximos por segmento), minimizando simultáneamente el costo computacional cuadrático y maximizando la coherencia semántica entre oraciones consecutivas.

El núcleo del sistema es un algoritmo de **Programación Dinámica (DP)** que garantiza la solución óptima global, a diferencia de enfoques greedy que solo aseguran óptimos locales. Se implementan además variantes como DP 2D (para encontrar el número óptimo de segmentos k\*), DP con solapamiento y ventana deslizante, para comparación exhaustiva. Un módulo de calibración estima parámetros óptimos analíticamente y valida la coherencia entre el k\* teórico y el k\* encontrado por DP.

El proyecto incluye una interfaz web interactiva (Streamlit), una CLI completa y una API Python pública, todo respaldado por una suite de 113+ tests.

> **Tesis:** Universidad Mayor de San Andrés (UMSA) — 2025

---

## Requisitos

| Componente | Versión mínima |
|-----------|----------------|
| Python | 3.11+ |
| uv | 0.4+ |
| Ollama | cualquier versión reciente |
| Modelo LLM local | `gemma2:2b` |
| Modelo spaCy | `en_core_web_sm` |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd llm-optimal-segmentation

# 2. Crear entorno e instalar dependencias con uv
uv sync

# 3. Activar el entorno virtual
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 4. Descargar el modelo de spaCy
python -m spacy download en_core_web_sm

# 5. Instalar y arrancar Ollama (si no lo tienes)
# https://ollama.com/download
ollama pull gemma2:2b
```

---

## Correr los tests

```bash
pytest tests/ -v
```

---

## Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. El panel lateral permite ajustar todos los parámetros interactivamente. El botón **Segmentar** corre los 4 métodos de segmentación; el botón **Evaluar con LLM real (lento)** ejecuta inferencia con Ollama por separado.

---

## CLI — Ejemplos de uso

### Segmentar con un método específico

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method dp
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method baseline
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method dp2d
```

Métodos disponibles: `dp`, `baseline`, `sliding_window`, `overlap`, `dp2d`.

### Comparar DP vs Baseline

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 compare
```

### Reporte completo de todos los métodos

```bash
# Sin evaluación LLM (rápido)
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 --fixed-cost 100 report

# Con evaluación LLM real (requiere Ollama)
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 --fixed-cost 100 report --llm
```

### Calibrar parámetros automáticamente

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 --fixed-cost 100 calibrate
```

Imprime el λ sugerido, el k\* analítico y valida la calibración contra el k\* del DP 2D.

### Flags globales

| Flag | Default | Descripción |
|------|---------|-------------|
| `--text` | (requerido) | Texto a segmentar |
| `--lmin` | 20 | Tokens mínimos por segmento |
| `--lmax` | 150 | Tokens máximos por segmento |
| `--overlap` | 20 | Overlap para sliding window |
| `--model` | `gpt-4o` | Tokenizador (gpt-4o / gpt-4 / gpt-3.5-turbo) |
| `--fixed-cost` | 100.0 | Overhead fijo por llamada al LLM |

---

## Arquitectura general

```
Text Input
    │
    ▼
┌─────────────┐
│   splitter  │  spaCy → lista de oraciones
└──────┬──────┘
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌─────────────┐                      ┌──────────────┐
│  tokenizer  │  tiktoken → n tokens │  embeddings  │  sentence-transformers → coherencia
└──────┬──────┘                      └──────┬───────┘
       │                                    │
       └──────────────┬─────────────────────┘
                      ▼
          ┌───────────────────────┐
          │  Algoritmo elegido    │
          ├───────────────────────┤
          │  dp          (DP 1D)  │
          │  dp_2d       (DP 2D)  │
          │  baseline    (greedy) │
          │  sliding_window       │
          │  dp_overlap           │
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │  SegmentationResult   │  Segments + costo total
          └──────────┬────────────┘
                     │
          ┌──────────┴──────────────────┐
          ▼                             ▼
   ┌─────────────┐             ┌─────────────────────┐
   │metrics/report│            │  calibration        │  suggest_lambda
   └─────────────┘             │  estimate_optimal_k │  validate_calibration
                               └──────────┬──────────┘
                                          │
                               ┌──────────┴──────────┐
                               ▼  (opcional)          ▼
                        ┌────────────┐       ┌─────────────────┐
                        │ llm_client │       │    evaluator    │
                        │  (Ollama)  │       │  (coherencia    │
                        └────────────┘       │   LLM-based)    │
                                             └─────────────────┘
```

---

## Módulos en `src/segmentation/`

| Módulo | Descripción |
|--------|-------------|
| `models.py` | Dataclasses `Segment` y `SegmentationResult` |
| `tokenizer.py` | Conteo de tokens con **tiktoken** (gpt-4o, gpt-4, gpt-3.5-turbo) |
| `splitter.py` | Segmentación de oraciones con **spaCy** (`en_core_web_sm`) |
| `embeddings.py` | Embeddings semánticos con **sentence-transformers** (`all-MiniLM-L6-v2`), similitud coseno y coherencia |
| `dp.py` | **Núcleo.** DP 1D (`segment_dp`) y DP 2D (`segment_dp_2d`) con función de costo: `token²  + λ·(1−coherencia) + fixed_cost` |
| `baseline.py` | Algoritmo greedy: acumula oraciones hasta `lmax` tokens |
| `sliding_window.py` | Ventana deslizante con solapamiento configurable |
| `overlap.py` | DP 1D extendido con estado de solapamiento: `dp[j][o]`, penalización `μ·overlap_tokens²` |
| `calibration.py` | `suggest_lambda` (CV de coherencias), `estimate_optimal_k` (k\* = n/√C), `validate_calibration` (compara k\* DP vs analítico), `calibrate_fixed_cost` (mide latencia Ollama) |
| `metrics.py` | `compute_metrics` → `SegmentationMetrics`; `compare` → `ComparisonReport` |
| `report.py` | `generate_full_report` → `FullReport` (todos los métodos + calibración analítica) |
| `llm_client.py` | Cliente **Ollama** (`gemma2:2b`). `run_inference`, `run_segmented_inference` |
| `evaluator.py` | Evaluación LLM: coherencia respuesta-prompt vía embeddings |
| `__init__.py` | Exporta toda la API pública |

---

## Modelo de costo

```
Total Cost = Σ segment_cost(i)

segment_cost = token_count²                   # costo computacional cuadrático
             + λ · (1 − avg_coherence)        # penalización por incoherencia semántica
             + μ · overlap_tokens²            # penalización por solapamiento (DP+Overlap)
             + fixed_cost                     # overhead fijo por llamada al LLM
```

| Parámetro | Efecto |
|-----------|--------|
| `λ` (coherence_lambda) alto | Prioriza segmentos semánticamente coherentes |
| `μ` (overlap_mu) alto | Minimiza el solapamiento entre segmentos |
| `fixed_cost` alto | Favorece menos segmentos (curva en U en DP 2D) |

---

## Flujo de datos

```
str (texto crudo)
    │  split_sentences()
    ▼
list[str]  (oraciones)
    │
    ├─ count_tokens_batch()  →  list[int]  (tokens por oración)
    └─ get_embeddings()      →  ndarray (384-d)
                                    └─ compute_coherence_scores() → list[float]
    │
    │  algoritmo de segmentación
    ▼
SegmentationResult { segments, total_cost, num_segments }
    │
    ├─ compute_metrics()         →  SegmentationMetrics
    ├─ compare()                 →  ComparisonReport
    ├─ estimate_optimal_k()      →  int  (k* analítico)
    ├─ validate_calibration()    →  dict { k_dp, k_analytical, deviation_pct, valid }
    └─ evaluate_segmentation()   →  EvaluationReport  (vía Ollama, opcional)
```

---

## Stack de versiones

| Paquete | Versión |
|---------|---------|
| Python | ≥ 3.11 |
| spacy | ≥ 3.8 |
| tiktoken | ≥ 0.12 |
| sentence-transformers | ≥ 5.4 |
| ollama | ≥ 0.6 |
| streamlit | ≥ 1.56 |
| fastapi | ≥ 0.135 |
| matplotlib | ≥ 3.10 |
| pytest | ≥ 9.0 |
| numpy | (dependencia transitiva) |

---

## API Python

```python
from src.segmentation import (
    segment_dp, segment_dp_2d, segment_baseline,
    compute_metrics, compare,
    suggest_lambda, estimate_optimal_k, validate_calibration,
    generate_full_report,
)

# Segmentación óptima
result = segment_dp(text="...", lmin=20, lmax=150,
                    coherence_lambda=0.5, fixed_cost=100.0)

# Métricas
metrics = compute_metrics(result)
print(f"Segmentos: {metrics.num_segments}, Costo: {metrics.total_cost:.0f}")

# Calibración
lam = suggest_lambda(text)
k_star = estimate_optimal_k(text, fixed_cost=100.0)
cal = validate_calibration(text, lmin=20, lmax=150,
                           coherence_lambda=lam, fixed_cost=100.0)
print(f"k* DP={cal['k_dp']}, k* analítico={cal['k_analytical']}, válido={cal['valid']}")

# Reporte completo
report = generate_full_report(text, lmin=20, lmax=150, fixed_cost=100.0)
```