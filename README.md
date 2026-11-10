# LLM Optimal Segmentation

Sistema de segmentación óptima de textos para procesamiento con Modelos de Lenguaje Grande (LLMs). El objetivo es dividir un texto en segmentos que respeten las restricciones de ventana de contexto del LLM (tokens mínimos y máximos por segmento), minimizando simultáneamente el costo computacional cuadrático y maximizando la coherencia semántica entre oraciones consecutivas.

El núcleo del sistema es un algoritmo de **Programación Dinámica (DP)** que garantiza la solución óptima global, a diferencia de enfoques greedy que solo aseguran óptimos locales. Se implementan además variantes como DP 2D (k\* óptimo), DP con solapamiento, ventana deslizante y **TextTiling** para comparación exhaustiva. Un módulo de calibración estima parámetros óptimos analíticamente y valida la coherencia entre el k\* teórico y el k\* encontrado por DP.

El proyecto incluye una interfaz web interactiva (Streamlit), una CLI completa y una API Python pública, todo respaldado por una suite de **173+ tests**.

> **Tesis:** Universidad Mayor de San Andrés (UMSA) — 2025

---

## Requisitos

| Componente | Versión |
|-----------|---------|
| Python | 3.13+ |
| uv | 0.4+ |
| Ollama | cualquier versión reciente *(opcional)* |
| Modelo LLM local | `gemma2:2b` *(opcional, solo para evaluación real)* |
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

# 5. (Opcional) Instalar y arrancar Ollama para evaluación real
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

Abre `http://localhost:8501`. Funcionalidades:

- **5 métodos de segmentación** visualizados simultáneamente: DP Óptimo, Baseline, Sliding Window, DP+Overlap, TextTiling
- **Carga de texto**: escribe, pega, sube un archivo `.txt` (máx. 5 MB) o selecciona uno de los 20 textos del corpus
- **Fusión automática de oraciones** configurable (`Máx. oraciones`) para textos muy largos
- **Indicador de longitud** en tiempo real con estimación de tiempo de procesamiento
- **Distribución de tokens** por segmento con gráficos Plotly interactivos y tooltips
- **Mapa de cortes** sobre el texto con tarjetas por segmento (color por método)
- **Curva de costo DP 2D** (costo vs. k segmentos) con k\* marcado y curva en U
- **Calibración analítica** (λ sugerido, k\* analítico vs. k\* DP, desviación)
- **Tabla comparativa** con segmentos, costo, avg/std tokens, coherencia, **Pk** y **WindowDiff** (vs DP óptimo)
- **Descarga de resultados** en `.txt` y `.json`
- **Evaluación con LLM real** vía Ollama (opcional, botón separado)

---

## CLI — Ejemplos de uso

### Segmentar con un método específico

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method dp
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method baseline
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method dp2d
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 segment --method texttiling
```

Métodos disponibles: `dp`, `baseline`, `sliding_window`, `overlap`, `dp2d`, `texttiling`.

### Comparar DP vs Baseline

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 compare
```

### Reporte completo de todos los métodos

```bash
# Sin evaluación LLM (rápido)
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 report

# Con evaluación LLM real (requiere Ollama)
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 report --llm
```

### Calibrar parámetros automáticamente

```bash
python main.py --text "Tu texto aquí..." --lmin 20 --lmax 150 --fixed-cost 1000 calibrate
```

Imprime el λ sugerido, el k\* analítico (con λ incluido), valida contra el k\* del DP 2D y —si Ollama está disponible— calibra el costo fijo midiendo latencia real.

### Benchmark de reproducibilidad

```bash
# Parámetros canónicos de la tesis
python experiments/reproduce.py

# Con parámetros personalizados y guardado en JSON
python experiments/run_benchmark.py \
    --lmin 20 --lmax 150 --lambda 0.5 --mu 0.03 \
    --fixed-cost 1000 --model gpt-4o \
    --save-json outputs/benchmark_results.json
```

Corre los 6 métodos sobre los 20 textos del corpus. Calcula métricas de costo, coherencia, calibración k\*, Pk/WindowDiff por texto, t-test, Cohen's d y FDR. Genera adicionalmente dos figuras en `outputs/`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--lmin` | 20 | Tokens mínimos por segmento |
| `--lmax` | 150 | Tokens máximos por segmento |
| `--lambda` | 0.5 | Peso de coherencia semántica (λ) |
| `--mu` | 0.03 | Peso del overlap (μ, solo DP+Overlap) |
| `--fixed-cost` | 1000.0 | Overhead fijo por llamada al LLM (Cf) |
| `--overlap` | 30 | Tokens de solapamiento en Sliding Window |
| `--model` | `gpt-4o` | Tokenizador tiktoken |
| `--save-json` | — | Ruta para guardar resultados en JSON |
| `--no-plots` | — | Omitir regeneración de gráficos |

### Análisis estadístico (requiere benchmark previo)

```bash
# t-test pareado, Cohen's d, IC 95%, FDR para las 4 comparaciones
python experiments/statistical_analysis.py

# Sobre un JSON de benchmark específico
python experiments/statistical_analysis.py --results outputs/mi_benchmark.json
```

Calcula para cada comparación (DP vs Baseline, Sliding Window, DP+Overlap, TextTiling):
- **t-test pareado** con grados de libertad n−1
- **Cohen's d** basado en diferencias pareadas: `d = mean(diffs) / std(diffs)`
- **IC al 95%** usando distribución t de Student
- **p-valor corregido por FDR** (Benjamini-Hochberg) para controlar la tasa de falsos positivos en comparaciones múltiples

Genera `outputs/statistical_analysis.png` (boxplots con d y p por método) y `outputs/statistical_results.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--results` | `outputs/benchmark_results.json` | JSON de resultados del benchmark |
| `--no-plots` | — | Omitir generación de figura |

### Análisis de sensibilidad (grid λ × Cf)

```bash
# Grid 5×5 sobre 5 textos representativos (~1 min)
python experiments/sensitivity_analysis.py --n-texts 5

# Sobre todos los textos (~15 min)
python experiments/sensitivity_analysis.py --all-texts

# Con parámetros personalizados
python experiments/sensitivity_analysis.py --n-texts 10 --max-sentences 150 --lmin 10 --lmax 200
```

Para cada combinación del grid explora cómo cambian k\* (número óptimo de segmentos) y la reducción de costo DP vs Baseline, revelando la robustez de los parámetros default y las regiones de mejor rendimiento.

Genera `outputs/sensitivity_analysis.png` (dos heatmaps anotados: k\* y reducción de costo) y `outputs/sensitivity_results.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--n-texts` | 5 | Número de textos del corpus a usar (uniformemente distribuidos) |
| `--all-texts` | — | Usar los 20 textos completos (ignora `--n-texts`) |
| `--lmin` | 20 | Tokens mínimos por segmento |
| `--lmax` | 150 | Tokens máximos por segmento |
| `--max-sentences` | 100 | Límite de oraciones por texto (controla velocidad del DP) |
| `--model` | `gpt-4o` | Tokenizador |
| `--no-plots` | — | Omitir generación de figura |

### Análisis de complejidad computacional

```bash
# Medición estándar (n hasta 200, 3 repeticiones)
python experiments/complexity_analysis.py --max-n 200 --reps 3

# Medición rápida para verificar
python experiments/complexity_analysis.py --max-n 50 --reps 2
```

Genera textos sintéticos de n ∈ {10, 20, 30, 50, 75, 100, 150, 200} oraciones y mide la mediana del tiempo de ejecución de DP, Baseline y TextTiling. El gráfico log-log muestra la pendiente empírica (exponente de complejidad) para cada método.

Genera `outputs/complexity_analysis.png` (gráfico lineal + log-log con curvas teóricas) y `outputs/complexity_results.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--max-n` | 200 | Número máximo de oraciones a medir |
| `--reps` | 3 | Repeticiones por punto (se reporta la mediana) |
| `--lmin` | 10 | Tokens mínimos para el DP |
| `--lmax` | 60 | Tokens máximos para el DP |
| `--model` | `gpt-4o` | Tokenizador |
| `--no-plots` | — | Omitir generación de figura |

### Casos de estudio cualitativos (DP vs Baseline, corte a corte)

```bash
# 6 textos representativos con detalle de fronteras (default)
python experiments/case_studies.py

# Textos específicos por ID
python experiments/case_studies.py --ids llm_inference nlp_segmentation history_roman_empire

# Modo compacto sin detalle de fronteras individuales
python experiments/case_studies.py --n-texts 8 --brief
```

Para cada texto muestra: similitud coseno en cada frontera de corte, previews del texto a ambos lados, estadística de calidad (DP vs Baseline), y veredicto semántico. Genera `outputs/case_studies.png` y `outputs/case_studies.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--n-texts` | 6 | Número de textos a analizar |
| `--ids` | — | IDs específicos del corpus (espacio-separados) |
| `--lmin` / `--lmax` | 20 / 150 | Parámetros de segmentación |
| `--lambda` | 0.5 | Peso de coherencia semántica |
| `--brief` | — | Omitir detalle de fronteras individuales |
| `--no-plots` | — | Omitir generación de figura |

### Caracterización del corpus

```bash
# Analiza los 20 textos del corpus (tokens, coherencia, λ, k* por texto y dominio)
python experiments/corpus_analysis.py
```

Genera tabla de estadísticas por texto y por dominio, y 4 gráficos: distribución de tokens por dominio, coherencia media por dominio, k* vs N, y CV de coherencia vs λ sugerido. Salidas: `outputs/corpus_analysis.png` y `outputs/corpus_analysis.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--model` | `gpt-4o` | Tokenizador |
| `--no-plots` | — | Omitir generación de figuras |

### Validación del modelo de costo con LLM real

```bash
# Requiere Ollama corriendo con gemma2:2b descargado
python experiments/cost_validation.py --n-texts 3

# Guardar resultados en JSON específico
python experiments/cost_validation.py --n-texts 5 --save-json outputs/cost_validation.json
```

Segmenta textos con DP y Baseline, corre inferencia real en Ollama para cada segmento, mide latencia en ms y calcula la regresión `latency = α·t² + β`. Reporta R² para validar que el costo cuadrático predice bien la latencia real. Salidas: `outputs/cost_validation.png` y `outputs/cost_validation.json`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--n-texts` | 3 | Textos del corpus a usar |
| `--llm-model` | `gemma2:2b` | Modelo Ollama para inferencia |
| `--lambda` | 0.5 | λ para la segmentación DP |
| `--fixed-cost` | 1000.0 | Cf para la segmentación DP |
| `--save-json` | `outputs/cost_validation.json` | Ruta de salida JSON |
| `--no-plots` | — | Omitir figura |

### Pipeline de validación completo (para tesis)

Ejecutar en este orden para reproducir todos los resultados:

```bash
# 1. Benchmark sobre el corpus completo (necesario para los pasos siguientes)
python experiments/run_benchmark.py \
    --lmin 20 --lmax 150 --lambda 0.5 --fixed-cost 1000 \
    --save-json outputs/benchmark_results.json

# 2. Análisis estadístico — t-test, Cohen's d, IC95%, FDR
python experiments/statistical_analysis.py
#    → outputs/statistical_analysis.png + statistical_results.json

# 3. Análisis de sensibilidad — grid λ × Cf
python experiments/sensitivity_analysis.py --n-texts 5
#    → outputs/sensitivity_analysis.png + sensitivity_results.json

# 4. Complejidad computacional — runtime vs n
python experiments/complexity_analysis.py --max-n 200 --reps 3
#    → outputs/complexity_analysis.png + complexity_results.json

# 5. Validación del modelo de costo (requiere Ollama)
python experiments/cost_validation.py --n-texts 3
#    → outputs/cost_validation.png + cost_validation.json

# 6. Casos de estudio cualitativos
python experiments/case_studies.py --n-texts 8
#    → outputs/case_studies.png + case_studies.json

# 7. Caracterización del corpus
python experiments/corpus_analysis.py
#    → outputs/corpus_analysis.png + corpus_analysis.json
```

### Flags globales

| Flag | Default | Descripción |
|------|---------|-------------|
| `--text` | (requerido) | Texto a segmentar |
| `--lmin` | 20 | Tokens mínimos por segmento |
| `--lmax` | 150 | Tokens máximos por segmento |
| `--overlap` | 20 | Overlap para sliding window |
| `--model` | `gpt-4o` | Tokenizador (gpt-4o / gpt-4 / gpt-3.5-turbo) |
| `--fixed-cost` | 1000.0 | Overhead fijo por llamada al LLM (Cf) |

---

## Arquitectura general

```
Text Input
    │
    ▼
┌─────────────┐
│   splitter  │  spaCy → oraciones (con chunking para textos >800K chars)
└──────┬──────┘
       │  (fusión automática si max_sentences está activo)
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌─────────────┐                      ┌──────────────┐
│  tokenizer  │  tiktoken → tokens   │  embeddings  │  all-MiniLM-L6-v2 + cache
└──────┬──────┘                      └──────┬───────┘
       │  prefix-sums de tokens             │  prefix-sums de similitudes
       └──────────────┬─────────────────────┘
                      ▼
          ┌───────────────────────────┐
          │  Algoritmo elegido        │
          ├───────────────────────────┤
          │  dp            (DP 1D)    │
          │  dp_2d         (DP 2D)    │
          │  baseline      (greedy)   │
          │  sliding_window           │
          │  dp_overlap    (DP+O)     │
          │  texttiling    (Hearst)   │
          └──────────┬────────────────┘
                     │
                     ▼
          ┌───────────────────────────┐
          │  SegmentationResult       │  segments + total_cost + num_segments
          └──────────┬────────────────┘
                     │
          ┌──────────┴──────────────────────┐
          ▼                                 ▼
   ┌─────────────────┐          ┌────────────────────────┐
   │ metrics / report│          │ calibration            │
   │ recompute_full_ │          │ suggest_lambda         │
   │ cost (fair cmp) │          │ estimate_optimal_k(λ)  │
   │ Pk / WindowDiff │          │ validate_calibration   │
   └─────────────────┘          └──────────┬─────────────┘
                                           │ (opcional)
                                ┌──────────┴──────────┐
                                ▼                     ▼
                         ┌────────────┐     ┌─────────────────┐
                         │ llm_client │     │    evaluator    │
                         │  (Ollama)  │     │  (coherencia    │
                         └────────────┘     │   LLM-based)    │
                                            └─────────────────┘
```

---

## Módulos en `src/segmentation/`

| Módulo | Descripción |
|--------|-------------|
| `models.py` | Dataclasses `Segment` y `SegmentationResult` |
| `tokenizer.py` | Conteo de tokens con **tiktoken** (gpt-4o, gpt-4, gpt-3.5-turbo) |
| `splitter.py` | Segmentación de oraciones con **spaCy**; chunking automático para textos >800K chars |
| `embeddings.py` | Embeddings con **all-MiniLM-L6-v2** (384-d); similitud coseno; **cache LRU** de 8 entradas para evitar recomputo |
| `dp.py` | **Núcleo.** DP 1D y DP 2D con función de costo escalada, prefix-sums O(1), ventana bisect, `_merge_to_limit` |
| `baseline.py` | Greedy O(n): acumula oraciones hasta `lmax`, respeta `lmin` en el último segmento |
| `sliding_window.py` | Ventana deslizante con solapamiento configurable |
| `overlap.py` | DP con estado de overlap propagado `dp[j][o]`; costo neto: penalidad lineal menos bonus de coherencia en borde |
| `texttiling.py` | **TextTiling** (Hearst 1997) con embeddings semánticos: depth score + suavizado + greedy respetando lmin/lmax |
| `calibration.py` | `suggest_lambda` (CV de coherencias); `estimate_optimal_k` con λ: `k* = N·√((1+λ·(1−coh))/Cf)`; `validate_calibration` por sub-optimalidad ≤5% en curva DP 2D |
| `metrics.py` | `compute_metrics`, `compare`, `recompute_full_cost` (evaluación con función DP completa), `compare_full_cost` |
| `segmentation_metrics.py` | **Pk** (Beeferman 1999) y **WindowDiff** (Pevzner & Hearst 2002) — métricas estándar de segmentación |
| `report.py` | `generate_full_report` → `FullReport` (todos los métodos + TextTiling + calibración + dp_vs_texttiling) |
| `llm_client.py` | Cliente **Ollama** (`gemma2:2b`): `run_inference`, `run_segmented_inference` |
| `evaluator.py` | Evaluación LLM: latencia, tokens, coherencia respuesta-prompt vía embeddings |
| `__init__.py` | Exporta toda la API pública |

---

## Modelo de costo

```
Total Cost = Σ segment_cost(i)

segment_cost = token_count² · (1 + λ · (1 − avg_coherence))   # cuadrático escalado por coherencia
             + μ · overlap_tokens · (1 − boundary_sim)         # overlap neto (DP+Overlap)
             + fixed_cost                                       # overhead fijo por llamada al LLM
```

El escalado por `t²` garantiza que λ tenga peso real en la optimización (sin él, el término de coherencia es ruido numérico frente al cuadrático).

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `λ` (coherence_lambda) | 0.5 | Mayor → prioriza segmentos semánticamente coherentes |
| `μ` (overlap_mu) | 0.03 | Mayor → penaliza más el solapamiento |
| `fixed_cost` (Cf) | 1000.0 | Mayor → favorece menos segmentos (curva en U más pronunciada) |

### Calibración analítica de k\*

```
k* = N · √( (1 + λ·(1 − coh_avg)) / Cf )
```

Derivado de minimizar `N²/k · (1+λ·(1−coh)) + k·Cf`. Para λ=0 equivale a `k* = N/√Cf`.
La validación comprueba que k\* analítico esté dentro del **5% del costo óptimo** en la curva DP 2D.

---

## Optimizaciones de rendimiento

| Optimización | Impacto |
|---|---|
| **Prefix-sums de similitudes** | Coherencia de segmento en O(1) vs. O(n) original |
| **Ventana bisect** | Bucle DP interno salta directamente al rango válido [lmin, lmax] |
| **Break cuando span < lmin** | Evita iterar i's inválidos al final del rango |
| **Cache de embeddings** (LRU 8 entradas) | Reuse al cambiar solo λ/Cf sin cambiar el texto: 0 ms de recomputo |
| **Fusión de oraciones** (`max_sentences`) | Reduce n a un umbral configurable (default 300) — O(300²) máx. |
| **Chunking en splitter** | Textos >800K chars se procesan en chunks para evitar el límite de spaCy |

Con `max_sentences=300`: cualquier texto, sin importar su longitud, se procesa en tiempo acotado (~1-2 segundos).

---

## Corpus de evaluación

`data/corpus.json` contiene **20 textos en español** con diversidad de dominio y longitud:

| Dominio | Textos |
|---------|--------|
| LLM / Deep Learning / NLP | 5 textos (inferencia, transformers, coherencia semántica, segmentación, PLN) |
| Algoritmos | 3 textos (programación dinámica, complejidad computacional, optimización) |
| Ciencias | 3 textos (metabolismo celular, replicación ADN, mecánica cuántica) |
| Historia | 2 textos (Imperio Romano, Revolución Francesa) |
| Otras disciplinas | 7 textos (economía, filosofía, derecho, medicina, geografía, IR, software) |

Longitudes: **2 short** (~250 palabras), **10 medium** (350–460 palabras), **8 long** (600–790 palabras).

---

## Validación estadística

### Scripts de análisis (`experiments/`)

| Script | Propósito | Salidas |
|--------|-----------|---------|
| `run_benchmark.py` | Corre los 6 métodos sobre el corpus; calcula Pk/WindowDiff, t-test, Cohen's d y FDR | `benchmark_results.json` |
| `statistical_analysis.py` | t-test pareado, Cohen's d, IC 95%, FDR (Benjamini-Hochberg) para 4 comparaciones | `statistical_analysis.png`, `statistical_results.json` |
| `sensitivity_analysis.py` | Grid 5×5 de λ × Cf; registra k\* promedio y reducción de costo por combinación | `sensitivity_analysis.png`, `sensitivity_results.json` |
| `complexity_analysis.py` | Runtime vs n con textos sintéticos; gráfico lineal + log-log con exponente estimado | `complexity_analysis.png`, `complexity_results.json` |
| `cost_validation.py` | Valida el modelo de costo cuadrático comparando predicción vs latencia Ollama real | `cost_validation.png`, `cost_validation.json` |
| `case_studies.py` | Comparación cualitativa DP vs Baseline: similitud en cada frontera, veredicto semántico | `case_studies.png`, `case_studies.json` |
| `corpus_analysis.py` | Estadísticas del corpus por texto y por dominio: tokens, coherencia, λ, k\* | `corpus_analysis.png`, `corpus_analysis.json` |
| `reproduce.py` | Parámetros canónicos para reproducibilidad de la tesis | — |

### Resultados clave (corpus: 20 textos en español)

| Comparación | Reducción media | IC 95% | Cohen's d | p (FDR) |
|-------------|----------------|--------|-----------|---------|
| DP vs Baseline | ~49% | [47%, 51%] | 4.22 (grande) | < 0.001 |
| DP vs Sliding Window | ~57% | [54%, 59%] | 3.56 (grande) | < 0.001 |
| DP vs TextTiling | — | — | — | — |

> Los resultados exactos dependen del benchmark corrido localmente. La tabla se actualiza con `python experiments/statistical_analysis.py`.

### Métricas de frontera — Pk y WindowDiff

El benchmark calcula automáticamente **Pk** (Beeferman et al. 1999) y **WindowDiff** (Pevzner & Hearst 2002) para cada método usando DP como referencia:

```
Pk(DP, método) = proporción de ventanas en las que hay desacuerdo de frontera
WindowDiff     = proporción de ventanas en las que el conteo de fronteras difiere
```

Ambas métricas aparecen en la **tabla comparativa de la UI** y en el JSON de resultados del benchmark (`boundary_metrics` por texto y promedios en `aggregates`).

### Análisis de sensibilidad

Grid λ ∈ {0, 0.5, 1.0, 1.5, 2.0} × Cf ∈ {200, 500, 1000, 2000, 5000} sobre el corpus:
- **k\*** varía ≈4 segmentos en todo el espacio de parámetros (robustez alta)
- La **reducción de costo** varía hasta 58 puntos porcentuales (λ y Cf son determinantes)
- λ=0.5, Cf=1000 (defaults) corresponden a la región de balance óptimo

---

## Análisis de fallos (benchmark)

El benchmark detecta automáticamente 5 categorías de problemas:

1. **Calibración inválida**: k\* analítico fuera del 5% de sub-optimalidad en la curva DP 2D
2. **DP no mejora sobre baseline**: reducción de costo negativa con `recompute_full_cost`
3. **k\* lejano del analítico**: `deviation_pct > 15%`
4. **Overlap inefectivo**: `total_overlap_tokens == 0` (DP+Overlap degenerado a DP 1D)
5. **Segmentos irregulares**: `std_tokens > 0.5 · avg_tokens`

---

## Métricas estándar de segmentación

```python
from src.segmentation import boundaries_from_result, pk, window_diff

# Convertir segmentaciones a representación de boundaries
ref_boundaries = boundaries_from_result(dp_result)
hyp_boundaries = boundaries_from_result(baseline_result)

# Pk (menor es mejor)
score_pk = pk(ref_boundaries, hyp_boundaries)

# WindowDiff (menor es mejor)
score_wd = window_diff(ref_boundaries, hyp_boundaries)
```

---

## Flujo de datos

```
str (texto crudo)
    │  split_sentences()  [chunking automático si >800K chars]
    ▼
list[str]  (oraciones)
    │  _merge_to_limit()  [si max_sentences activo]
    │
    ├─ count_tokens_batch()  →  list[int]  + prefix_cumtok
    └─ get_embeddings()      →  ndarray (384-d) + cache
                                    └─ _build_similarity_arrays() → sims + prefix_sims
    │
    │  algoritmo de segmentación (bisect window, O(n·w))
    ▼
SegmentationResult { segments, total_cost, num_segments }
    │
    ├─ compute_metrics()          →  SegmentationMetrics
    ├─ recompute_full_cost()      →  float  (evaluación justa con función DP)
    ├─ compare_full_cost()        →  dict   (comparación equitativa)
    ├─ pk() / window_diff()       →  float  (métricas estándar)
    ├─ estimate_optimal_k()       →  int    (k* analítico con λ)
    ├─ validate_calibration()     →  dict { k_dp, k_analytical, deviation_pct,
    │                                       relative_suboptimality, valid }
    └─ evaluate_segmentation()    →  EvaluationReport  (vía Ollama, opcional)
```

---

## Stack de versiones

| Paquete | Versión |
|---------|---------|
| Python | ≥ 3.13 |
| spacy | 3.8.14 (pinned) |
| tiktoken | 0.12.0 (pinned) |
| sentence-transformers | 5.4.0 (pinned) |
| scipy | ≥ 1.15.0 |
| plotly | ≥ 6.7.0 |
| ollama | ≥ 0.6.1 |
| streamlit | ≥ 1.56.0 |
| matplotlib | ≥ 3.10.8 |
| pytest | ≥ 9.0.3 |

---

## API Python

```python
from src.segmentation import (
    # Segmentación
    segment_dp, segment_dp_2d, segment_baseline,
    segment_sliding_window, segment_dp_overlap, segment_texttiling,
    # Métricas
    compute_metrics, compare, recompute_full_cost, compare_full_cost,
    boundaries_from_result, pk, window_diff,
    # Calibración
    suggest_lambda, estimate_optimal_k, validate_calibration, calibrate_fixed_cost,
    # Reporte completo
    generate_full_report,
    # Utilidades
    clear_embedding_cache,
)

# Segmentación óptima
result = segment_dp(text="...", lmin=20, lmax=150,
                    coherence_lambda=0.5, fixed_cost=1000.0,
                    max_sentences=300)  # fusión automática si >300 oraciones

# Comparación justa (misma función de costo para todos)
dp_cost  = recompute_full_cost(dp_result, coherence_lambda=0.5, fixed_cost=1000.0)
tt_cost  = recompute_full_cost(tt_result, coherence_lambda=0.5, fixed_cost=1000.0)
cmp = compare_full_cost(dp_result, tt_result, coherence_lambda=0.5, fixed_cost=1000.0)
print(f"DP vs TextTiling: {cmp['cost_reduction_pct']:.1f}%")

# Métricas estándar
ref = boundaries_from_result(dp_result)
hyp = boundaries_from_result(tt_result)
print(f"Pk={pk(ref, hyp):.4f}  WindowDiff={window_diff(ref, hyp):.4f}")

# Calibración con λ
lam    = suggest_lambda(text)
k_star = estimate_optimal_k(text, fixed_cost=1000.0, coherence_lambda=lam)
cal    = validate_calibration(text, lmin=20, lmax=150,
                               coherence_lambda=lam, fixed_cost=1000.0)
print(f"k*={k_star}, sub-optimalidad={cal['relative_suboptimality']:.1%}, válido={cal['valid']}")

# Reporte completo (todos los métodos + TextTiling + calibración)
report = generate_full_report(text, lmin=20, lmax=150, fixed_cost=1000.0)
print(f"λ sugerido: {report.suggested_lambda}")
print(f"DP vs TextTiling: {report.dp_vs_texttiling['cost_reduction_pct']:.1f}%")

# Limpiar cache de embeddings si se necesita liberar memoria
clear_embedding_cache()
```
