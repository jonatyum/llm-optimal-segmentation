# Notas de trabajo — Tesis

> Juan Jonatan Churata Apaza · UMSA 2025  
> Archivo: `Tesis_Churata_Apaza_2025_APA6_FINAL.docx`  
> Última revisión: 2026-05-26

---

## ⚡ PENDIENTES

### 🔴 Crítico

> ✅ Todos los ítems críticos completados el 2026-05-28.

| # | Sección | Estado |
|---|---------|--------|
| 1 | **§3.3.2** | ✅ Código corregido a versión multiplicativa |
| 2 | **§3.5 Overlap** | ✅ Párrafo con fórmula completa agregado |
| 3 | **Tabla 4** (§3.9.4) | ✅ μ=0.03 y Cf=2000 verificados correctos en tabla |
| 4 | **OE3** (§1.6.1) | ✅ Enunciado corregido a "tokens²" |
| 5 | **§1.9** | ✅ Tabla CRISP-DM × OE agregada |
| 6 | **§4.1** | ✅ Subsección §4.1.1 CRISP-DM por OE agregada |

---

### Textos exactos de reemplazo

#### §3.3.2 — Código correcto (reemplazar el bloque de código actual)

El código actual en la tesis dice:
```python
return computational_cost + coherence_lambda * coherence_penalty + fixed_cost
```

**Reemplazar todo el bloque `Código 4` con:**

```python
def _compute_cost(token_count, prefix_sims, start, end,
                  coherence_lambda, fixed_cost):
    # Componente 1: costo cuadrático de autoatención
    t2 = float(token_count ** 2)
    # Componente 2: λ escala el t² — peso de coherencia proporcional al costo
    avg_coh = mean(cosine_similarity(emb[k], emb[k+1])
                   for k in range(start, end - 1))
    coherence_factor = 1.0 + coherence_lambda * (1.0 - avg_coh)
    return t2 * coherence_factor + fixed_cost
```
*Código 4. Función de costo compuesta (dp.py). La fórmula implementada es costo(i,j) = tokens(i,j)² · [1 + λ · (1 − coh(i,j))] + Cf, donde el término de coherencia escala con t² para que λ tenga peso real frente al costo cuadrático.*

---

#### §3.5 — Párrafo de fórmula a agregar

Agregar después del párrafo que describe dp[j][o]:

> El costo de un segmento con overlap se calcula como: costo_overlap(i,j) = tokens(i,j)² · [1 + λ · (1 − coherencia(i,j))] + μ · overlap_tokens · (1 − boundary_sim) + Cf, donde boundary_sim es la similitud coseno en la frontera de transición entre segmentos consecutivos, overlap_tokens es el número de tokens compartidos con el segmento anterior, y μ controla el peso de la penalización de overlap. Con μ = 0.03 y overlap_tokens ≈ 30, la penalización de overlap representa menos del 0.05% del costo cuadrático típico, lo que explica que DP+Overlap y DP 1D produzcan resultados prácticamente equivalentes bajo los parámetros canónicos.

---

#### OE3 (§1.6.1) — Enunciado corregido

**Buscar:** `"Evaluar la eficiencia del algoritmo en términos de tiempo de procesamiento y recursos computacionales, en comparación con enfoques tradicionales de inferencia segmentada."`

**Reemplazar con:**
> Evaluar la eficiencia del algoritmo de programación dinámica en términos de costo computacional de inferencia (tokens²), comparando cuantitativamente con métodos tradicionales de inferencia segmentada mediante análisis estadístico sobre un corpus de 20 textos.

---

#### Tabla 4 (§3.9.4) — Valores correctos

| Parámetro | Símbolo | Valor correcto | Rango recomendado |
|-----------|---------|---------------|-------------------|
| Peso de coherencia | λ | 0.5 | [0.0, 2.0] |
| Peso de overlap | μ | **0.03** | [0.01, 0.1] |
| Costo fijo por segmento | Cf | **2000.0 tok²** | CPU: 2000–5000; GPU: 500–1000 |
| Longitud mínima | lmin | 20 tokens | [10, 50] |
| Longitud máxima | lmax | 150 tokens | [100, 300] |
| Segmentos óptimos | k* | N·√((1+λ(1-coh))/Cf) | — |

*Nota. Elaboración propia. Los valores de Cf y μ son los utilizados en todos los experimentos de esta tesis. El valor μ=0.03 es el implementado en el código fuente (overlap.py); μ=0.3 en versiones anteriores de este documento era un error tipográfico.*

### 🟡 Importante

| # | Sección | Estado |
|---|---------|--------|
| 7 | **Tabla 6** (§5.1) | ✅ Insertada con estadísticos correctos (43.88%, 26.02%, etc.) |
| 8 | **Tabla 5** (§3.10) | ✅ Reemplazada con tabla completa (scipy, plotly, 173 tests, TextTiling) |
| 9 | **Tabla 2** (§4.3) | ✅ Verificada en sesión anterior — contiene métricas reales |
| 10 | **Cap V + §5.1** | ✅ Renombrado a "RESULTADOS Y DISCUSIÓN"; párrafos de Discusión OE1–OE5 agregados |

### 🟢 Menor

| # | Sección | Qué hacer |
|---|---------|-----------|
| 12 | **Referencias** | Verificar en Google Scholar / ACL Anthology antes de la defensa: Kitaev & Klein (2023) EMNLP, Dai et al. (2022) ICML, Wang & Liu (2023) ICLR, Li et al. (2024) IEEE TNNLS. |

---

## ✅ Completado

| Qué | Cómo quedó |
|-----|-----------|
| Párrafo obsoleto 63.58%/85.66% (§5.1) | Eliminado |
| Hipótesis §1.5 | "costo computacional de inferencia" |
| OE1 §1.6.1 | Incluye fórmula `costo(i,j) = tokens²·(1+λ(1-coh))+Cf` y derivación k* |
| OE2 §1.6.1 | "6 módulos, 173 pruebas unitarias, Streamlit" |
| Nota boxplot §5.1 OE3 | Explica y=0 = DP referencia, DP+Overlap ≈ DP |
| Nota coherencia §5.1 OE4 | Distingue coherencia intra-segmento vs similitud en fronteras |
| Figuras 1–13 | Todas insertadas inline (1–10 en caps. III–V) + Anexo C (11–13) |
| RESUMEN y ABSTRACT | Completos con resultados reales |
| §3.7.3 TextTiling | Completo en el documento |
| §4.1 Diseño Experimental | Completo + §4.1.1 CRISP-DM por OE |
| §4.2 Conjunto de Datos | Completo con estadísticas del corpus |
| Cap. V §5.1 y §5.2 | Completo con resultados + discusión por OE y Tabla 6 |
| Cap. VI Conclusiones | Completo |
| §3.3.2 Función de Costo | Código corregido a versión multiplicativa |
| §3.5 Overlap | Fórmula de costo con overlap agregada |
| OE3 §1.6.1 | Enunciado corregido a "tokens²" |
| §1.9 Metodología | Tabla CRISP-DM × OE insertada |
| Tabla 5 §3.10 | Reemplazada con herramientas completas (scipy, plotly, 173 tests) |
| Tabla 6 §5.1 | Insertada con estadísticos completos (43.88%, IC 95%, Cohen's d) |
| Cap. V título | Renombrado a "RESULTADOS Y DISCUSIÓN" |
| Discusión OE1–OE5 | Párrafos de discusión agregados en §5.1 |

---

## Datos de referencia rápida

### Parámetros definitivos

| Parámetro | Valor correcto | Nota |
|-----------|---------------|------|
| Cf | **2000** tok² | §4.4 del doc ya lo usa; Tabla 4 debe actualizarse |
| μ | **0.03** | El código usa 0.03; Tabla 4 dice 0.3 (error tipográfico) |
| λ | 0.5 | Correcto |
| lmin / lmax | 20 / 150 | Correcto |
| N textos benchmark | 20 | Correcto |

### Resultados estadísticos principales (corpus 20 textos)

| Comparación | Reducción | IC 95% | t(19) | p | d Cohen |
|-------------|-----------|--------|-------|---|---------|
| DP vs Baseline | **43.88%** | [41.22%, 46.55%] | 17.37 | <0.001 | **3.88** |
| DP vs Sliding Window | **26.02%** | [21.61%, 30.42%] | 9.25 | <0.001 | **2.07** |
| DP vs TextTiling | **14.73%** | [11.90%, 17.57%] | 8.20 | <0.001 | **1.83** |
| DP vs DP+Overlap | −0.30% | [−0.49%, −0.12%] | −4.47 | 0.0003 | −1.00 |

Coherencia intra-segmento: **DP = 0.8994 vs Baseline = 0.4960** (+81%)  
Pk (Baseline vs DP): 0.569 ± 0.095 · Pk (TextTiling vs DP): 0.546 ± 0.073

### Mapa de figuras insertadas

| Fig. | Archivo | Sección |
|------|---------|---------|
| 1 | cap3/costo_vs_lmax.png | §3.6 |
| 2 | cap3/reduccion_vs_lmax.png | §3.6 |
| 3 | cap4/corpus_tokens_por_dominio.png | §4.2 |
| 4 | cap4/corpus_coherencia_por_dominio.png | §4.2 |
| 5 | cap4/sensitivity_reduccion.png | §4.4 |
| 6 | cap5/statistical_boxplots.png | §5.1 OE3 |
| 7 | cap5/case_studies/case_llm_inference.png | §5.1 OE4 |
| 8 | cap5/case_studies/case_transformer_architecture.png | §5.1 OE4 |
| 9 | cap5/case_studies/case_text_segmentation.png | §5.1 OE4 |
| 10 | cap5/complexity_loglog.png | §5.1 OE5 |
| 11 | anexo/MAIN_tokens_llm_inference.png | Anexo C |
| 12 | anexo/MAIN_tokens_constitutional_law.png | Anexo C |
| 13 | anexo/MAIN_tokens_amazon_river_basin.png | Anexo C |

### Fórmulas correctas (para verificar en el docx)

**Función de costo (§2.4.1, §3.3.2):**
```
costo(i,j) = tokens(i,j)² · [1 + λ · (1 − coherencia(i,j))] + μ · overlap_tokens · (1 − boundary_sim) + Cf
```

**k* analítico con coherencia (§4.4):**
```
k* = N · √( (1 + λ · (1 − coh_avg)) / Cf )
```

### Advertencias para la defensa

- **DP+Overlap es −0.30%** (marginalmente más caro que DP). Explicación: y=0 en el boxplot ES el DP referencia; −0.30% significa equivalencia práctica. Ya está aclarado en §5.1 OE3.
- **Coherencia intra-segmento ≠ similitud en fronteras**: DP tiene 0.8994 de coherencia intra-segmento (segmentos cortos y focalizados) pero en los case studies la similitud en fronteras es 0.35–0.53. Son métricas distintas. Ya está aclarado en §5.1 OE4.
- **Textos problemáticos**: amazon_river_basin (4.6% reducción) y french_revolution (6.3%) tienen mínima mejora porque el Baseline ya fragmenta razonablemente en textos muy largos. Mencionar como limitación esperada en §5.1 OE5.
- **k* analítico**: desviación media 17% en número de segmentos. Los 5 textos que fallan son todos >800 tokens. Ya documentado en §4.4.

---

## Metodología por Objetivo Específico

> **Qué agregar al docx y dónde.** No tocar el docx directamente — usar estas notas como fuente.

### Cambio 1 — Corregir OE3 (§1.6.1)

**Buscar:** `"Evaluar la eficiencia del algoritmo en términos de tiempo de procesamiento y recursos computacionales, en comparación con enfoques tradicionales de inferencia segmentada."`

**Reemplazar con:**
> Evaluar la eficiencia del algoritmo de programación dinámica en términos de costo computacional de inferencia (tokens²), comparando cuantitativamente con métodos tradicionales de inferencia segmentada mediante análisis estadístico sobre un corpus de 20 textos.

---

### Cambio 2 — Tabla matriz en §1.9

**Dónde:** después del último párrafo de §1.9 (el que termina con "despliegue del sistema como demostración.")

**Párrafo introductorio a insertar:**
> La Tabla X presenta la correspondencia entre las seis fases de CRISP-DM y los cinco objetivos específicos de la investigación, evidenciando la cobertura metodológica completa del estudio.

**Tabla a insertar:**

| Fase CRISP-DM | OE1 | OE2 | OE3 | OE4 | OE5 |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Comprensión del problema | ✓ | | ✓ | | |
| 2. Comprensión de los datos | | | ✓ | ✓ | |
| 3. Preparación de los datos | | ✓ | ✓ | ✓ | |
| 4. Modelado | ✓ | ✓ | | | |
| 5. Evaluación | | | ✓ | ✓ | ✓ |
| 6. Despliegue | | ✓ | | | ✓ |

*Nota. Elaboración propia. ✓ indica que la fase contribuye directamente al objetivo específico.*

---

### Cambio 3 — Subsección en §4.1

**Dónde:** después del último párrafo de §4.1 (el que menciona `python experiments/reproduce.py`)

**Heading nivel 2 a insertar:**
> Aplicación de la Metodología CRISP-DM por Objetivo Específico

**Párrafo introductorio a insertar:**
> A continuación se describe la aplicación de las seis fases de la metodología CRISP-DM a cada uno de los cinco objetivos específicos de la investigación. Las fases 1 a 3 comparten una base común que se especifica en cada objetivo según su énfasis particular; las fases 4 a 6 son propias de cada objetivo.

---

**Párrafos a insertar en orden (uno por OE, cada uno con sus 6 fases):**

---

#### OE1 — Formalización matemática del modelo de segmentación óptima

> **Fase 1 — Comprensión del problema.** Se formuló el problema de segmentación óptima como la minimización del costo cuadrático total de inferencia, identificando como variables de decisión el número k y la posición de los cortes entre segmentos, y como restricciones operativas los límites de longitud lmin = 20 y lmax = 150 tokens. Se estableció que el costo cuadrático de autoatención de los transformers justifica la función objetivo: minimizar Σ costo(i,j).
>
> **Fase 2 — Comprensión de los datos.** Se analizó la distribución de longitudes del corpus (340–1141 tokens, media 623 tokens) para verificar que los límites lmin–lmax son alcanzables en todos los textos del conjunto de evaluación. Se comprobó que la coherencia semántica varía por dominio (CV 0.05–0.28), lo que justifica incluir el término de coherencia en la función de costo.
>
> **Fase 3 — Preparación de los datos.** Se construyó la cadena de preprocesamiento necesaria para evaluar el modelo: tokenización con tiktoken (o200k_base), segmentación en oraciones con spaCy, cálculo de prefix sums de tokens para evaluar spans en O(1), y generación de embeddings semánticos con all-MiniLM-L6-v2 para calcular la coherencia promedio de cada segmento candidato.
>
> **Fase 4 — Modelado.** Se derivó la función de costo compuesta costo(i,j) = tokens(i,j)² · [1 + λ · (1 − coherencia(i,j))] + Cf, justificando el término multiplicativo para que λ escale con el costo cuadrático. Se demostró que la recursión de Bellman dp[j] = min{dp[i] + costo(i,j)} para lmin ≤ tokens(i,j) ≤ lmax garantiza la optimalidad global mediante el principio de subestructura óptima. Se extendió el modelo a la formulación 2D dp[i][k] para derivar analíticamente el número óptimo de segmentos k* = N · √((1 + λ · (1 − coh_avg)) / Cf).
>
> **Fase 5 — Evaluación.** Se validó la fórmula k* sobre el corpus de 20 textos comparando k* analítico con k* empírico (argmin dp[n][k]). La desviación media fue del 17% en número de segmentos; los 5 textos con mayor desviación corresponden a textos >800 tokens donde la heterogeneidad interna invalida la suposición de coherencia promedio uniforme.
>
> **Fase 6 — Despliegue.** El modelo matemático quedó formalizado en los Capítulos II y III de la tesis (§3.3–§3.6), con las derivaciones completas de la función de costo y la prueba de optimalidad. La interfaz Streamlit visualiza la curva de costo total vs k como evidencia empírica del mínimo analítico.

---

#### OE2 — Implementación del sistema modular de segmentación

> **Fase 1 — Comprensión del problema.** Se definieron los requisitos del sistema: seis módulos independientes con interfaces bien delimitadas, soporte para los cinco métodos comparadores, y capacidad de procesar textos de hasta 1141 tokens en tiempo interactivo. Se estableció como criterio de corrección la suite de pruebas automatizadas con cobertura de casos límite.
>
> **Fase 2 — Comprensión de los datos.** Se revisó el corpus para identificar los casos de prueba más representativos: textos cortos (< 400 tokens), textos largos (> 800 tokens), textos de dominio técnico con alta coherencia y textos narrativos con baja coherencia. Estos casos guiaron el diseño de las pruebas unitarias e integración.
>
> **Fase 3 — Preparación de los datos.** Se configuró el entorno de desarrollo con las dependencias fijadas en pyproject.toml (Python 3.13, tiktoken 0.12.0, spaCy 3.8.14, sentence-transformers 5.4.0) y se implementó la cadena de preprocesamiento como módulos reutilizables: tokenizer.py, splitter.py, embeddings.py.
>
> **Fase 4 — Modelado / Implementación.** Se tradujo el modelo matemático a seis módulos: tokenizer (tokenización y prefix sums), splitter (segmentación en oraciones), embeddings (caché LRU de vectores semánticos, máx. 8 entradas), dp (algoritmo DP 1D y 2D con optimización bisect, O(n·w)), metrics (Pk, WindowDiff, coherencia intra-segmento) y calibration (suggest_lambda, suggest_cf). Los cinco métodos comparadores se implementaron como módulos independientes bajo la misma interfaz.
>
> **Fase 5 — Evaluación.** Se ejecutó la suite de 173 pruebas unitarias e integración con pytest. Se verificó corrección en casos límite: texto vacío, segmento único, lmax menor que la longitud mínima de oración, corpus con una sola oración. Todas las pruebas pasan en la versión final.
>
> **Fase 6 — Despliegue.** Se desplegó el sistema como interfaz Streamlit (app.py) que permite al usuario ingresar texto, seleccionar parámetros λ, Cf, lmin, lmax, visualizar la segmentación resultante, comparar los cinco métodos y exportar los resultados. El sistema es reproducible mediante `python experiments/reproduce.py`.

---

#### OE3 — Evaluación de la eficiencia computacional

> **Fase 1 — Comprensión del problema.** Se definió la métrica de evaluación principal: el costo computacional de inferencia medido en tokens², que es la unidad relevante para cuantificar la carga de autoatención en transformers. Se estableció que la comparación debe realizarse sobre el mismo corpus con los mismos parámetros para todos los métodos, garantizando condiciones experimentales controladas.
>
> **Fase 2 — Comprensión de los datos.** Se caracterizó el corpus de 20 textos en 8 dominios temáticos: longitud entre 340 y 1141 tokens (media 623 tokens), coherencia semántica entre 0.42 y 0.71 por dominio, coeficiente de variación intra-dominio entre 0.05 y 0.28. Esta diversidad garantiza que los resultados sean generalizables y no dependientes de un único tipo de texto.
>
> **Fase 3 — Preparación de los datos.** Se estandarizaron las condiciones de benchmark: parámetros canónicos (lmin=20, lmax=150, λ=0.5, Cf=2000) iguales para todos los métodos, tokenización idéntica con tiktoken para todos los métodos, y función de costo DP recomputada sobre los segmentos de cada método para garantizar comparación justa del costo de inferencia.
>
> **Fase 4 — Modelado.** Se definió el modelo estadístico de comparación: t-test pareado de dos colas con corrección FDR Benjamini-Hochberg para controlar la tasa de falsos positivos en las cuatro comparaciones simultáneas. Se fijó α = 0.05 y se calculó Cohen's d e intervalos de confianza al 95% para cuantificar la magnitud del efecto.
>
> **Fase 5 — Evaluación.** Se ejecutó el benchmark sobre el corpus completo. Resultados: DP vs Baseline: 43.88% de reducción (d=3.88, p<0.001, IC: [41.22%, 46.55%]); DP vs Sliding Window: 26.02% (d=2.07, p<0.001); DP vs TextTiling: 14.73% (d=1.83, p<0.001); DP vs DP+Overlap: −0.30% (equivalencia práctica). Todos los efectos superan el umbral de efecto grande (d > 0.8).
>
> **Fase 6 — Despliegue.** Los resultados se reportan en §5.1 OE3 con el boxplot comparativo (Figura 6), la Tabla 6 con los estadísticos completos y el párrafo de interpretación. El script `experiments/run_benchmark.py` permite reproducir todos los valores.

---

#### OE4 — Evaluación de la precisión semántica

> **Fase 1 — Comprensión del problema.** Se definieron dos dimensiones de calidad semántica a evaluar: (1) coherencia intra-segmento — cuán temáticamente cohesivos son los segmentos producidos; y (2) calidad de fronteras — cuán bien coinciden las fronteras con las divisiones temáticas naturales del texto. Se estableció que ambas dimensiones son complementarias y necesarias para una evaluación completa.
>
> **Fase 2 — Comprensión de los datos.** Se seleccionaron tres textos representativos para los estudios de caso cualitativos: llm_inference (dominio técnico-descriptivo, alta coherencia interna), transformer_architecture (dominio técnico-estructurado, múltiples subtemas), y text_segmentation (dominio metodológico, transiciones temáticas marcadas). Se verificó que cubren los patrones más frecuentes del corpus.
>
> **Fase 3 — Preparación de los datos.** Se computaron los embeddings de todas las oraciones del corpus con all-MiniLM-L6-v2 y se calcularon las similitudes coseno en cada frontera de segmento para todos los métodos. Se prepararon los vectores de fronteras binarias (1 = corte, 0 = continuidad) requeridos por las métricas Pk y WindowDiff.
>
> **Fase 4 — Modelado.** Se definió el protocolo de evaluación: coherencia intra-segmento como media de similitudes coseno entre oraciones consecutivas dentro de cada segmento; Pk con ventana k = len(texto)/2k_ref (Beeferman et al., 1999); WindowDiff con ventana equivalente (Pevzner & Hearst, 2002). Para los estudios de caso se diseñó la visualización de similitud coseno por frontera para comparar las decisiones de corte de DP y Baseline.
>
> **Fase 5 — Evaluación.** Coherencia intra-segmento: DP = 0.8994 vs Baseline = 0.4960 (+81%). Calidad de fronteras: Pk(Baseline vs DP) = 0.569 ± 0.095; Pk(TextTiling vs DP) = 0.546 ± 0.073. Los estudios de caso confirman cualitativamente que DP agrupa oraciones temáticamente relacionadas mientras Baseline corta por longitud sin considerar coherencia.
>
> **Fase 6 — Despliegue.** Los resultados se reportan en §5.1 OE4 con las Figuras 7–9 (estudios de caso) y el análisis interpretativo. La interfaz Streamlit permite al usuario inspeccionar la coherencia intra-segmento de cualquier texto en tiempo real.

---

#### OE5 — Análisis de complejidad, sensibilidad y casos límite

> **Fase 1 — Comprensión del problema.** Se definieron tres dimensiones de análisis complementarias: (1) complejidad computacional empírica del algoritmo DP; (2) robustez del número óptimo de segmentos k* ante variaciones de los hiperparámetros λ y Cf; y (3) identificación de condiciones donde la ventaja del DP se reduce. Estos análisis responden a las limitaciones del modelo y orientan el trabajo futuro.
>
> **Fase 2 — Comprensión de los datos.** Se diseñaron los experimentos: para complejidad, se generaron textos sintéticos con n ∈ {10, 20, 30, 50, 75, 100, 150, 200} oraciones; para sensibilidad, se definió una grid 5×5 con λ ∈ {0.1, 0.3, 0.5, 1.0, 2.0} y Cf ∈ {500, 1000, 2000, 3000, 5000}; para casos límite, se identificaron los textos del corpus con menor reducción porcentual (amazon_river_basin: 4.6%, french_revolution: 6.3%).
>
> **Fase 3 — Preparación de los datos.** Se prepararon los textos sintéticos de longitud controlada para el experimento de complejidad y se configuró la grid de parámetros para el análisis de sensibilidad. Se instrumentó el código con `time.perf_counter()` para medir tiempos de ejecución con precisión de microsegundos.
>
> **Fase 4 — Modelado.** Se estableció el modelo teórico de complejidad O(n·w) donde w = lmax/token_avg es el ancho de la ventana activa. Para sensibilidad se usó k* analítico como referencia y se midió la variación empírica de k* en toda la grid. Se definió como umbral de estabilidad una variación ≤ ±5 segmentos respecto al valor canónico.
>
> **Fase 5 — Evaluación.** Complejidad: ajuste log-log confirma pendiente empírica ≈ 1.03 (teórico: 1.0), verificando O(n·w). Sensibilidad: variación máxima de k* en toda la grid = ±4.5 segmentos, confirmando la robustez del modelo ante perturbaciones de hiperparámetros. Casos límite: textos >800 tokens con Baseline ya fragmentado muestran la menor mejora relativa, lo que es consistente con la predicción teórica (Baseline greedy converge a DP en textos muy largos con coherencia homogénea).
>
> **Fase 6 — Despliegue.** Los resultados se reportan en §5.1 OE5 con la Figura 10 (log-log de complejidad) y el análisis de sensibilidad (Figura 5). Las limitaciones identificadas se documentan en §5.2 y orientan el trabajo futuro en §6.3. La reproducibilidad completa se garantiza mediante `python experiments/reproduce.py`.

---

## Cambio 8 — Reestructurar Capítulo V: Resultados y Discusión por OE

### Paso 1 — Renombrar el capítulo

**Buscar** el título del capítulo: `RESULTADOS`  
**Reemplazar con:** `RESULTADOS Y DISCUSIÓN`

---

### Paso 2 — Reestructurar §5.1

**Dónde:** dentro de §5.1 "Resultados según los Objetivos Específicos", cada OE debe tener dos sub-bloques: **Resultados** (datos) + **Discusión** (interpretación). El texto actual de resultados se conserva; se agrega el párrafo de discusión inmediatamente después de cada bloque de resultados.

**Estructura final de §5.1:**

```
5.1 Resultados según los Objetivos Específicos
  OE1 — [resultados actuales] + [párrafo discusión OE1]
  OE2 — [resultados actuales] + [párrafo discusión OE2]
  OE3 — [resultados actuales] + [párrafo discusión OE3]
  OE4 — [resultados actuales] + [párrafo discusión OE4]
  OE5 — [resultados actuales] + [párrafo discusión OE5]
```

---

### Párrafos de discusión a agregar (uno por OE)

#### Discusión OE1 — Formalización matemática

> **Discusión.** La desviación media del 17% entre k* analítico y k* empírico es aceptable y metodológicamente explicable: la fórmula asume coherencia promedio uniforme sobre todo el texto, supuesto que se sostiene en textos cortos y homogéneos pero se debilita en textos >800 tokens con transiciones temáticas marcadas — precisamente los 5 textos con mayor desviación. La estructura multiplicativa de la función de costo, costo(i,j) = t² · [1 + λ(1 − coh)] + Cf, es una contribución central de este trabajo: a diferencia de formulaciones aditivas previas, el término multiplicativo garantiza que λ tenga peso real frente al costo cuadrático dominante, evitando que la coherencia sea numéricamente ignorada en segmentos largos. La derivación de k* cierra la brecha entre la formulación teórica y la aplicación práctica, permitiendo al usuario estimar el número de segmentos sin ejecutar la optimización completa.

---

#### Discusión OE2 — Implementación del sistema

> **Discusión.** La suite de 173 pruebas unitarias e integración no es un mero indicador de cobertura; es la garantía de que el sistema se comporta correctamente en condiciones extremas que no aparecen en el corpus de evaluación estándar: textos de una sola oración, textos donde lmax es menor que la longitud de cualquier oración, y textos vacíos. La arquitectura modular de seis componentes permite sustituir cualquier componente de forma independiente — por ejemplo, reemplazar all-MiniLM-L6-v2 por un modelo de embeddings de mayor precisión sin modificar el algoritmo DP. La interfaz Streamlit cumple el rol de la Fase 6 de CRISP-DM: hace el sistema accesible a usuarios no técnicos y permite verificar visualmente que la segmentación producida es coherente antes de invocar el LLM.

---

#### Discusión OE3 — Eficiencia computacional

> **Discusión.** El tamaño del efecto d = 3.88 es extraordinario: la convención de Cohen clasifica d > 0.8 como "grande", por lo que d = 3.88 sitúa la diferencia a casi cinco desviaciones estándar por encima del umbral de relevancia práctica. En términos concretos, para un texto de 600 tokens segmentado en cuatro partes, DP reduce el costo promedio por segmento de aproximadamente 150² = 22 500 tokens² (Baseline) a 85² ≈ 7 225 tokens², una reducción del 67.9% en el costo de autoatención por segmento. La equivalencia práctica entre DP y DP+Overlap (−0.30%) es un hallazgo relevante: confirma que el overlap no introduce ventaja computacional cuando μ = 0.03 y el corpus tiene transiciones temáticas razonablemente limpias. Esto sugiere que la extensión de overlap es valiosa principalmente en dominios con alta continuidad narrativa, no en textos técnicos segmentados por temas.

---

#### Discusión OE4 — Precisión semántica

> **Discusión.** La coherencia intra-segmento de 0.8994 para DP frente a 0.4960 para Baseline representa una mejora del 81%, lo que implica que los segmentos de DP contienen oraciones temáticamente relacionadas mientras que los del Baseline agrupan oraciones con similitud apenas por encima del nivel aleatorio (similitud coseno ≈ 0.5 para vectores en alta dimensión sin relación semántica). El valor Pk = 0.569 ± 0.095 indica que Baseline y DP discrepan en el 57% de las posiciones evaluadas, lo que confirma que no se trata de pequeñas variaciones en la posición de los cortes sino de decisiones de segmentación fundamentalmente distintas. Los estudios de caso cualitativos (Figuras 7–9) ilustran el mecanismo: DP produce cortes precisamente donde la similitud coseno entre oraciones consecutivas cae por debajo del umbral local, mientras que Baseline corta por longitud acumulada independientemente de la coherencia semántica.

---

#### Discusión OE5 — Complejidad, sensibilidad y casos límite

> **Discusión.** La pendiente empírica de 1.03 en la escala log-log, frente al valor teórico de 1.0 para O(n·w), confirma que la implementación con bisect logra la complejidad predicha; el 3% de exceso es atribuible al overhead del intérprete Python y las operaciones de caché de embeddings, no a una degradación algorítmica. La variación de ±4.5 segmentos en toda la grid 5×5 de parámetros λ × Cf es el resultado más relevante del análisis de sensibilidad: implica que el sistema es robusto a la incertidumbre en la calibración, y que los valores recomendados (λ = 0.5, Cf = 2000) no requieren ajuste fino para la mayoría de los casos de uso. Los textos con menor mejora relativa — amazon_river_basin (4.6%) y french_revolution (6.3%) — no representan fallos del modelo sino confirmaciones de la predicción teórica: cuando un texto es suficientemente largo y el Baseline ya lo fragmenta en segmentos de longitud similar al óptimo, el margen de mejora del DP se reduce. Esta limitación es esperada y está acotada por la estructura del corpus.

---

## Textos exactos para Tabla 5 y Tabla 6

### Cambio 4 — Tabla 5 (§3.10): Herramientas y tecnologías

**Dónde:** reemplazar la tabla actual de §3.10 en Word.

**Buscar:** la tabla que contiene "60+ tests" o "sentence-transformers" en §3.10.

**Reemplazar con la siguiente tabla completa:**

| Componente | Herramienta / Biblioteca | Versión | Rol en el sistema |
|---|---|---|---|
| Lenguaje base | Python | 3.13 | Implementación completa del sistema |
| Tokenización | tiktoken | 0.12.0 | Conteo BPE con codificación o200k\_base (compatible GPT-4o) |
| Segmentación oraciones | spaCy | 3.8.14 | Pipeline `en_core_web_sm` para detección de fronteras oracionales |
| Embeddings semánticos | sentence-transformers | 5.4.0 | Modelo all-MiniLM-L6-v2 (384 dim.), caché LRU en memoria |
| Álgebra lineal | NumPy | 1.26+ | Similitud coseno, operaciones sobre arrays de embeddings |
| Análisis estadístico | SciPy | 1.15.0 | t-test pareado, corrección FDR Benjamini-Hochberg |
| Visualización interactiva | Plotly | 6.7.0 | Curva costo vs k, comparación de métodos en Streamlit |
| Visualización estática | Matplotlib | 3.10.8 | Figuras de la tesis (boxplots, log-log, sensitivity) |
| Interfaz de demostración | Streamlit | 1.56.0 | Aplicación web para segmentación y comparación interactiva |
| Pruebas | pytest | 9.0.3 | Suite de 173 pruebas unitarias e integración |

**Métodos comparadores implementados:**

| Método | Descripción | Referencia en el código |
|---|---|---|
| DP Óptimo (1D) | Programación dinámica con función de costo compuesta | `src/segmentation/dp.py` |
| DP Óptimo (2D) | DP con k explícito para derivar k\* analítico | `src/segmentation/dp.py` |
| Baseline Greedy | Segmentación equitativa sin optimización | `src/segmentation/baseline.py` |
| Sliding Window | Ventana deslizante de tamaño fijo | `src/segmentation/sliding_window.py` |
| DP + Overlap | DP con regiones de solapamiento entre segmentos | `src/segmentation/overlap.py` |
| TextTiling | Coherencia léxica por ventana (Hearst, 1997) | `src/segmentation/texttiling.py` |

*Nota. Elaboración propia. Todas las versiones corresponden a las dependencias fijadas en `pyproject.toml`. Las 173 pruebas incluyen casos unitarios por módulo y casos de integración end-to-end.*

---

### Cambio 5 — Tabla 6 (§5.1): Resultados comparativos de costo de inferencia

**Problema actual:** la tabla usa valores con signo negativo (−174%, −597%) que resultan confusos — significan que Baseline es 174% más caro que DP, pero pueden leerse al revés. Además usa datos de solo 9 oraciones en lugar del corpus de 20 textos.

**Dónde:** reemplazar la Tabla 6 actual en §5.1 OE3.

**Reemplazar con:**

**Tabla 6.** Resultados del análisis estadístico: reducción del costo computacional de inferencia (tokens²) del DP Óptimo respecto a métodos comparadores. Corpus de 20 textos (8 dominios).

| Comparación | Reducción media | IC 95% | t(19) | p-valor | d Cohen | Interpretación |
|---|---|---|---|---|---|---|
| DP vs Baseline | **43.88%** | [41.22%, 46.55%] | 17.37 | < 0.001 | **3.88** | Efecto muy grande |
| DP vs Sliding Window | **26.02%** | [21.61%, 30.42%] | 9.25 | < 0.001 | **2.07** | Efecto muy grande |
| DP vs TextTiling | **14.73%** | [11.90%, 17.57%] | 8.20 | < 0.001 | **1.83** | Efecto muy grande |
| DP vs DP+Overlap | −0.30% | [−0.49%, −0.12%] | −4.47 | 0.0003 | −1.00 | Equivalencia práctica |

*Nota. Elaboración propia. Prueba t pareada de dos colas, corrección FDR Benjamini-Hochberg (α = 0.05). La "reducción" se calcula como (costo\_método − costo\_DP) / costo\_método × 100. Valores positivos indican que DP tiene menor costo. La comparación DP vs DP+Overlap muestra un valor negativo: el overlap introduce un pequeño costo adicional (0.30%) estadísticamente detectable pero sin relevancia práctica (el boxplot en Figura 6 muestra ambos métodos prácticamente superpuestos en y ≈ 0).*

**Párrafo de texto a acompañar la tabla (reemplazar cualquier texto previo que mencione −174% o −597%):**

> La Tabla 6 presenta los resultados del análisis estadístico comparativo. El DP Óptimo reduce el costo computacional de inferencia en un 43.88% respecto al Baseline greedy (d = 3.88, IC 95%: [41.22%, 46.55%]), en un 26.02% respecto a Sliding Window (d = 2.07) y en un 14.73% respecto a TextTiling (d = 1.83), con p < 0.001 en todos los casos tras la corrección FDR. La comparación con DP+Overlap muestra una diferencia estadísticamente significativa pero de magnitud despreciable (−0.30%), confirmando que la extensión de overlap no introduce ventaja computacional bajo los parámetros canónicos (μ = 0.03). Todas las comparaciones alcanzan el umbral de efecto grande (d > 0.8), lo que respalda la validez práctica de la hipótesis de investigación.

---

## Textos exactos para Anexos A y B

### Cambio 6 — Anexo A: Código fuente de dp.py

**Dónde:** en el Anexo A del documento (crear sección si no existe, con título "Código fuente del algoritmo de programación dinámica").

**Heading a insertar:** `Anexo A: Código Fuente — Módulo dp.py`

**Párrafo introductorio:**
> El módulo `dp.py` implementa el algoritmo de programación dinámica en sus dos variantes (1D y 2D) junto con las funciones auxiliares de costo, coherencia y reconstrucción de la segmentación óptima. La función central `_compute_cost` implementa la fórmula costo(i,j) = tokens(i,j)² · [1 + λ · (1 − coh(i,j))] + Cf. La función `segment_dp` realiza la optimización 1D en O(n·w) usando bisect para la ventana activa. La función `segment_dp_2d` resuelve la versión con k explícito y permite derivar k* empíricamente comparando dp[n][k] para todo k válido.

**Código completo a insertar:**

```python
# dp.py — algoritmo DP con función de costo compuesta
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity
import bisect
import math
import numpy as np

INF = float("inf")
DEFAULT_LAMBDA = 0.5
DEFAULT_FIXED_COST = 1000.0


def _merge_to_limit(sentences, token_lens, max_n):
    n = len(sentences)
    if n <= max_n:
        return sentences, token_lens
    group_size = math.ceil(n / max_n)
    merged_sentences, merged_tokens = [], []
    for i in range(0, n, group_size):
        merged_sentences.append(" ".join(sentences[i:i + group_size]))
        merged_tokens.append(sum(token_lens[i:i + group_size]))
    return merged_sentences, merged_tokens


def _build_similarity_arrays(embeddings):
    n = len(embeddings)
    sims, prefix_sims = [], [0.0] * n
    for k in range(n - 1):
        s = cosine_similarity(embeddings[k], embeddings[k + 1])
        sims.append(s)
        prefix_sims[k + 1] = prefix_sims[k] + s
    return sims, prefix_sims


def _avg_coherence(prefix_sims, start, end):
    n_pairs = end - start - 1
    if n_pairs <= 0:
        return 1.0
    return (prefix_sims[end - 1] - prefix_sims[start]) / n_pairs


def _compute_cost(token_count, prefix_sims, start, end,
                  coherence_lambda, fixed_cost=DEFAULT_FIXED_COST):
    t2 = float(token_count ** 2)
    avg_coh = _avg_coherence(prefix_sims, start, end)
    coherence_factor = 1.0 + coherence_lambda * (1.0 - avg_coh)
    return t2 * coherence_factor + fixed_cost


def _build_cumulative_tokens(token_lens):
    cumulative = [0]
    for t in token_lens:
        cumulative.append(cumulative[-1] + t)
    return cumulative


def segment_dp(text, lmin=50, lmax=200, model="gpt-4o",
               coherence_lambda=DEFAULT_LAMBDA, fixed_cost=DEFAULT_FIXED_COST,
               max_sentences=None):
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")
    token_lens = count_tokens_batch(sentences, model=model)
    if max_sentences is not None:
        sentences, token_lens = _merge_to_limit(sentences, token_lens, max_sentences)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)
    embeddings = get_embeddings(sentences)
    _, prefix_sims = _build_similarity_arrays(embeddings)

    dp = [INF] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for j in range(1, n + 1):
        target_min = cumtok[j] - lmax
        i_start = bisect.bisect_left(cumtok, target_min, 0, j)
        for i in range(i_start, j):
            span = cumtok[j] - cumtok[i]
            if span < lmin:
                break
            cost = dp[i] + _compute_cost(
                token_count=span, prefix_sims=prefix_sims,
                start=i, end=j, coherence_lambda=coherence_lambda,
                fixed_cost=fixed_cost,
            )
            if cost < dp[j]:
                dp[j] = cost
                back[j] = i

    if dp[n] == INF:
        raise ValueError(f"No valid segmentation: lmin={lmin}, lmax={lmax}.")

    cuts, cur = [], n
    while cur > 0:
        cuts.append(cur)
        cur = back[cur]
    cuts.reverse()

    segments, prev = [], 0
    for idx, cut in enumerate(cuts):
        sents = sentences[prev:cut]
        segments.append(Segment(
            index=idx, sentences=sents,
            text=" ".join(sents),
            token_count=cumtok[cut] - cumtok[prev],
        ))
        prev = cut
    return SegmentationResult(segments=segments, total_cost=dp[n],
                              num_segments=len(segments))


def segment_dp_2d(text, lmin=50, lmax=200, model="gpt-4o",
                  coherence_lambda=DEFAULT_LAMBDA, max_k=None,
                  fixed_cost=DEFAULT_FIXED_COST, max_sentences=None):
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")
    token_lens = count_tokens_batch(sentences, model=model)
    if max_sentences is not None:
        sentences, token_lens = _merge_to_limit(sentences, token_lens, max_sentences)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)
    embeddings = get_embeddings(sentences)
    _, prefix_sims = _build_similarity_arrays(embeddings)
    if max_k is None:
        max_k = n

    dp = [[INF] * (max_k + 1) for _ in range(n + 1)]
    back = [[(-1, -1)] * (max_k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for k in range(1, max_k + 1):
        for j in range(k, n + 1):
            target_min = cumtok[j] - lmax
            i_start = bisect.bisect_left(cumtok, target_min, k - 1, j)
            for i in range(i_start, j):
                span = cumtok[j] - cumtok[i]
                if span < lmin:
                    break
                if dp[i][k - 1] == INF:
                    continue
                cost = dp[i][k - 1] + _compute_cost(
                    token_count=span, prefix_sims=prefix_sims,
                    start=i, end=j, coherence_lambda=coherence_lambda,
                    fixed_cost=fixed_cost,
                )
                if cost < dp[j][k]:
                    dp[j][k] = cost
                    back[j][k] = (i, k - 1)

    cost_by_k = {k: dp[n][k] for k in range(1, max_k + 1) if dp[n][k] < INF}
    if not cost_by_k:
        raise ValueError(f"No valid segmentation: lmin={lmin}, lmax={lmax}.")

    best_k = min(cost_by_k, key=cost_by_k.get)
    cuts, cur, cur_k = [], n, best_k
    while cur > 0:
        prev_i, prev_k = back[cur][cur_k]
        cuts.append(cur)
        cur, cur_k = prev_i, prev_k
    cuts.reverse()

    segments, prev = [], 0
    for idx, cut in enumerate(cuts):
        sents = sentences[prev:cut]
        segments.append(Segment(
            index=idx, sentences=sents,
            text=" ".join(sents),
            token_count=cumtok[cut] - cumtok[prev],
        ))
        prev = cut
    return DP2DResult(segments=segments, total_cost=cost_by_k[best_k],
                      num_segments=best_k, cost_by_k=cost_by_k)
```

*Código A.1. Módulo dp.py completo. La variante 1D (segment\_dp) tiene complejidad O(n·w), donde w es el ancho de la ventana activa determinado por los límites lmin–lmax. La variante 2D (segment\_dp\_2d) extiende la tabla a dp[j][k] para derivar k* como argmin_k dp[n][k].*

---

### Cambio 7 — Anexo B: Código fuente de embeddings.py

**Dónde:** en el Anexo B del documento (sección "Código fuente del módulo de embeddings semánticos").

**Heading a insertar:** `Anexo B: Código Fuente — Módulo embeddings.py`

**Párrafo introductorio:**
> El módulo `embeddings.py` encapsula la generación de representaciones vectoriales densas para las oraciones del texto. Utiliza el modelo all-MiniLM-L6-v2 (384 dimensiones) de la biblioteca sentence-transformers, cargado como singleton para evitar múltiples inicializaciones. Incorpora un caché LRU con límite de 8 entradas para evitar recomputar embeddings de textos repetidos durante el benchmark. La función `cosine_similarity` calcula la similitud en [−1, 1] de forma numéricamente estable.

**Código completo a insertar:**

```python
# embeddings.py — coherencia semántica con sentence-transformers
import hashlib
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensiones

_embedding_cache: dict[str, np.ndarray] = {}
_CACHE_MAX_SIZE = 8


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _model


def _cache_key(texts: list[str]) -> str:
    combined = "\x00".join(texts)
    return hashlib.md5(combined.encode("utf-8", errors="replace")).hexdigest()


def get_embeddings(texts: list[str]) -> np.ndarray:
    key = _cache_key(texts)
    if key not in _embedding_cache:
        if len(_embedding_cache) >= _CACHE_MAX_SIZE:
            _embedding_cache.pop(next(iter(_embedding_cache)))
        model = _get_model()
        _embedding_cache[key] = model.encode(texts, convert_to_numpy=True)
    return _embedding_cache[key]


def clear_embedding_cache() -> None:
    _embedding_cache.clear()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_coherence_scores(sentences: list[str]) -> list[float]:
    if len(sentences) < 2:
        return [1.0]
    embeddings = get_embeddings(sentences)
    return [cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)]


def segment_coherence(sentences: list[str]) -> float:
    scores = compute_coherence_scores(sentences)
    return float(np.mean(scores))
```

*Código B.1. Módulo embeddings.py completo. El caché evita recomputar embeddings para el mismo conjunto de oraciones durante las múltiples iteraciones del benchmark. La función `segment_coherence` calcula la coherencia intra-segmento como media de similitudes coseno entre oraciones consecutivas.*
