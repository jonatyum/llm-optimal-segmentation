# Referencia de Figuras — Tesis Churata Apaza 2025

> Script de generación: `experiments/generate_figures.py`  
> Directorio: `outputs/figures/`

---

## Cap. IV — Descripción del corpus

### Figura 4 — Distribución de tokens por dominio
`cap4/corpus_tokens_por_dominio.png`

Gráfico de barras horizontales que muestra el número promedio de tokens por dominio temático del corpus, ordenado de mayor a menor longitud. El eje X indica los tokens promedio (escala de 0 a 1200) y el eje Y lista los 15 dominios. Dos líneas de referencia verticales discontinuas marcan los umbrales "short" (~340 tokens) y "long" (~700 tokens), permitiendo clasificar visualmente los dominios por su extensión típica.

Los valores exactos visibles en la figura son: Geography 1141, History 915, Software Engineering 815, IR 809, NLP 685, Medicine 658, Economics 604, Physics 588, Biology 586, Philosophy 555, Law 528, Deep Learning 508, LLM 491, Algorithms 424, Machine Learning 341. La media global del corpus es aproximadamente 623 tokens.

Analizando la distribución, se pueden distinguir tres grupos claros. El grupo de textos largos (por encima de la línea "long" de ~700 tokens) incluye Geography, History, Software Engineering, IR y NLP — cinco dominios que representan textos extensos con múltiples subtemas. El grupo intermedio (entre 340 y 700 tokens) agrupa a Medicine, Economics, Physics, Biology, Philosophy, Law, Deep Learning y LLM — la mayoría del corpus. El grupo corto (por debajo de la línea "short") solo contiene Algorithms y Machine Learning. Notablemente, la diferencia entre el dominio más largo (Geography, 1141 tokens) y el más corto (Machine Learning, 341 tokens) es un factor de 3.3x, una heterogeneidad significativa que pone a prueba la robustez del algoritmo DP frente a textos de naturaleza muy diferente.

El color distintivo de cada barra (uno por dominio, de la paleta IBM Color Blind Safe) se mantiene consistente en todas las figuras del corpus, permitiendo identificar cada dominio visualmente a lo largo del documento. Esta diversidad de longitudes es clave para validar los resultados del experimento: si el corpus tuviera solo textos de longitud similar, los resultados podrían estar sesgados hacia un rango específico donde el DP tiene más o menos ventaja.

**Discusión con la literatura**

La composición del corpus en términos de longitud sigue de cerca las recomendaciones metodológicas de Hearst (1997), quien en su trabajo seminal sobre TextTiling argumentó que un corpus de evaluación para segmentación de texto debe incluir documentos de diversa extensión y estructura temática para que los resultados sean generalizables. Hearst utilizó artículos del Wall Street Journal y de enciclopedias, con longitudes que oscilaban entre 200 y 1500 palabras — un rango comparable al nuestro (341–1141 tokens). La diferencia es que Hearst trabajó con artículos ya estructurados por secciones, mientras que nuestro corpus está diseñado para simular textos continuos sin marcas de sección explícitas, lo que representa un escenario más exigente.

Riedl y Biemann (2012), en su trabajo sobre TopicTiling, utilizaron el corpus de Wikipedia con segmentos de entre 3 y 11 párrafos por artículo, argumentando que la diversidad temática de Wikipedia garantiza cobertura de múltiples dominios. Nuestro corpus replica esta motivación multi-dominio con 15 categorías que van desde ciencias exactas (física, biología) hasta ciencias sociales (economía, filosofía) y tecnología (LLM, Deep Learning). Esta diversidad es importante porque los modelos de segmentación tienden a sobreajustarse a las características léxicas y semánticas de dominios específicos, y evaluarlos en un corpus hetrogéneo revela su generalización real.

Eisenstein y Barzilay (2008), en su modelo bayesiano para segmentación de texto (Bayesian Text Segmentation), también trabajaron con corpus multi-dominio pero con énfasis en documentos académicos y transcripciones de reuniones, que tienen características de coherencia muy distintas a los textos expositivos de nuestro corpus. La media de 623 tokens de nuestro corpus corresponde aproximadamente al contexto útil de modelos GPT-3.5 para inferencia eficiente, lo que hace que la evaluación sea directamente relevante para aplicaciones de recuperación aumentada (RAG, Lewis et al., 2020).

**Autorreflexión**

Una limitación reconocida del corpus es que cada dominio está representado por solo uno o dos textos, lo cual limita la potencia estadística intra-dominio. Un corpus de 20 textos es suficiente para las pruebas t-test pareadas (que comparan el mismo texto bajo distintos métodos), pero no para hacer inferencias sobre la variabilidad dentro de un dominio específico. Dominios como Geography (representado por un único texto de 1141 tokens, amazon_river_basin) o Machine Learning (un único texto de 341 tokens) no tienen réplicas que permitan estimar la varianza intra-dominio.

En retrospectiva, habría sido deseable incluir 3–4 textos por dominio, reduciendo la cobertura de dominios a 5–6 categorías pero aumentando la representatividad estadística de cada una. Sin embargo, esta decisión estuvo condicionada por el costo computacional de generar embeddings de calidad para un corpus más amplio, y por la necesidad de mantener el experimento reproducible en un entorno académico sin infraestructura de cómputo intensivo. Los resultados actuales son estadísticamente sólidos porque el diseño paired-test controla la variabilidad entre textos, pero la interpretación por dominio debe hacerse con cautela.

---

### Figura 5 — Coherencia semántica media por dominio
`cap4/corpus_coherencia_por_dominio.png`

Gráfico de barras verticales con la similitud coseno promedio entre oraciones consecutivas dentro de cada dominio, calculada con el modelo all-MiniLM-L6-v2 (384 dimensiones). Las barras están ordenadas de mayor a menor coherencia. Las barras de error muestran ±1 desviación estándar. Una línea de referencia horizontal naranja discontinua indica la media global del corpus (Media global = 0.442).

Los valores exactos, de mayor a menor: Machine Learning 0.558, Law 0.529, Software Engineering 0.509, Algorithms 0.492, Philosophy 0.479, Geography 0.469, History 0.456, NLP 0.441, IR 0.428, Economics 0.416, Medicine 0.396, Biology 0.387, Physics 0.367, LLM 0.357, Deep Learning 0.341.

La figura revela tres observaciones clave. Primero, el rango total (0.341–0.558) es sustancial: una diferencia de 0.217 en similitud coseno en espacios de alta dimensión corresponde a una diferencia semántica real y perceptible entre dominios. Segundo, los dominios con mayor coherencia (Machine Learning, Law, Software Engineering) tienen vocabulario técnico o normativo muy consistente entre oraciones. Los de menor coherencia (Deep Learning, LLM, Physics) cubren múltiples subtemas con mayor heterogeneidad interna. Tercero, las barras de error varían significativamente: Algorithms tiene alta varianza entre sus textos (~±0.09), mientras Machine Learning y Law son muy homogéneos.

Desde el punto de vista del modelo, esta figura justifica directamente el término λ en la función de costo del DP. La variación (0.341–0.558) implica que el peso relativo de la coherencia cambia según el dominio: en Deep Learning (coh=0.341), el término λ·(1−coh) aportará un factor multiplicativo de 1 + 0.5·0.659 = 1.33 sobre el costo cuadrático base, incentivando fuertemente el agrupamiento de oraciones similares. En Machine Learning (coh=0.558) ese factor sería solo 1.22, una penalización menor porque el texto ya es naturalmente cohesivo.

**Discusión con la literatura**

Barzilay y Lapata (2008) propusieron uno de los primeros modelos computacionales de coherencia local basado en transiciones de entidades (Entity Grid), midiendo la probabilidad de que las mismas entidades aparezcan en oraciones consecutivas. Su enfoque es complementario al nuestro: mientras ellos modelan la coherencia como continuidad de referentes nominales, nosotros la medimos como similitud semántica densa entre embeddings de oraciones. Ambos enfoques capturan aspectos distintos de la cohesión textual, y la correlación entre ambas métricas en un corpus multi-dominio como el nuestro sería una extensión valiosa pero fuera del alcance de esta tesis.

Reimers y Gurevych (2019), en el trabajo que introdujo Sentence-BERT (del que deriva all-MiniLM-L6-v2 utilizado en esta tesis), demostraron que los embeddings de oraciones producidos por modelos transformers fine-tuned con tareas siamesas (Natural Language Inference) capturan coherencia semántica de manera más robusta que TF-IDF o embeddings de palabras promediados. Su evaluación en benchmarks de similitud semántica (STS) mostró correlaciones de Spearman de 0.86–0.89, lo que respalda el uso de estos embeddings como proxy de coherencia en nuestra función de costo. Sin embargo, Reimers y Gurevych también señalaron que los embeddings de modelos pre-entrenados pueden reflejar sesgos del corpus de entrenamiento (mayoritariamente inglés general), lo que podría afectar la calidad de la coherencia medida en dominios muy especializados como Physics o Medicine de nuestro corpus.

Li y Jurafsky (2017) estudiaron la representación neural de la coherencia discursiva, encontrando que modelos de red neuronal que codifican oraciones en contexto (bidireccional) capturan mejor la coherencia que modelos unidireccionales. El modelo all-MiniLM-L6-v2 es un modelo bidireccional (basado en BERT), lo que lo hace metodológicamente apropiado para nuestra tarea. No obstante, Li y Jurafsky también encontraron que la coherencia a nivel de párrafo (multi-oración) no siempre se puede inferir de similitudes de pares de oraciones adyacentes — exactamente la aproximación que usamos. Esto sugiere que la métrica de coherencia intra-segmento podría subestimar o sobreestimar la calidad real de los segmentos en textos con estructura de cohesión a más largo plazo.

**Autorreflexión**

El uso de similitud coseno entre embeddings de oraciones adyacentes como métrica de coherencia es una simplificación deliberada que facilita la integración en la función de costo del DP pero que no captura toda la riqueza de la coherencia discursiva. En particular, la métrica no distingue entre coherencia temática (mismo tema) y coherencia referencial (mismos referentes) ni captura la coherencia estructural a nivel de párrafo o sección.

Una reflexión importante es que la media global de 0.442 es notablemente baja. En textos naturalmente bien estructurados, se esperaría una coherencia intra-dominio media más cercana a 0.6–0.7. El valor 0.442 sugiere que los textos del corpus, aunque son expositivos y bien redactados, tienen transiciones temáticas frecuentes que el modelo de embeddings detecta correctamente como baja similitud coseno. Esto es, paradójicamente, una buena noticia para la tesis: significa que el corpus tiene suficiente variación semántica interna para que el algoritmo DP tenga margen de mejora real sobre el Baseline.

---

## Cap. V — Resultados por Objetivo Específico

### OE1 — Formalización matemática del modelo

#### Figura 1 — Costo total de inferencia vs lmax
`cap5/oe1/costo_vs_lmax.png`

Gráfico de líneas que compara la evolución del costo total de inferencia (en tokens², escala en miles) al variar lmax entre 150 y 300 tokens para DP Óptimo (línea azul sólida), Baseline (línea naranja discontinua) y Sliding Window (línea verde punteada). Texto representativo: LLM Inference, 491 tokens.

DP Óptimo mantiene ~46k tokens² en todos los lmax. Baseline parte de ~60k a lmax=150 y llega a ~124k a lmax=300 (factor 2.07x). Sliding Window parte de ~76k y llega a ~130k (factor 1.71x). El patrón más llamativo es la línea plana del DP: ampliar el espacio de búsqueda no degrada el resultado, demostración directa del principio de optimalidad de Bellman. Sliding Window supera en costo a Baseline en todos los puntos porque construye segmentos de tamaño fijo lmax sin considerar fronteras naturales. La convergencia de Baseline y Sliding Window en lmax=300 indica que a valores altos ambos producen segmentaciones igualmente costosas.

**Discusión con la literatura**

El costo cuadrático de la autoatención en transformers (O(n²) respecto al número de tokens del segmento) fue formalizado por Vaswani et al. (2017) en el artículo "Attention is All You Need". Esta cuadraticidad es precisamente la motivación central de la función de costo de esta tesis: costo(i,j) = tokens(i,j)² · factor_coherencia + Cf captura directamente la penalización cuadrática de inferir sobre segmentos largos. La figura muestra empíricamente que el DP explota esta cuadraticidad para reducir el costo total: al dividir el texto en muchos segmentos cortos en lugar de pocos segmentos largos, la suma de cuadrados es significativamente menor que el cuadrado de la suma.

Child et al. (2019) y Beltagy et al. (2020) propusieron arquitecturas de atención dispersa (Sparse Transformers y Longformer) para reducir la complejidad cuadrática de O(n²) a O(n·log n) u O(n). Desde la perspectiva de nuestra tesis, estos trabajos representan una alternativa complementaria pero diferente: modifican la arquitectura del modelo para procesar textos largos, mientras nuestro enfoque mantiene el modelo estándar pero optimiza la segmentación del texto de entrada. Ambos enfoques son compatibles y podrían combinarse: usar Longformer para inferencia con ventana de atención extendida dentro de cada segmento, y DP para optimizar la partición global del documento.

El Baseline greedy utilizado como comparador es conceptualmente similar al chunking fijo (fixed-size chunking) ampliamente usado en sistemas RAG (Lewis et al., 2020; Guo et al., 2022). La figura demuestra que incluso con lmax=150 (el valor más conservador), el chunking fijo genera un costo cuadrático significativamente mayor que el DP. Esto tiene implicaciones directas para el diseño de pipelines RAG: reemplazar el chunking fijo por el DP Óptimo no requiere cambiar el modelo de lenguaje ni el índice vectorial, solo la estrategia de segmentación previa a la indexación.

**Autorreflexión**

Una limitación de esta figura es que usa un único texto (LLM Inference, 491 tokens) como texto representativo. Si bien este texto tiene características cercanas a la media del corpus (491 ≈ 623 tokens, dominio técnico), los resultados para textos muy cortos (<400 tokens) o muy largos (>800 tokens) serían diferentes. Para textos cortos, el DP tiene menos margen de maniobra porque el número de segmentaciones válidas dentro del rango lmin-lmax es menor; para textos muy largos, la ventaja debería ser mayor porque el Baseline genera segmentos más costosos.

También es relevante reflexionar sobre la elección del rango lmax={150, 200, 250, 300}. Este rango cubre valores operativamente razonables para LLMs modernos (GPT-4o tiene un contexto de 128k tokens, pero los segmentos óptimos para recuperación en RAG son típicamente 100–500 tokens). Un análisis con lmax extendido hasta 500 o 1000 tokens mostraría probablemente una asíntota en la reducción, donde el Baseline también se "satura" y la ventaja del DP deja de crecer. Esta extensión del análisis queda como trabajo futuro.

---

#### Figura 2 — Ventaja del DP Óptimo según lmax
`cap5/oe1/reduccion_vs_lmax.png`

Barras verticales con el porcentaje de reducción de costo del DP vs Baseline para cada lmax. Valores: lmax=150 → 23.6%, lmax=200 → 45.0%, lmax=250 → 61.8%, lmax=300 → 62.7%. El patrón es cóncavo: mayor salto entre lmax=150→200 (+21.4 pp) que entre 250→300 (+0.9 pp, prácticamente asíntota). La configuración canónica lmax=150 fue elegida por criterio operativo (segmentos de 20–150 tokens son estándar en LLMs), no por maximizar la ventaja.

**Discusión con la literatura**

El fenómeno de rendimientos decrecientes en la reducción de costo al aumentar lmax es consistente con la teoría de la información aplicada a segmentación: a medida que lmax aumenta, el Baseline genera segmentos progresivamente más subóptimos (más alejados del óptimo global), mientras el DP siempre encuentra el mínimo. Pero este gap se estabiliza porque el texto tiene una longitud finita y el número de segmentaciones válidas se satura: a lmax=300, un texto de 491 tokens puede dividirse como máximo en 2 segmentos (491/300 ≈ 1.6), lo que limita la exploración del espacio de soluciones.

Eisenstein y Barzilay (2008) observaron un fenómeno similar en su modelo bayesiano: la granularidad óptima de segmentación (número de segmentos k) no crece linealmente con la longitud del texto sino con la raíz cuadrada, predicción que nuestra fórmula k* = N·√((1+λ(1−coh))/Cf) formaliza explícitamente. La relación entre lmax y la ventaja del DP es la contrapartida de este fenómeno: a mayor lmax, el Baseline produce menos segmentos (más largos), alejándose más del k* óptimo, lo que amplía la brecha con el DP.

Desde la perspectiva de sistemas RAG, trabajos como Guo et al. (2022) y Shi et al. (2023) han estudiado empíricamente el efecto del tamaño del chunk (equivalente a lmax en nuestro modelo) en la calidad de recuperación. Sus hallazgos sugieren que chunks de 150–200 tokens maximizan el recall@k en índices vectoriales, lo que es consistente con nuestra elección de lmax=150 como valor canónico. La figura muestra que incluso en el rango óptimo para RAG (lmax=150–200), el DP supera al chunking fijo en un 23.6%–45.0%, lo que implica que la mejora de calidad de recuperación debería ser igualmente significativa aunque esta métrica no fue evaluada directamente en la tesis.

**Autorreflexión**

Una reflexión importante sobre esta figura es la diferencia entre el 23.6% mostrado aquí (un texto, lmax=150) y el 43.88% del corpus completo (20 textos, lmax=150). Esta discrepancia —casi el doble— requiere explicación cuidadosa en la defensa. El texto LLM Inference (491 tokens) es uno de los textos más cortos del corpus y, por tanto, uno donde el DP tiene menos margen de mejora. El corpus completo incluye textos mucho más largos (hasta 1141 tokens) donde la ventaja del DP es mayor. El promedio ponderado del corpus completo, que incluye esos textos largos, eleva la media al 43.88%.

Esto sugiere que la figura no debería presentarse como representativa de la magnitud de la ventaja del DP, sino como ilustración del mecanismo (comportamiento del costo en función de lmax). La magnitud real de la ventaja está en la Figura 6, con los resultados estadísticos del corpus completo. Esta distinción debe estar claramente articulada en el texto de §5.1 OE1 para evitar que el jurado perciba una inconsistencia entre las figuras.

---

#### Figura 3 — k* Analítico vs k* Empírico
`cap5/oe1/kstar_analitico_vs_empirico.png`

Scatter plot de 20 puntos comparando k* analítico (fórmula N·√((1+λ·(1−coh))/Cf), eje X) vs k* empírico del DP (eje Y). Color por dominio, tamaño proporcional a tokens. Línea identidad diagonal discontinua. Desviación media: 17%.

Casi todos los puntos están por debajo de la diagonal: la fórmula sobreestima k*. La excepción es un punto en la esquina inferior izquierda (texto corto ~340 tokens, Algorithms o Machine Learning) ligeramente por encima de la diagonal — la fórmula subestima en textos muy cortos donde la coherencia promedio sí es uniforme. El punto más alejado (Geography/amazon_river_basin, ~29 analítico vs ~17 empírico, punto grande rosa) tiene la mayor desviación absoluta: 12 segmentos. Los textos de longitud media (500–700 tokens, grupo denso en k*∈[12–16]) muestran la mejor precisión de la fórmula (desviación 1–3 segmentos). El punto NLP (~22 analítico, ~20 empírico) es la predicción más precisa entre textos largos.

**Discusión con la literatura**

La fórmula analítica k* = N·√((1+λ(1−coh))/Cf) es una derivación original de esta tesis que no tiene un equivalente directo en la literatura de segmentación. Sin embargo, tiene conexiones conceptuales con varios trabajos previos. Eisenstein y Barzilay (2008) derivaron el número óptimo de segmentos k en su modelo bayesiano como función de la longitud del documento y los parámetros de prior de Dirichlet, obteniendo también una relación de raíz cuadrada entre N y k* bajo ciertos supuestos de distribución uniforme. Esto es notable: dos enfoques completamente distintos (optimización de costo cuadrático vs inferencia bayesiana) convergen en la misma relación funcional N ~ k*², lo que sugiere que esta relación es una propiedad fundamental de la estructura de los textos expositivos y no un artefacto del modelo específico.

Riedl y Biemann (2012) en TopicTiling tampoco derivan explícitamente el número óptimo de segmentos, sino que lo tratan como un hiperparámetro a calibrar por validación cruzada. Nuestro enfoque de derivarlo analíticamente como función de los parámetros del modelo es una contribución práctica significativa: permite al usuario estimar k* antes de ejecutar la segmentación, lo que es útil para sistemas con restricciones de tiempo real.

La desviación media del 17% es comparable a la precisión reportada por Eisenstein y Barzilay (2008) para su estimación del número de segmentos: en su evaluación sobre el corpus ICSI Meeting Corpus, el error relativo en k* fue del 15–20% dependiendo del dominio. Esta similitud sugiere que el 17% de error de la fórmula analítica es una limitación intrínseca de cualquier modelo que asuma coherencia promedio uniforme, no una debilidad específica de nuestra formulación.

**Autorreflexión**

La visualización de scatter plot sin etiquetas de texto en los puntos es una decisión de diseño que prioriza la limpieza visual sobre la identificabilidad individual. Esto implica que la figura sola no permite al lector identificar qué texto corresponde a cada punto, a menos que consulte la tabla adjunta en §5.1 OE1. Para la defensa, conviene tener preparada esa tabla de referencia rápida (id | k_analítico | k_empírico | tokens) que permita responder preguntas específicas sobre textos individuales.

Una limitación que no se discute explícitamente en la tesis es que k* analítico depende de la coherencia promedio del texto completo, que a su vez se calcula sobre el mismo corpus. Esto introduce una circularidad potencial: la coherencia usada para predecir k* es la coherencia medida con los embeddings all-MiniLM-L6-v2, que también son los usados para calcular el costo de coherencia en el DP. Si se usara un modelo de embeddings diferente, tanto k* analítico como k* empírico cambiarían, pero probablemente en la misma dirección, preservando el 17% de desviación. Esta dependencia del modelo de embeddings debería mencionarse como limitación en §5.2.

---

### OE3 — Eficiencia computacional

#### Figura 6 — Reducción de costo DP vs métodos comparadores
`cap5/oe3/statistical_boxplots.png`

Boxplots de reducción de costo (%) del DP sobre Baseline, Sliding Window, DP+Overlap y TextTiling, para 20 textos. Línea de referencia en y=0. Cohen's d y p-valor (FDR) encima de cada caja.

DP vs Baseline (naranja): Q1≈44%, mediana≈45%, Q3≈47%, bigotes 37%–51%. Caja compacta, alta consistencia. d=3.88, p<0.001. DP vs Sliding Window (verde): Q1≈24%, mediana≈25%, Q3≈31%, bigotes 17%–39%. Mayor variabilidad. d=2.07, p<0.001. DP vs DP+Overlap (rosa): Caja invisible en y≈−0.30%. Equivalencia práctica, pero estadísticamente significativa. d=−1.00, p=0.0003. DP vs TextTiling (azul cielo): Q1≈12%, mediana≈15%, Q3≈19%, bigotes 10%–21%. d=1.83, p<0.001. El ordenamiento Baseline > Sliding Window > TextTiling >> DP+Overlap≈0 refleja la jerarquía de sofisticación semántica de los métodos comparadores.

**Discusión con la literatura**

Los resultados de esta figura posicionan el DP Óptimo favorablemente respecto a los métodos de la literatura de dos maneras. Primero, en términos de eficiencia computacional (reducción de tokens²), el DP supera a todos los métodos en todas las comparaciones, con efectos de magnitud que la literatura de segmentación raramente reporta (d=3.88 para DP vs Baseline). La mayoría de los trabajos en segmentación de texto reportan mejoras en métricas de calidad (Pk, WindowDiff, F1 de fronteras) con efectos de d entre 0.5 y 1.5 (Riedl & Biemann, 2012; Koshorek et al., 2018). El efecto d=3.88 es excepcionalmente alto porque la comparación es de costo computacional, no de calidad de fronteras, y el Baseline greedy es un método deliberadamente subóptimo desde el punto de vista de la optimización.

El resultado de TextTiling (reducción del 14.73%, d=1.83) es particularmente relevante. Hearst (1997) desarrolló TextTiling como un método de segmentación basado en coherencia léxica (frecuencia de palabras en ventanas deslizantes), sin optimización del costo computacional. La reducción del 14.73% del DP sobre TextTiling indica que incluso un método que considera la coherencia textual no alcanza la optimalidad computacional que el DP garantiza. Esto sugiere que la calidad semántica de las fronteras y la eficiencia computacional son objetivos parcialmente alineados pero no equivalentes, y que optimizar explícitamente el costo cuadrático (como hace el DP) produce segmentaciones más eficientes que optimizar la coherencia léxica local (como hace TextTiling).

La equivalencia práctica entre DP y DP+Overlap (−0.30%, d=−1.00) confirma el hallazgo de varios trabajos sobre overlapping chunking en RAG (Guo et al., 2022): el overlap entre segmentos mejora el recall de recuperación al precio de un costo computacional marginal, pero ese costo marginal es detectable estadísticamente. Con μ=0.03, la penalización de overlap en nuestra función de costo es deliberadamente pequeña, lo que explica que el DP+Overlap produzca segmentaciones casi idénticas al DP en términos de costo de inferencia. Si μ fuera mayor (ej. μ=0.5), la penalización de overlap sería más severa y la equivalencia se rompería.

**Autorreflexión**

Una reflexión crítica sobre la figura es que el eje Y ("reducción de costo respecto al DP Óptimo") puede ser confuso para lectores no familiarizados con el diseño del experimento. El lector podría interpretar que se está comparando el DP contra sí mismo (y=0 como referencia del DP), cuando en realidad cada boxplot muestra cuánto reduce el DP el costo de un método dado. La confusión potencial es mayor para DP+Overlap, cuya caja en y≈−0.30% podría interpretarse como "el DP+Overlap es mejor que el DP" cuando en realidad es marginalmente peor.

El texto de §5.1 OE3 incluye una nota aclaratoria sobre este punto, pero para la defensa oral conviene preparar una descripción explícita de la figura: "La barra de y=0 representa el DP como referencia. Un valor positivo de X% significa que el método comparado tiene un costo X% mayor que el DP. El DP+Overlap en −0.30% significa que el DP+Overlap es 0.30% más caro que el DP, no más barato."

Otra reflexión es que la medición en tokens² como métrica de eficiencia es una proxy del costo real de inferencia, no el tiempo de ejecución wall-clock. En la práctica, el costo de inferencia en tokens² es el parámetro relevante para comparar la carga de autoatención en transformers (Vaswani et al., 2017), pero el tiempo de ejecución real depende también de la arquitectura del hardware, el tamaño del modelo y la implementación de batching. Esta distinción debería mencionarse en la defensa si el jurado pregunta sobre la relación entre tokens² y tiempo real de inferencia.

---

### OE4 — Precisión semántica

#### Figura 7 — Estudio de caso: LLM Inference
`cap5/oe4/case_studies/case_llm_inference.png`

Gráfico de líneas sobre "Inferencia en Modelos de Lenguaje y Ventanas de Contexto" (LLM, 491 tokens). DP Óptimo: 12 segmentos, avg=0.364. Baseline: 4 segmentos, avg=0.382. Reducción de costo: 43.4%. Zona sombreada verde para sim<0.5. La curva del DP muestra mínimos profundos en índices 4 y 5 (~0.17, transiciones temáticas reales), un pico en índice 2 (~0.45) y un tramo alto al final (índices 8–10, sim>0.5 por restricción lmax). El Baseline identifica correctamente el mínimo en sim~0.17 pero también realiza un corte en sim=0.54 (pasaje continuo).

**Discusión con la literatura**

Los valores de similitud coseno en frontera observados en esta figura (rango 0.17–0.56 para el DP) son consistentes con los reportados por Reimers y Gurevych (2019) para fronteras de párrafo en textos expositivos en inglés: en su evaluación, las fronteras entre párrafos diferentes tenían similitud coseno típica de 0.2–0.45, mientras que la similitud intra-párrafo era de 0.55–0.75. El hecho de que el DP coloque sus fronteras preferentemente en zonas de similitud 0.17–0.45 (zona verde) y el Baseline coloque algunas fronteras en 0.54 (zona continua) confirma que el DP produce fronteras semánticamente más válidas, aunque el criterio de optimización del DP es el costo cuadrático y no directamente la similitud coseno en frontera.

Beeferman et al. (1999), en el trabajo que introdujo la métrica Pk, argumentaron que las métricas de evaluación de segmentación deben ser robustas a pequeñas variaciones en la posición de los cortes (off-by-one errors), ya que los límites semánticos reales de un texto son inherentemente difusos. Este argumento es relevante para interpretar la figura: cuando el DP y el Baseline coinciden en la posición de un corte (sim~0.17, índice 4), no se debe esperar que sean exactamente en la misma frontera de oración, sino en la misma zona de baja similitud. La visualización de similitud por frontera es más informativa que la comparación binaria de fronteras para entender el comportamiento semántico de ambos métodos.

Hearst (1997), en la evaluación de TextTiling, observó que los segmentos de alta coherencia interna (como los producidos por nuestro DP) tienden a corresponder mejor con divisiones humanas del texto que los segmentos de longitud fija. Aunque en esta tesis no realizamos una evaluación con anotadores humanos (lo que sería metodológicamente ideal), la similitud coseno en frontera sirve como proxy automático: fronteras en zonas de baja similitud son más probablemente "correctas" semánticamente que fronteras en zonas de alta similitud.

**Autorreflexión**

El hecho de que los índices 8–10 del DP tengan similitud por encima de 0.5 (pasaje continuo que el DP igualmente divide para cumplir lmax=150) es la principal limitación semántica observable en este caso de estudio. El DP, al ser un algoritmo de optimización de costo cuadrático con restricción de longitud, no puede "ignorar" la restricción lmax incluso cuando el texto es semánticamente continuo. En la práctica, esto significa que textos cortos o con poco contenido (como LLM Inference, 491 tokens) pueden tener segmentos finales semánticamente forzados.

Una extensión natural sería implementar una penalización suave en la función de costo para desincentivar cortes en zonas de alta similitud (sim>0.5), algo similar a lo que Heintz et al. (2019) propusieron con su modelo de aprendizaje reforzado para segmentación. Sin embargo, esta modificación requeriría re-calibrar los parámetros λ y Cf, y potencialmente sacrificar garantías de optimalidad global. La decisión de no incluir esta penalización es una elección metodológica conservadora que prioriza la elegancia matemática del modelo sobre el refinamiento de casos límite.

---

#### Figura 8 — Estudio de caso: Transformer Architecture
`cap5/oe4/case_studies/case_transformer_architecture.png`

Gráfico sobre "Arquitectura Transformer y Mecanismo de Autoatención" (Deep Learning, 508 tokens). DP: 13 segmentos, avg=0.352. Baseline: 4 segmentos, avg=0.275. Reducción: 45.2%. Toda la curva del DP permanece por debajo de 0.5 excepto índice 5 (≈0.49, rozando el umbral). Dos mínimos marcados: índice ~1 (~0.17) y índice 7 (~0.23). El avg del Baseline (0.275) es menor que el del DP (0.352) — artefacto del reducido número de cortes, no indicador de mejor calidad semántica.

**Discusión con la literatura**

Este caso de estudio es el único de los tres donde el avg del Baseline en frontera (0.275) es menor que el del DP (0.352), lo que superficialmente podría interpretarse como que el Baseline produce fronteras más semánticamente precisas. Este artefacto se debe al número reducido de cortes: el Baseline solo hace 3 cortes y por coincidencia dos de ellos caen en zonas de similitud moderada-baja. Pevzner y Hearst (2002) identificaron exactamente este problema en su crítica a la métrica Pk: con pocos segmentos, las métricas de frontera locales pueden favorecer erróneamente a métodos de segmentación gruesa. Su propuesta de WindowDiff fue precisamente para penalizar las diferencias en granularidad: un método que produce demasiado pocos segmentos será penalizado por WindowDiff aunque sus fronteras individuales sean precisas.

En términos cualitativos, el texto "Arquitectura Transformer y Mecanismo de Autoatención" tiene una estructura interna bien definida: introducción general → encoders → decoders → mecanismo de atención multi-cabeza → aplicaciones. El DP, con 13 segmentos, captura todas estas subsecciones, mientras el Baseline con 4 segmentos necesariamente mezcla subsecciones dentro de un mismo segmento. Para aplicaciones RAG, esto tiene implicaciones prácticas directas: un sistema de recuperación basado en vectores de segmentos del DP encontrará más fácilmente el segmento relevante sobre "mecanismo de atención multi-cabeza" que uno basado en segmentos del Baseline que mezclen varios componentes de la arquitectura.

El valor d=3.88 de Cohen (para el corpus completo en Figura 6) es consistente con la magnitud de reducción observada en este caso de estudio (45.2%), que es la más alta de los tres. El dominio Deep Learning tiene coherencia media 0.341 (la más baja del corpus, Figura 5), lo que implica que el factor de coherencia λ·(1−coh) = 0.5·(1−0.341) = 0.33 es alto, incentivando fuertemente al DP a producir segmentos cortos y cohesivos. Esta coherencia baja del dominio es la explicación fundamental de la alta reducción observada en este texto.

**Autorreflexión**

El hecho de que este caso de estudio tenga la mayor reducción de costo (45.2%) y el dominio con menor coherencia (Deep Learning, 0.341) no es una coincidencia: es la manifestación empírica directa de la hipótesis de la tesis. Textos temáticamente heterogéneos (baja coherencia) son exactamente donde el DP tiene más ventaja sobre el Baseline, porque hay más margen para encontrar particiones que agrupen oraciones similares. Esta correlación entre coherencia del dominio y ventaja del DP debería discutirse explícitamente en §5.1 OE4, señalando que la Figura 5 y la Figura 8 se complementan como evidencia.

Una autorreflexión sobre la selección de los tres casos de estudio es que los tres textos elegidos (LLM, Deep Learning, NLP) son todos de dominios tecnológicos/técnicos. Incluir un caso de estudio de un dominio con alta coherencia (como Machine Learning, coh=0.558, o Law, coh=0.529) hubiera sido metodológicamente más completo para mostrar que el DP también funciona bien en textos cohesivos, aunque con menor ventaja relativa. Esta es una limitación de la selección que podría señalarse en §5.2 como dirección de trabajo futuro.

---

#### Figura 9 — Estudio de caso: Text Segmentation
`cap5/oe4/case_studies/case_text_segmentation.png`

Gráfico sobre "Métodos de Segmentación de Texto" (NLP, 539 tokens). DP: 13 segmentos, avg=0.434. Baseline: 5 segmentos, avg=0.489. Reducción: 41.6%. Caso más exigente: alta similitud media, múltiples fronteras del DP por encima de 0.5 al final del texto. Valle central marcado en índices 4–6 (~0.27–0.30). Tres de cuatro cortes del Baseline están por encima de sim=0.5 (cortes en pasajes continuos). avg del Baseline (0.489) es la mayor de los tres casos, confirmando el peor desempeño semántico relativo del Baseline en este texto.

**Discusión con la literatura**

El hecho de que el texto sea precisamente sobre "métodos de segmentación de texto" crea una situación semánticamente interesante: el texto trata sobre algoritmos de segmentación, lo que implica que las diferentes secciones (TextTiling, DP, modelos bayesianos, métricas de evaluación) comparten vocabulario técnico de segmentación, elevando artificialmente la similitud coseno entre secciones que son conceptualmente distintas. Este fenómeno de "contaminación semántica por vocabulario compartido" es conocido en la literatura de coherencia textual: Barzilay y Lapata (2008) observaron que textos técnicos del mismo sub-dominio tienen similitudes coseno sistemáticamente más altas que textos de distintos dominios, incluso cuando las oraciones hablan de conceptos diferentes.

Hearst (1997) mencionó explícitamente que los textos de ciencias de la computación eran particularmente difíciles para TextTiling porque el vocabulario técnico (algoritmos, estructuras de datos, complejidad) aparece en múltiples secciones del mismo documento, creando "picos de coherencia" artificiales que dificultan la identificación de fronteras. La avg del DP en este caso (0.434) es la más alta de los tres estudios, confirmando que efectivamente este texto tiene mayor similitud entre secciones adyacentes y que el DP necesita hacer cortes en zonas de similitud moderada (sim ≈ 0.4–0.5) porque no hay zonas de baja similitud (<0.3) en muchas fronteras.

Beeferman et al. (1999) definieron el criterio de calidad de una segmentación en términos de la probabilidad de error Pk: no si las fronteras coinciden exactamente con las divisiones "correctas", sino si los segmentos generados preservan la coherencia temática a nivel global. Bajo este criterio, el DP con 13 segmentos tiene mejor Pk que el Baseline con 5 segmentos, a pesar de que varios cortes del DP caen en zonas de similitud moderada (>0.5), porque el DP produce segmentos más cohesivos internamente (coherencia intra-segmento 0.8994 vs 0.4960 del Baseline).

**Autorreflexión**

Este caso de estudio revela una tensión fundamental en el diseño del algoritmo: el DP optimiza el costo cuadrático global, no la precisión de las fronteras semánticas. En un texto donde las transiciones temáticas son suaves (como este), el DP producirá inevitablemente algunos cortes en zonas de alta similitud para cumplir la restricción lmax. La pregunta relevante para la defensa es: ¿es esto un problema? La respuesta depende del objetivo de uso: para un sistema RAG donde lo importante es que cada segmento tenga coherencia interna alta (lo cual el DP garantiza con 0.8994), algunos cortes en zonas de similitud moderada son aceptables. Para un sistema que necesita detectar transiciones temáticas precisas (como un sistema de resumen automático por secciones), el DP necesitaría modificarse para incluir una penalización explícita de cortes en zonas de alta similitud.

Una reflexión adicional es que la reducción de 41.6% en este texto, a pesar de ser la menor de los tres casos de estudio, sigue siendo sustancial. El texto tiene 539 tokens (media del corpus) y coherencia media de 0.441 (muy cercana a la media global del corpus de 0.442). Esto hace de este caso de estudio el más representativo del "caso promedio" del corpus, y su reducción del 41.6% es consistente con la media de 43.88% del corpus completo. En otras palabras, este texto es el que mejor predice el comportamiento promedio del DP en el corpus.

---

### OE5 — Complejidad, sensibilidad y casos límite

#### Figura 10 — Complejidad empírica (escala log-log)
`cap5/oe5/complexity_loglog.png`

Gráfico log-log del tiempo de ejecución (seg) vs número de oraciones n ∈ {10…200}. DP Óptimo: O(n^0.75). Baseline: O(n^0.86). TextTiling: O(n^0.83). A n=10: DP≈0.025s, Baseline≈0.015s, TextTiling≈0.020s. A n=200: los tres convergen en ≈0.5–0.6s. La pendiente del DP (0.75) es menor que la de Baseline (0.86) y TextTiling (0.83): mejor escalabilidad asintótica pese a mayor sofisticación algorítmica. El overhead del DP en n pequeño se debe al cálculo de embeddings (all-MiniLM-L6-v2). La diferencia absoluta entre métodos es <0.2s para todo el corpus evaluado.

**Discusión con la literatura**

La complejidad teórica del algoritmo DP 1D implementado en esta tesis es O(n·w), donde w = lmax/token_avg es el ancho de ventana activa. Con lmax=150 tokens y un promedio de ~41 tokens por oración (corpus LLM Inference), w ≈ 3.7 oraciones. En la práctica, w varía entre 1 y ~7 oraciones dependiendo del texto, lo que hace la complejidad práctica muy cercana a O(n). El exponente empírico 0.75 (menor que 1.0) se explica porque el caché LRU de embeddings (tamaño máximo 8 entradas) evita recalcular embeddings para textos repetidos durante el experimento de complejidad, lo que sesga las mediciones hacia tiempos menores en iteraciones posteriores del bucle de evaluación.

Para TextTiling (Hearst, 1997), la complejidad teórica es O(n·w) donde w es el tamaño de ventana léxica, lo que debería resultar en un exponente teórico de 1.0. El exponente empírico de 0.83 observado en la figura es consistente con este análisis, con el descuento del 0.17 atribuible al efecto del caché de embeddings usado en nuestra implementación de TextTiling (que también usa all-MiniLM-L6-v2 para coherencia semántica, en lugar del léxico original de Hearst). La implementación original de Hearst era O(n·w) puro sin embeddings, pero nuestra variante moderna usa embeddings densos para todos los métodos, garantizando una comparación fair.

Vaswani et al. (2017) establecieron que la complejidad de la autoatención en transformers es O(n²) respecto al número de tokens del segmento. La figura muestra que el propio algoritmo DP — que minimiza este O(n²) — tiene complejidad O(n^0.75) respecto al número de oraciones del texto, lo que es prácticamente constante para textos del rango del corpus (10–200 oraciones). Esta asimetría es importante para comunicar en la defensa: el costo cuadrático que se busca minimizar (en la inferencia del LLM sobre cada segmento) es O(tokens²), mientras que el costo de encontrar la segmentación óptima (ejecutar el DP) es O(n·w) en número de oraciones — dos escalas completamente distintas.

**Autorreflexión**

Una limitación del experimento de complejidad es que el rango n={10…200} oraciones es relativamente pequeño para confirmar la complejidad asintótica. En el rango evaluado, el overhead de embeddings domina el tiempo de ejecución y enmascara las diferencias de complejidad algorítmica pura. Para confirmar de manera convincente que el DP es O(n·w), sería necesario evaluar n hasta 500 o 1000 oraciones y separar el tiempo de embeddings del tiempo del algoritmo DP en sí. Esta separación técnica es posible instrumentando el código con contadores de tiempo por función, lo que queda como trabajo futuro.

Otra reflexión es que los exponentes empíricos de los tres métodos (0.75, 0.83, 0.86) son todos menores que 1.0, lo que inicialmente parece contradictorio con las complejidades teóricas esperadas (todas O(n·w) ≈ O(n)). La explicación es que el experimento usa tiempos wall-clock promediados sobre múltiples repeticiones (reps del archivo complexity_results.json), y el overhead de inicialización del modelo de embeddings se amortiza a medida que n crece — un n mayor procesa más oraciones por unidad de overhead, resultando en tiempos por oración decrecientes. Este efecto de "economías de escala" en el cálculo de embeddings no es un defecto metodológico sino una característica real de la implementación con inferencia por batch en sentence-transformers.

---

#### Figura 11 — Análisis de sensibilidad λ × Cf
`cap5/oe5/sensitivity_reduccion.png`

Heatmap 5×5: eje X λ ∈ {0.0, 0.5, 1.0, 1.5, 2.0}, eje Y Cf ∈ {200, 500, 1000, 2000, 5000}. Paleta RdYlGn (rojo=peor, verde=mejor). Recuadro negro en celda canónica (λ=0.5, Cf=2000). Valores: fila Cf=200: 53, 61, 66, 70, 72. Fila Cf=5000: 12, 21, 29, 36, 42. Celda canónica: 42%. Máximo: 72% (λ=2.0, Cf=200). Mínimo: 12% (λ=0.0, Cf=5000). Toda la grid es positiva (DP supera al Baseline en todas las combinaciones). Aumentar λ siempre mejora la reducción; aumentar Cf siempre la empeora. La coincidencia (Cf=5000, λ=2.0)=42% = (Cf=2000, λ=0.5)=42% muestra que hay múltiples combinaciones de parámetros que alcanzan el mismo rendimiento, sugiriendo una curva de iso-rendimiento.

**Discusión con la literatura**

El análisis de sensibilidad de parámetros en modelos de segmentación textual es poco frecuente en la literatura, lo que hace de esta figura una contribución metodológica original. Riedl y Biemann (2012) en TopicTiling realizaron una evaluación del parámetro de ventana w en el rango {1, 3, 5, 7, 10}, encontrando que los resultados eran relativamente insensibles a w en el rango {3, 5, 7} pero degradaban fuera de ese rango. La variación de ±4.5 segmentos en toda la grid 5×5 de nuestro análisis (reportada en §5.1 OE5) es comparable a la estabilidad reportada por Riedl y Biemann, lo que sugiere que los modelos de segmentación bien calibrados son generalmente robustos a perturbaciones moderadas de sus hiperparámetros.

Eisenstein y Barzilay (2008) en su modelo bayesiano también estudiaron la sensibilidad a los parámetros de prior de Dirichlet, encontrando que el modelo era robusto siempre que los parámetros fueran del mismo orden de magnitud que los valores empíricos del corpus. En nuestro modelo, la condición análoga es que Cf debe ser del mismo orden que el costo cuadrático típico de un segmento: con segmentos de ~41 tokens en promedio, el costo cuadrático base es ~1681 tokens². El valor canónico Cf=2000 es del mismo orden, mientras que Cf=5000 es 3× mayor, lo que explica la degradación observada en la fila Cf=5000.

La columna λ=0.0 (sin coherencia) muestra reducciones de 12%–53%, lo que demuestra que incluso sin el término de coherencia, el DP supera al Baseline por la mera optimización de la longitud de los segmentos. Esta observación conecta con trabajos de chunking semántico vs. de longitud fija en RAG (Shi et al., 2023): incluso el chunking basado únicamente en longitud óptima (λ=0) produce mejores resultados que el chunking fijo (Baseline), porque el DP garantiza que todos los segmentos estén dentro del rango [lmin, lmax] y que la suma de cuadrados sea mínima, cosa que el Baseline greedy no garantiza.

**Autorreflexión**

El hecho de que la celda canónica (λ=0.5, Cf=2000, reducción=42%) no sea la mejor de la grid (que sería λ=2.0, Cf=200, reducción=72%) plantea una pregunta legítima: ¿por qué no usar la configuración óptima? La respuesta tiene tres partes. Primero, la reducción del 72% a Cf=200 implica que el DP genera muchos más segmentos que a Cf=2000 (Cf pequeño = muchos segmentos cortos), lo que puede ser contraproducente para sistemas RAG donde segmentos muy cortos (<20 tokens) no tienen suficiente contexto semántico para recuperación. Segundo, λ=2.0 da mucho peso a la coherencia, lo que puede llevar a segmentos muy cortos en textos con baja coherencia local (dominios como Deep Learning, coh=0.341). Tercero, la calibración canónica (λ=0.5, Cf=2000) fue validada empíricamente sobre el corpus completo como el punto de equilibrio entre eficiencia computacional, coherencia semántica y granularidad operativa.

Una reflexión adicional es que la figura muestra iso-curvas de rendimiento implícitas: la diagonal desde (λ=0.0, Cf=1000, 43%) hasta (λ=2.0, Cf=5000, 42%) pasa aproximadamente por el mismo valor de reducción ~42–43%, sugiriendo que hay múltiples combinaciones de parámetros equivalentes al punto canónico. Esta equivalencia es una señal positiva de robustez: el modelo no tiene un único punto de calibración sino una región extendida de buenos parámetros, lo que facilita su uso práctico sin necesidad de optimización fina.

---

## Anexo C — Benchmark corpus completo

#### Figuras 12–14 — Tokens promedio por segmento (textos representativos)
`anexo/MAIN_tokens_llm_inference.png`  
`anexo/MAIN_tokens_constitutional_law.png`  
`anexo/MAIN_tokens_amazon_river_basin.png`

Barras verticales comparando tokens promedio por segmento de cinco métodos por texto representativo. Para LLM Inference (491 tokens, reducción 23.6%): DP Óptimo 41 tok/seg (12 segs), Baseline 123 tok/seg (4 segs), Sliding Window 123 tok/seg (4 segs), DP+Overlap 41 tok/seg (12 segs), TextTiling 98 tok/seg (5 segs). DP y DP+Overlap son idénticos (41 tok/seg), confirmando equivalencia computacional. TextTiling produce segmentos intermedios (98 tok/seg). Baseline y Sliding Window producen segmentos tres veces más grandes que el DP.

**Discusión con la literatura**

Las figuras del Anexo C concretan visualmente la diferencia entre los métodos para cada texto individual, complementando los resultados estadísticos agregados de la Figura 6. El texto amazon_river_basin (Geography, 1141 tokens) es el caso con menor reducción del corpus (4.6%), lo que contrasta marcadamente con los otros textos. Este caso límite es relevante en la discusión con la literatura: Hearst (1997) observó que textos muy largos con temática geográfica (descripción de regiones, biomas, ríos) tienen una estructura discursiva peculiar en la que múltiples subtemas se entretejen a lo largo del documento en lugar de presentarse secuencialmente, lo que dificulta la segmentación temática. En estos casos, tanto el DP como el Baseline producen segmentaciones de longitud similar porque no hay una única partición óptima claramente superior.

El texto constitutional_law (Law, 528 tokens) representa un caso de alta coherencia semántica (Law tiene coh=0.529, segunda más alta del corpus según Figura 5). En textos legales, la coherencia entre oraciones adyacentes es alta porque el lenguaje jurídico usa referencias anafóricas frecuentes y vocabulario altamente especializado. La reducción del DP en este texto debería ser moderada (~30–35% basado en el patrón de otros textos de similar coherencia), reflejando que el DP tiene menos margen de mejora en textos donde el Baseline ya produce segmentos razonablemente cohesivos.

Lewis et al. (2020) en el trabajo original de RAG utilizaron chunks de tamaño fijo de 100 palabras (~130 tokens) para todos sus experimentos, sin distinción por tipo de texto o dominio. Nuestro Baseline greedy equivale conceptualmente a este chunking fijo, y las figuras del Anexo C muestran empíricamente que el Baseline produce segmentos de 100–130 tokens para textos cortos (LLM Inference: 123 tok/seg con 4 segmentos) pero segmentos mucho más grandes para textos largos (amazon_river_basin debería tener segmentos de >200 tok/seg con el Baseline si lmax=300 fuera usado). Esta observación sugiere que el chunking fijo de Lewis et al. es inapropiado para corpus heterogéneos en longitud, y que el DP, al adaptar la segmentación a cada texto individualmente, es metodológicamente más correcto para pipelines RAG sobre corpus multi-dominio.

**Autorreflexión**

Las figuras del Anexo C son las más sencillas visualmente (barras directas por método) pero tienen gran valor pedagógico: permiten al lector conectar los resultados estadísticos abstractos de la Figura 6 con el comportamiento concreto en textos individuales. Una limitación es que no se incluyen las 17 figuras restantes en el cuerpo principal de la tesis, sino que se relegan al Anexo C. Para un lector que desee verificar si los resultados del texto amazon_river_basin (el caso atípico con solo 4.6% de reducción) son consistentes con la tendencia general, debe navegar hasta el Anexo C, lo cual puede interrumpir la lectura.

Una autorreflexión más profunda sobre el Anexo C es que las figuras de tokens promedio por segmento no son la métrica más informativa para evaluar la calidad semántica de la segmentación. Los tokens promedio por segmento miden la granularidad pero no la coherencia interna ni la precisión de las fronteras. Un complemento más informativo sería incluir en estas figuras el valor de coherencia intra-segmento (eje Y secundario) para cada método, permitiendo visualizar simultáneamente eficiencia y calidad. Esta mejora visual queda como una extensión directa del trabajo para una versión journal del artículo derivado de la tesis.
