# SpecRAMA: Propuesta de Paper y Hoja de Ruta de Publicación

> **Documento de Trabajo y Blueprint Académico**  
> **Título Oficial:** *SpecRAMA: Parameter-Efficient Fine-Tuning of Low-Bit Quantized Large Language Models via 2D Bipartite TSP Permutations and Multiscale Spectral Cores*  
> **Autor:** Mario Raúl Carbonell Martínez  
> **Venues Objetivo:** EMNLP / COLM (Conference on Language Modeling) / ICLR / NeurIPS Main Track  
> **Repositorio:** `https://github.com/mcarbonell/spec-rama`  
> **Proyecto Hermano (Pre-entrenamiento Inductivo):** `dirichlet-regularization`

---

## 1. Resumen Ejecutivo y Posicionamiento Científico

### 1.1 La Tesis Central
Los modelos de lenguaje de gran escala (LLMs) pre-entrenados almacenan sus representaciones en matrices de pesos densas sin una correlación espacial obvia entre filas y columnas contiguas. Debido a esta alta entropía armónica en coordenadas nativas, cualquier intento de comprimir o adaptar estos modelos en el dominio frecuencial sufre una dispersión masiva de energía hacia las altas frecuencias.

**SpecRAMA** resuelve este dilema mediante una formulación combinatoria dual:
1. **Reordenación de Coordenadas vía 2D Bipartite TSP:** Encuentra permutaciones ortogonales discretas $(\pi_{\text{row}}, \pi_{\text{col}})$ que minimizan la Variación Total ($L_1$), forzando a los pesos pre-entrenados a comportarse como **variedades 2D continuas y suaves**.
2. **Adaptación Espectral Multiescala (DWT / DCT-2D):** Con la energía concentrada (>92% en la sub-banda de baja frecuencia), se entrena un núcleo espectral minúsculo (Wavelet o DCT) con modulación multiplicativa-aditiva (RAMA) y **Escalado de Energía de Parseval**.

### 1.2 Resultados Clave ya Demostrados
* **$6.0\times$ más compacto que LoRA:** En GPT-2 Small (48 proyecciones lineales), SpecRAMA Wavelet ($32\times32$) utiliza solo **384 KB de adaptador** frente a los **2.36 MB de LoRA ($r=4$)**, logrando **37.96 PPL** en bases NF4.
* **Cierre de Brecha de Cuantización:** Queda a menos de **+0.078 bits/token** (+0.078 bpt) del límite superior adaptado en FP32.
* **Régimen Sub-Kilobyte (0.88 KB):** Con `SharedSpecRAMAModel` (un solo núcleo maestro global compartido entre las 48 capas con ganancias escalares por capa), adapta el modelo completo con solo **224 parámetros (896 bytes)**.
* **Fusión de Latencia Cero:** En tiempo de inferencia, el núcleo adaptado se transforma mediante IDWT/IDCT, se reordena con las permutaciones inversas y se suma in-place a la matriz base congelada.

---

## 2. Borrador del Abstract Académico

```latex
\begin{abstract}
Parameter-efficient fine-tuning (PEFT) has emerged as the de facto standard for adapting 
large language models under strict hardware constraints, particularly when combined with 
low-bit post-training quantization (e.g., QLoRA). However, classical spatial low-rank 
factorizations (such as LoRA) still impose a substantial parameter footprint (typically 
megabytes per model) and require careful rank tuning. Spectral adaptation methods (such as 
FourierFT) attempt to mitigate this by parameterizing updates in the frequency domain, but 
are constrained to 1D transforms that fail to exploit the natural 2D covariance of linear 
projections. In this work, we introduce \textbf{SpecRAMA}, a parameter-efficient fine-tuning 
framework that bridges discrete combinatorial optimization with multiscale 2D spectral core 
adaptation. SpecRAMA demonstrates that unstructured, pre-trained weight matrices can be 
transformed into smooth 2D manifolds via alternating 2D Bipartite Traveling Salesperson 
Problem (TSP) channel permutations that minimize Total Variation ($\text{TV}$). This re-indexing 
empirically concentrates over $92\%$ of matrix energy into low-frequency Discrete Wavelet 
(DWT) and Cosine (DCT) subbands. Building upon this smooth coordinate frame, SpecRAMA learns 
compact multiscale spectral cores modulated by additive-multiplicative scaling and stabilized 
via a dimension-agnostic Parseval Energy Scaling formulation. 
On GPT-2 Small across 48 linear projections under 4-bit NormalFloat (NF4) quantization, 
SpecRAMA achieves a validation perplexity of 37.96 (5.246 bits/token), falling within 
$+0.078$ bpt of the domain-adapted FP32 upper bound while utilizing a $6.0\times$ smaller 
adapter footprint (384.0 KB vs. 2.36 MB for LoRA $r=4$). Furthermore, with cross-layer core 
sharing, SpecRAMA fine-tunes the entire 124M parameter architecture using only 224 trainable 
parameters (896 bytes). Finally, we show that 2D TSP coordinate permutations resolve the 
quantization collapse of extreme sub-2b spectral quantization, linking post-hoc channel 
permutations with continuous topological pre-training regularization.
\end{abstract}
```

---

## 3. Simetría Teórica con `dirichlet-regularization`

Ambos trabajos constituyen un programa de investigación unificado sobre **variedades espectrales en redes neuronales**:

| Dimensión | `dirichlet-regularization` (Paper 1) | `spec-rama` (Paper 2) |
| :--- | :--- | :--- |
| **Fase del Ciclo de Vida** | Pre-entrenamiento desde cero | Adaptación (PEFT) de modelos congelados |
| **Formulación Matemática** | Continua: Energía de Dirichlet $L_2$ ($\nabla W$) | Discreta: Variación Total $L_1$ ($\text{TV}$) + TSP Bipartito |
| **Mecanismo** | Flujo de calor de Laplace-Beltrami | Algoritmo voraz alternado en grafos de distancia L2 |
| **Objetivo Principal** | Compresión extrema a sub-1.0 bpp en silicio Edge | Adaptación PEFT con tamaño sub-kilobyte en bases NF4 |
| **Overhead de Coordenadas** | Cero (los pesos nacen ordenados) | Permutación determinista invertible $\hat{W} = P_r^\top \hat{W}_\pi P_c$ |
| **Integración Hardware** | Micro-kernel C con DMA en $<1$ MB SRAM | Fusión in-place de latencia cero en DRAM/VRAM |

### Cómo se citan mutuamente:
* **En `dirichlet-regularization`:**
  > *"For pre-trained foundational models where re-training with continuous Dirichlet regularization is prohibitive, discrete coordinate re-indexing via 2D Bipartite TSP permutations (as in SpecRAMA) provides a natural post-hoc counterpart to concentrate energy prior to spectral compression."*
* **En `spec-rama`:**
  > *"While Dirichlet Weight Regularization enforces continuous harmonic smoothness during pre-training, SpecRAMA establishes that discrete bipartite channel permutations can uncover latent spectral concentration on frozen, unstructured pre-trained weights without altering the underlying base parameters."*

---

## 4. Estructura y Secciones Detalladas del Paper

### Section 1: Introduction
* La explosión de tamaño de los LLMs y el auge de PEFT (LoRA, QLoRA).
* El "cuello de botella de rango": por qué LoRA no puede bajar de cientos de kilobytes ($r \ge 1$).
* La oportunidad espectral: 1D Fourier (FourierFT) pierde la estructura 2D de las matrices.
* La tesis de SpecRAMA: Reordenar coordenadas en 2D transforma cualquier matriz en un mapa suave donde núcleos espectrales minúsculos tienen enorme poder expresivo.

### Section 2: Related Work
* **PEFT Clásico:** LoRA, QLoRA, DoRA, AdaLoRA.
* **PEFT Espectral y Sub-Kilobyte:** FourierFT (ICML 2024), VeRA (ICLR 2024), NOLA.
* **Transformaciones Ortogonales en LLMs:** QuIP#, QuaRot, SpinQuant (rotaciones de Hadamard para eliminar outliers) vs. SpecRAMA (permutaciones TSP para concentrar energía armónica).
* **Topología Continua en Pesos:** Conexión con regularización armónica de Dirichlet.

### Section 3: Methodology
* **3.1 2D Bipartite TSP Permutation Algorithm:**
  * Formulación como problema del viajante de comercio métrico sobre distancias $L_2$.
  * Alternancia fila/columna y criterio de parada adaptativo por convergencia de Variación Total:
    $$\text{TV}(W) = \sum |W_{i+1, j} - W_{i,j}| + \sum |W_{i, j+1} - W_{i,j}|$$
* **3.2 Multiscale Wavelet & DCT Core Adaptation:**
  * Descomposición DWT-2D (Haar, Daubechies) y concentración en el sub-banda LL.
  * Formulación de RAMA: modulación multiplicativa $M$ y aditiva $A$.
* **3.3 Parseval Energy Scaling:**
  * Demostración analítica de por qué el escalado estándar $\alpha / r$ falla en núcleos 2D.
  * Derivación de la conservación de energía de Parseval:
    $$\text{Scale} = \frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}}} \cdot \frac{1}{\text{std}(W_0)}$$
* **3.4 Cross-Layer Core Sharing (`SharedSpecRAMAModel`):**
  * Compartición de un único tensor $8\times8$ para las 48 capas con ganancias escalares $\gamma_l, \beta_l$.
* **3.5 Zero-Latency In-Place Merging:**
  * Reconstrucción matemática exacta en FP32 sin overhead de inferencia.

### Section 4: Experimental Setup & Rigor (Blindaje tras la Auditoría)
* **Arquitectura:** GPT-2 Small (124M), 48 módulos lineales (`c_attn`, `c_proj`, `c_fc`).
* **Congelación Estricta:** Verificación de gradientes nulos en `wte`, `wpe` y `LayerNorm`.
* **Corpus de Evaluación:** WikiText-2 (100 bloques estándar de 256 tokens + evaluación en test set completo).
* **Baselines Controlados:** Native FP32, FP32 + SpecRAMA, NF4 Base, NF4 + SpecRAMA, NF4 + LoRA ($r=4$ auditado con embeddings congelados), NF3/4 Asimétrico.

### Section 5: Experimental Results
* **Tabla Principal de Resultados** (EXP-11 / EXP-12 corregidos).
* **Análisis de Sensibilidad a Learning Rate:** La estabilidad Parseval de SpecRAMA frente a LoRA.
* **Compresión Extrema y Régimen Sub-Kilobyte:** Resultados de `SharedSpecRAMAModel` (896 bytes).
* **Experimento Cruzado con Base-3 TritQ:** Demostración de que la permutación TSP previene el colapso de cuantización espectral (reducción de PPL de 3.752 a 373 en régimen conservador).

### Section 6: Discussion & Threats to Validity
* Aislamiento entre adaptación de dominio y recuperación de daño de cuantización.
* Análisis de complejidad temporal del TSP voraz ($\mathcal{O}(N^2)$ en GPU, <30 segundos para todo GPT-2).
* Por qué persistir las permutaciones no consume espacio si se generan a partir de $W_0$.

### Section 7: Future Work & Conclusion
* Escalado a modelos 8B+ (LLaMA-3, Qwen-2.5) en tareas de razonamiento (MMLU, GSM8K).
* Kernels fusionados Triton/CUDA para DWT/DCT + GEMM.
* Permutaciones aprendibles continuas (Monarch / factorizaciones Butterfly).

---

## 5. Tareas Técnicas Inmediatas en el Repo `spec-rama`

Para cuando abras el repositorio de `spec-rama`, esta es la lista de acciones listas para implementar:

1. **Fix en `spec_rama/permutation.py` (Línea 129):**
   * Corregir la llamada a `compute_joint_bipartite_2d_permutation` cambiando `num_iters=2` por `max_iters=2`.
2. **Re-ejecución del Control LoRA Congelado:**
   * Asegurar que en `exp11` y `exp12` se ejecute `for p in model.parameters(): p.requires_grad = False` antes de inyectar LoRA, garantizando que los embeddings `wte` estén estrictamente congelados.
3. **Optimización de Buffers en `state_dict`:**
   * Modificar `SpecRAMALinear` para que los buffers de permutación no se guarden en el archivo `.pt` del adaptador (ya que se recalculan deterministamente en 0.5s por capa a partir de la base congelada), reduciendo el archivo en disco de 2.75 MB a **384 KB reales**.
4. **Creación de la plantilla LaTeX del paper (`paper/paper-draft.tex` en `spec-rama`):**
   * Con la misma estructura limpia y compilable de `dirichlet-regularization`.
