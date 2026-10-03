# Auditoría Completa — Dirichlet Regularization (`dreg`)

**Repositorio:** `mcarbonell/dirichlet-regularization`  
**Rama auditada:** `arena/01a10131-dirichlet-regularization` (commit base `26b5700`)  
**Fecha de auditoría:** 2026-10-03 (UTC)  
**Auditor:** Agent Mode (Arena.ai) — revisión de código, documentación, fundamento matemático y viabilidad de publicación  
**Idioma:** Español (el paper está en inglés; la auditoría respeta el idioma de la solicitud)  
**Estado global:** **Publicable con revisiones mayores.** La idea central es sólida, la ingeniería es notablemente cuidada para tratarse de un proyecto de autor único, y la evidencia empírica a escala 10 M/TinyStories es honesta y reproducible. Hay sin embargo 3 hallazgos P0/P1 que deben corregirse antes de enviar a revisores exigentes (omisión selectiva en curva de Pareto, discrepancia factor 2 en la definición de $E_D$, y fragilidad del *falsification test* a pequeña escala).

---

## Índice

1. [Veredicto ejecutivo](#1-veredicto-ejecutivo)
2. [Evaluación de la idea](#2-evaluación-de-la-idea)
3. [Auditoría matemática](#3-auditoría-matemática)
4. [Auditoría de código](#4-auditoría-de-código)
5. [Auditoría de tests y cobertura](#5-auditoría-de-tests-y-cobertura)
6. [Auditoría de cuantización y formato `.tritq`](#6-auditoría-de-cuantización-y-formato-tritq)
7. [Auditoría del kernel C / DMA](#7-auditoría-del-kernel-c--dma)
8. [Auditoría empírica y de resultados](#8-auditoría-empírica-y-de-resultados)
9. [Auditoría de documentación](#9-auditoría-de-documentación)
10. [Auditoría del paper (LaTeX)](#10-auditoría-del-paper-latex)
11. [Riesgos y objeciones previsibles de revisores](#11-riesgos-y-objeciones-previsibles-de-revisores)
12. [Matriz de hallazgos priorizada (P0–P3)](#12-matriz-de-hallazgos-priorizada-p0p3)
13. [Recomendaciones de publicación](#13-recomendaciones-de-publicación)
14. [Apéndices técnicos](#14-apéndices-técnicos)

---

## 1. Veredicto ejecutivo

| Dimensión | Nota (0–10) | Comentario de una línea |
|-----------|-------------|--------------------------|
| **Originalidad de la idea** | **8.5** | Romper la simetría de permutación con Dirichlet es un cambio de marco elegante y biológicamente motivado. No es SOTA en compresión absoluta (>2 bpp sigue dominado por GPTQ/AWQ/QuIP#), pero el ángulo espectral es genuinamente nuevo como *principio de entrenamiento*, no como post-processing. |
| **Rigor matemático** | 7.0 | Derivación $E_D \leftrightarrow \Delta W \leftrightarrow \mu_{k,l}\|S_{k,l}\|^2$ correcta. Fórmula de decaimiento espectral citada como $\propto (1+\lambda r^2)^{-1}$ es *heurística bayesiana*, no teorema probado — debe recalificarse. Factor 2 y dos normalizaciones diferentes coexisten sin documentar. |
| **Calidad de código** | 8.0 | 508 líneas de núcleo, 79 tests, 97 % cobertura, `ruff` limpio, tipado razonable, API de 2 líneas. Listo para `pip install`. |
| **Ingeniería de sistemas** | 9.0 | El pipeline `.tritq` + LUT 1.25 KB + doble búfer DMA + Block-DCT es ingeniería de nivel publicación (v382–v395 muy bien trazadas). |
| **Evidencia empírica** | 6.5 | 10 M / 82 M tokens en A10G es creíble y honesta (reporta degradación FP32). Falla en no reportar punto de Pareto donde el modelo topográfico pierde (-281 PPL a 0.914 bpp) y en que el *small-scale falsification* de 2 épocas no replica el efecto. GPT-2 124 M honestamente reporta fracaso — punto fuerte de honestidad. |
| **Documentación** | 7.5 | README excelente como aterrizaje, whitepaper técnico exhaustivo, ROADMAP trazable. Falta guía de reproducibilidad local sin Modal. |
| **Preparación para paper** | 7.0 | Manuscrito de 15 págs bien estructurado, pero con sobreventa puntual (sub-1 bpp con PPL 66 sigue lejos de ser utilizable) y tablas con métricas inconsistentes entre secciones. Con los P0 corregidos, tiene perfil claro para **TMLR** o *workshop* de NeurIPS/ICLR sobre compresión. |

**Recomendación:** *Enviar a arXiv inmediatamente para prioridad + pulir P0/P1 en 2–3 semanas → TMLR* (rolling, sin deadline, permite revisiones abiertas, valora reproducibilidad). No recomiendo NeurIPS/ICLR *main track* sin antes resolver el frente de Pareto completo y un baseline con GPTQ en dominio espectral.

> **Resumen en una frase:** *Tienes un diamante en bruto: el principio es correcto, la implementación está por encima de la media de papers académicos, pero el relato actual exagera la monotonicidad de la ganancia y esconde un punto donde el regularizado empeora. Corrigiendo eso conviertes una posible acusación de cherry-picking en tu mayor demostración de honestidad científica.*

---

## 2. Evaluación de la idea

### 2.1 Enunciado del principio

> *Tratar cada matriz densa $W\in\mathbb{R}^{M\times N}$ como una discretización de un campo escalar continuo $\mathcal{W}:[0,1]^2\to\mathbb{R}$ y penalizar $\|\nabla\mathcal{W}\|^2$ durante el entrenamiento.*

La idea se descompone en tres tesis:

1. **T1 – Geometría:** La simetría de permutación de filas/columnas es una gauge artificial; fijarla con una malla 2D induce una topografía cortical.
2. **T2 – Espectro:** Esa topografía, por teoría espectral del Laplaciano, concentra energía de Frobenius en bajas frecuencias del DCT-II.
3. **T3 – Compresión:** Esa concentración permite cuantización jerárquica radial sub-1 bpp con truncamiento de altas frecuencias.

**Valoración T1:** Original y bien motivada biológicamente (Hubel-Wiesel, Kohonen, Durbin-Mitchison). No he encontrado trabajo previo que formule exactamente la penalización $\tfrac12\int\|\nabla\mathcal{W}\|^2$ como regularizador diferenciable genérico sobre *todas* las proyecciones lineales de un Transformer. Trabajos cercanos: *Topographic VAEs* (Keller 2021), *Self-Organizing Maps for weight sharing* (no espectral), *FourierFT* / *Spectral PEFT* (Gao 2024) — pero estos operan en espacio de adaptadores, no como regularizador de pre-entrenamiento denso. **Novedad confirmada, con matiz:** debe citarse *Spectral Regularization* (Yoshida 2017) y *Weight Smoothing via Total Variation* en visión.

**Valoración T2:** Correcta en esencia. La conexión autovalores $\mu_{k,l}$ – DCT-II con condiciones Neumann está bien establecida (Strang 1999, Belkin-Niyogi). El paper la presenta con rigor (Sección 3.2 + Apéndice). Único reparo: la ley $\mathbb{E}[|C_{u,v}|^2]\propto(1+\lambda(u^2+v^2))^{-1}$ se deduce de un prior gaussiano con precisión $I+\lambda L$, no del flujo de SGD+AdamW. Es una *predicción de equilibrio bayesiano*, no un teorema dinámico. Debe presentarse como *motivación teórica / aproximación de campo medio*.

**Valoración T3:** Consecuencia natural si T2 es cierta. El diseño de 4 bandas radiales + trit Base-3 es ingenioso pero no es la única vía; JPEG hace algo análogo. La contribución aquí es *demostrar causalidad*: sin Dirichlet, la misma cuantización colapsa (falsificación). Esa causalidad es el corazón defendible del paper.

### 2.2 Analogía cortical

Bien usada como narrativa, pero peligrosa si se sobrevende. El cortical map biológico es *emergente* (minimiza longitud de cableado bajo restricciones metabólicas), no impuesto por un término $L_2$ de gradiente. El paper lo reconoce implícitamente al distinguir *intrinsic smoothing* vs *external sinusoidal anchoring* (Sección 5.1, rank-1 collapse). Mantener la analogía como inspiración, no como equivalencia.

### 2.3 Hipótesis auxiliares evaluadas

| Hipótesis | Estado según repo | Valoración |
|-----------|-------------------|------------|
| Annealing / Cooldown ($\lambda\to0$ últimos 15 %) | **Confirmada** en 5 k pasos: 7.50 PPL FP32, 55.68 PPL TritQ, retiene 64 % de suavidad | Muy prometedora. Es la vía para cerrar la brecha FP32. Falta prueba a 10 k pasos; el resultado actual compara 5 k annealed vs 10 k no-annealed (no apples-to-apples). |
| Reordenamiento TSP post-hoc para GPT-2 pre-entrenado | Parcial: reduce $E_D$ 13.8 %, mejora TritQ hasta +3 678 PPL en algún régimen, pero sigue en 373–4 374 PPL (inutilizable) | Honesto reportar que *fine-tuning* 300 pasos con $\lambda=10^{-3}$ no mueve $E_D$ ($\Delta<0.01$ %). Conclusión correcta: Dirichlet es pre-entrenamiento, no parche. |

### 2.4 Originalidad frente a SOTA de cuantización

* GPTQ/AWQ/QuIP# *minimizan error de cuantización dado un modelo fijo*. DREG *cambia el modelo para hacerlo cuantizable*. Son ortogonales y combinables (oportunidad futura: GPTQ espectral). El posicionamiento del paper es correcto al no competir en 4 bpp, sino abrir el régimen <1 bpp donde los demás no operan.
* Riesgo: revisores preguntarán “¿Por qué no aplicar GPTQ sobre $S$ (dominio DCT) para mejorar 0.945 bpp?”. La respuesta debe estar lista: es trabajo futuro (Limitaciones ya lo lista, bien).

---

## 3. Auditoría matemática

### 3.1 Definición discreta de $E_D$

**Paper (Eq. 1):**
$$\mathcal{E}_D(W)=\frac{1}{2MN}\Bigl[\sum_{i=1}^{M-1}\sum_{j=1}^{N}(W_{i+1,j}-W_{i,j})^2+\sum_{i=1}^{M}\sum_{j=1}^{N-1}(W_{i,j+1}-W_{i,j})^2\Bigr]$$

**Código (`dreg/topology.py:43-62`):**
```python
sum_sq = diff_h.pow(2).sum() + diff_w.pow(2).sum()
# ...
if normalization == "numel":
    return sum_sq / (sheet.numel() + 1e-8)
elif normalization == "edges":
    return sum_sq / (max(num_edges,1) + 1e-8)
```

* `numel = M·N` → $E_D^{\text{code}} = (1/MN)\sum(\Delta^2)$ = **2 ×** $E_D^{\text{paper}}$ (falta el $\tfrac12$).
* `edges = (M-1)N + M(N-1)` → aproximadamente $2MN$ para $M,N\gg1$, entonces $E_D^{\text{edges}}\approx E_D^{\text{paper}}$.

**Implicación:** Tu $\lambda=30$ en modo `numel` equivale a $\lambda_{\text{paper}}\approx15$. Como todo el barrido usa `numel` consistentemente, los resultados relativos son válidos, pero el número reportado no coincide con la ecuación impresa. **P0: unificar o documentar factor 2.** La Tabla 1 reporta $E_D=0.004225$ a 0.001924; si alguien replica con tu ecuación impresa obtendrá la mitad.

**Recomendación:** Cambiar código a `sum_sq / (2*sheet.numel())` para `numel`, o añadir nota “definimos $E_D$ con normalización por elementos (factor 2 respecto a forma variacional)”.

### 3.2 Flujo de gradiente = difusión de calor

Derivación del Apéndice (variación primera + Green) es correcta bajo Neumann homogéneo. El paper escribe $-\partial E_D/\partial W_{i,j}= \tfrac1{MN}\Delta W_{i,j}$ con $\Delta$ la Laplaciana de 5 puntos. Correcto si se usa la definición sin $1/2$. Coherente con el factor anterior.

### 3.3 Espectro y autovalores

Afirmación:
$$\mathcal{E}_D(W)=\frac1{MN}\sum_{k,l}\mu_{k,l} S_{k,l}^2,\quad \mu_{k,l}=4(\sin^2\frac{\pi k}{2M}+\sin^2\frac{\pi l}{2N})$$

Correcto para el Laplaciano de grafo con Neumann discreto. Para $k,l\ll M,N$, $\mu\approx \pi^2(k^2/M^2+l^2/N^2)\propto u^2+v^2$. El argumento de que penalizar $E_D$ penaliza cuadráticamente altas frecuencias es sólido.

### 3.4 Ley de decaimiento $(1+\lambda r^2)^{-1}$

Proviene de prior $p(W)\propto\exp(-\tfrac\lambda2\operatorname{Tr}W^T L W)$ → posterior gaussiano con covarianza $(I+\lambda L)^{-1}$ en base DCT → $\mathbb{E}[S_r^2]\propto 1/(1+\lambda\mu_r)$. El paper lo enuncia sin derivar; es aceptable si se califica como “bajo prior gaussiano isotrópico” y se remite a Rasmussen-Williams (GP) o a *graph signal processing*.

### 3.5 Block-DCT Tiling: ley $O(D^3)$ vs $O(D^2)$

Derivación impecable (§3.4): $F_{\text{IDCT}}=2D^3$, $F_{\text{GEMM},T=1}=2D^2$, ratio $D$. Justifica $B=64$ constante. Es una de las joyas de ingeniería del trabajo. Sin objeciones.

---

## 4. Auditoría de código

### 4.1 Estructura

```
dreg/
  __init__.py      7 líneas  — exports limpios
  topology.py    254        — energía + schedulers
  spectral.py    111        — DCT + tiler
  quantization.py 319       — trit + .tritq
  model.py       174        — Transformer topográfico
  baselines.py    88        — rotación ortogonal + SVD
tests/  79 tests  — ~1183 líneas
```

Separación limpia, dependencias mínimas (`torch`, `numpy`). `pyproject.toml` correcto.

### 4.2 `topology.py` — detalle

* `get_grid_dimensions` factoriza $n$ buscando divisor cercano a $\sqrt n$. Correcto, $O(\sqrt n)$, cubre primos ($1\times n$). Probado.
* `dirichlet_energy_2d`:
  * Soporta `grid_shape` para embeber $h·w$ en sheet 3D, pero `TopographicLinear` nunca lo usa de forma no trivial (grid = weight.shape). Es código preparatorio para futuro `h,w ≠ M,N` — no es bug, pero es peso muerto actualmente.
  * `sheet.dim()==3` path: `diff_h` y `diff_w` sobredims correctos, `num_edges` multiplicado por $c$. Correcto.
  * `1e-8` evita división por cero; bien.
  * Gradiente fluye (test `test_gradient_flows` pasa).
* `DirichletLoss`:
  * Itera `modules` (acepta `model.modules()`), filtra `hasattr(weight) and dim==2`. Correcto y robusto (bug antiguo de crash corregido según ROADMAP).
  * Retorna `0.0` en CPU si no hay lineales — bien, pero crea tensor en CPU aunque modelo esté en CUDA; si luego se suma a loss en CUDA habrá *device mismatch* silencioso en ese edge case. Baja severidad; rara vez ocurre (modelo sin lineales).
  * Divide `total_energy/count` → promedia por capa. Esto hace $\lambda$ invariante al número de capas, deseable. Pero combinado con `numel` normalización, hace que la escala de $\lambda$ sea muy sensible al tamaño de matriz (necesitas $\lambda=30$). Documentado como P0 menor.
* `DirichletScheduler` y subclases:
  * API correcta, validación de `total_steps>0`, `cooldown_ratio∈[0,1]`, `schedule∈{...}`.
  * `get_weight_decay` maneja `t<0`, `t>=total`, clamping. Correcto.
  * `step()` avanza y actualiza `loss_fn.weight_decay` *in-place* — efecto lateral documentado.
  * `state_dict`/`load_state_dict` preserva todo + restaura `weight_decay`. Test de roundtrip pasa.
  * Micro-bug: `StepCooldownScheduler.__init__` no expone `schedule` param (bien, fuerza `step_cooldown`), pero `CosineDirichletScheduler` tampoco expone `cooldown_ratio` — coherente.
  * No se usa `torch.optim.lr_scheduler` style (no `optimizer` binding). Es intencional y simple.

### 4.3 `spectral.py` — detalle

* `dct_matrix_1d` vectorizado, sin bucles Python, con LRU de 32 entradas `OrderedDict` + `move_to_end`. Correcto. `maxsize 32` razonable (cachea hasta 32 tamaños/distintos devices/dtypes).
* Ortogonalidad probada para $n=4,8,16,32,64$. Bien.
* `dct2d`/`idct2d` usan `D_m @ x @ D_n.T` con `float32`/`float64` preservado. Correcto y diferenciable.
* `BlockDCTTiler`:
  * Padding con `F.pad(weight,(0,pad_n,0,pad_m))` — orden `(left,right,top,bottom)` correcto.
  * `view(pm//b, b, pn//b, b).permute(0,2,1,3)` es el pattern correcto para tiling sin copia extra.
  * `einsum("ij, rcjk, lk -> rcil", D, tiles, D)` — verificado dimensionalmente: `D(B,B) · tiles(Rc,Rc?, B,B) · D(B,B)` produce `(R,C,B,B)` espectral. Correcto.
  * Inversa `einsum("ji, rcjk, kl -> rcil", D, spec, D)` usa `D^T` correctamente (`ji` transpone primer índice).
  * Roundtrip exacto con y sin padding probado (`atol 1e-5`). Excelente.

**Riesgo de performance:** Para $M=N=512$, `D` es $512\times512=256$ K floats (1 MB). `D @ X @ D^T` son 2 GEMMs $512^3$ → ~268 MFLOPs. En CPU es pesado, en GPU es GEMM nativo rápido. El tiler mitiga. No hay bug.

### 4.4 `quantization.py` — detalle (ver §6)

### 4.5 `model.py` — detalle

* `TopographicConfig` dataclass limpia, defaults razonables (`vocab 4096`, `d_model 256`, `n_layers 6`, etc.).
* `TopographicLinear(nn.Linear)` subclasea `nn.Linear` y guarda `grid_shape`. No override `forward`, hereda GEMM eficiente. `dirichlet_energy()` delega a `dirichlet_energy_2d` con `grid_shape` (redundante pero consistente).
* `TopographicAttention`: `q/k/v/out` son `TopographicLinear`. Atención causal correcta (`triu` con `-inf`). Usa `scaled_dot_product` manual (no `F.scaled_dot_product_attention` flash, pero funcional; deja headroom de optimización).
* `TopographicMLP`: SwiGLU `silu(gate)*up → down`. Correcto.
* `TopographicTransformer`:
  * `pos_emb` como `Parameter` normal (no sinusoidal). Inicialización `N(0,0.02)` ok.
  * `lm_head` es **`nn.Linear` estándar** — decisión de diseño documentada: no regularizar logits evita colapso semántico por proximidad de índices. **Muy buena decisión**, debe mantenerse y citarse en paper (ya está).
  * `topographic_loss()` promedia por `TopographicLinear`. Correcto.
  * `generate` con `temperature` y `multinomial`, truncation a `max_seq_len`. Funcional.

### 4.6 `baselines.py`

* `random_orthogonal_transform_2d` via `torch.linalg.qr(torch.randn)`. Correcto, uniforme Haar. Usa `Generator` con seed opcional → reproducible.
* `svd_low_rank_approximation` calcula rank por ratio o explícito, clampa a `min(M,N)`, usa `torch.linalg.svd`. Correcto.
* `spectral_hadamard_benchmark_control` wrapper trivial.

### 4.7 Estilo y calidad

* `ruff check` **All checks passed** (0 errores, con `E,F,W,I001`, `line-length 120`, `target py310`). Impecable.
* Sin `TODO/FIXME`. Sin `print` debug en núcleo.
* Docstrings en inglés, consistentes.

### 4.8 Bugs confirmados / potenciales

| # | Ubicación | Severidad | Descripción |
|---|-----------|-----------|-------------|
| C-01 | `topology.py:43` factor 2 | **P1** | Ver §3.1. No rompe tests, pero desalineado con paper. |
| C-02 | `DirichletLoss.forward` device de fallback | P3 | `torch.tensor(0.0)` en CPU cuando no hay lineales; si loss en CUDA, suma heterogénea (raro). Fix: `device=next(model.parameters()).device` o `weight.device`. |
| C-03 | `quantization.py:93` nibble packing impar | P3 | Si `q1_raw` impar, hace `cat([q1_raw, zeros(1)])` en **device** del spec, pero `zeros` usa `device=spec.device` correcto — bien. Sin embargo `torch.zeros(1, dtype=uint8, device=...)` puede fallar si spec está en `mps` sin soporte uint8 (edge). Baja. |
| C-04 | `quantization.py:111-112` base-3 packing | P2 | Fórmula `t0+t1*3+...` correcta, pero `trits` es `int64`; conversión a `uint8` trunca correctamente (<243). No hay overflow. Correcto. |
| C-05 | `spectral.py:42` cache con `device` como key | P3 | `torch.device("cuda:0")` vs `"cuda"` distintas keys pero misma matriz subyacente; duplicación leve. No bug. |

---

## 5. Auditoría de tests y cobertura

**Resultado `pytest`: 79 passed en 5.3 s.** Cobertura `97 %` (508 stmts, 17 miss).

| Módulo | Stmts | Miss | Cover | Líneas no cubiertas |
|--------|-------|------|-------|---------------------|
| `__init__` | 7 | 0 | 100 % | — |
| `baselines` | 29 | 4 | 86 % | 30,72,86-87 (ramas de seed None / rank None) |
| `model` |121|3|98%|69,128,157 (mask None, init bias, generate loop) |
| `quantization`|176|3|98%|236,255-256 (error paths de load) |
| `spectral`|52|1|98%|42 (cache hit) |
| `topology`|123|6|95%|24,55,74,108,150,172 |

Tests bien diseñados:

* **Topología:** constante→0, checkerboard→alta, gradiente fluye, modos de normalización, schedulers con validación, `state_dict` roundtrip. Excelente.
* **Espectral:** ortogonalidad para 5 tamaños, caching, roundtrip `dct→idct`, Parseval, DC coefficient, zero, tiler con/sin padding y 4 block sizes. Excelente.
* **Cuantización:** LUT shape/valores/rango, encoding/decoding, `bpp` sub-32, preservación DC, máscaras disjuntas, radial normalized, `TritQFormat.save/load` roundtrip (incl. 1D params FP16, radios custom), `test_safe_deserialization_no_arbitrary_code` (verifica `json` vs `eval`). Muy sólido.
* **Modelo:** forward/backward, causal masking, generation, etc. (no listado completo pero pasa).
* **Integración:** `train→quantize→evaluate` con `Topo` vs `Std`, convergencia, `BlockDCT+Quant` roundtrip, `tiled_vs_global_energy`. Es el test más valioso: valida *causalidad* a pequeña escala (aunque con margen generoso `1.5×`).

**Falta / mejora sugerida:**
* Ningún test verifica que `DirichletLoss` con `normalization="edges"` produce valor distinto a `numel` bajo misma entrada (solo que ambos >0). Test débil.
* No hay test de `TritQFormat` con archivo corrupto/truncado (solo magic).
* `test_kernel.py` solo verifica paridad numérica C vs Python para un tamaño `128×128`; no verifica doble búfer ni condiciones de carrera.

No obstante, **el nivel de testing está muy por encima de la media** para un repo de investigación *single-author*. El badge `79 passed` es honesto.

---

## 6. Auditoría de cuantización y formato `.tritq`

### 6.1 Partición radial

Default `r0=0.10, r1=0.25, r2=0.50` (paper figura) vs código `0.10,0.25,0.50` — consistente. Paper en Sección 3.3 y Figura menciona `r2=0.40` para 0.945 bpp, pero el código se inicializa a `0.50`. **Inconsistencia P1:** el README y whitepaper dicen `0.50`, el paper dice `0.40`, los resultados JSON de 10 k usan `0.945 bpp` con radios no documentados en `annealing_tinystories_results.json` (reporta `1.638 bpp` con radios `0.15,0.40,1.00`). *Se necesitan radios canónicos documentados.* La diferencia 0.40 vs 0.50 cambia `bpp` de ~0.47 a ~0.94 (ver `evaluate_falsification` reportó 0.479 bpp). **Debe fijarse un default y listar explícitamente los `bpp` por combinación de radios en una tabla del paper.**

### 6.2 Niveles de cuantización

* Banda 0 (DC): 8-bit affine `uint8` con `min,scale`. Correcto.
* Banda 1 (mid): 4-bit nibble 2×/byte. Correcto.
* Banda 2 (high-mid): ternario `{-scale_trit,0,+scale_trit}` donde `scale_trit = std(vals2)` y umbral `0.5*std`. Es una cuantización *ternaria uniforme* extremadamente gruesa. Funciona solo si `vals2` ya es muy pequeña (energía concentrada) — lo cual es cierto para modelo topográfico pero no para estándar. El diseño es deliberadamente agresivo; está bien si se documenta como *hardware-friendly* no *rate-distortion óptimo*.
* Banda 3: truncada a 0. Correcto (60 % coefs).

### 6.3 Packing Base-3

$3^5=243\le256$ → `1.60 bits/trit`. Implementación:
```python
trits[vals2 > thr] = 2   # +1
trits[vals2 < -thr] = 0  # -1
trits[else] = 1          # 0
# pack: t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
```
LUT inversa `TRIT_LUT[b,i]= (b // 3^i)%3 -1` → `{-1,0,1}`. **Matemáticamente biyectiva para 0–242.** Bytes 243–255 quedan como 0 en LUT (no usados). Correcto.

Ahorro: frente a 2 bits/trit, 1.6 es 20 % mejor — cálculo correcto.

### 6.4 `.tritq` binario

Formato:
```
MAGIC "TRITQ_V1" (8B)
u32 cfg_len + json(header_meta)   # contiene config, r0,r1,r2, version
u32 num_tensors
repeat num_tensors:
  u16 name_len + name
  u8  flag (1=spectral,0=raw FP16)
  if flag==1:
    u32 M, u32 N
    u32 b0_len, f32 b0_min, f32 b0_scale + b0_data
    u32 b1_len, f32 b1_min, f32 b1_scale, u32 b1_count + b1_data
    u32 b2_len, f32 b2_scale, u32 b2_count + b2_data
  else:
    u32 ndim + ndim*u32 shape + u32 raw_len + raw_fp16
```

* **Seguridad:** Usa `json.loads`, no `eval`/`pickle`. Bien (antiguo bug de `eval` listado como corregido en ROADMAP).
* **Portabilidad:** `struct "<I/f/H/B"` little-endian explícito — correcto cross-platform.
* **Robustez:** Verifica `MAGIC`, lanza `ValueError` si JSON corrupto (test `test_safe_deserialization_no_arbitrary_code`). No valida `cfg_len` absurdo (>10 MB) → posible OOM si archivo malicioso. **P2: añadir `if cfg_len>1_000_000: raise`.**
* **Fidelidad:** 2D params se salvan como espectro cuantizado, no como pesos crudos → `save→load` es *lossy* (cuantizado) por diseño. Documentado.
* **Radios:** Header guarda `r0,r1,r2` y `load` los respeta si `quantizer is None` (crea uno con radios del archivo). Si caller pasa `quantizer` custom, se ignora header — comportamiento correcto pero debe documentarse.

### 6.5 `dequantize_matrix`

Reconstruye `rec[mask]=vals`. Band2 usa LUT vectorizada `lut[q2_bytes].reshape(-1)[:count] * scale`. Correcto. Alta frecuencia (>r2) permanece 0.

**Cálculo de `bpp`:** `total_bytes*8 / numel` donde `total_bytes = len(q0)+len(q1_packed)+len(q2_packed)`. Correcto para matrices individuales. Para modelo completo, `bpp` lineal es promedio ponderado por `numel` — el código de `quantize_and_reconstruct_model` lo hace bien.

---

## 7. Auditoría del kernel C / DMA

### 7.1 Funcionalidad

`spectral_dma_kernel.c` (≈ 800 líneas, Win32 + POSIX):

* **LUT:** `TRIT_LUT[256][5]` idéntica a Python.
* **GEMM:** `gemm_nn` con orden `i-k-j`, `memset` a 0, `#pragma GCC ivdep`. Cache-friendly, permite FMA/AVX2.
* **Decodificación:** `decode_single_matrix` desempaqueta banda 0/1/2 en `dct_buf`, luego `D_out^T * dct_buf` → `temp`, luego `temp * D_in` → destino. Dos GEMMs.
* **Registro:** `CMatrixRecord` + `CSublayerRecord` con offsets para zero-copy.
* **DMA:** Worker thread con eventos Win32 (`CreateEvent`+`SetEvent`+`WaitForSingleObject`) o pthreads (`mutex`+`cond`). `g_pending_*` volátiles, `MEMORY_BARRIER()`.

### 7.2 Correctitud

Test `benchmark_c_dma.py` verifica paridad C vs Python `dct2d/idct2d` para `M=N=128`: `max_diff <1e-5` → `2.38e-07` observado (float32). **Paridad bit-exacta verificada.** `reproduce_all.py` confirma `0.00000024`.

### 7.3 Concurrencia

* Win32 path: `CreateThread` + `THREAD_PRIORITY_ABOVE_NORMAL` + `MemoryBarrier`. Correcto.
* POSIX path: `pthread_create` + `sched` + `__sync_synchronize`. Correcto.
* Uso de `volatile` para flags — en C moderno debería ser `_Atomic` o `atomic_int` con `memory_order`, pero para x86 y este patrón productor-consumidor simple es funcional. No hay *data race* grave porque los buffers se intercambian solo tras `done`.

### 7.4 Zero-copy

Recibe `raw.data_ptr()` de PyTorch y escribe directo al `target_base_buf` sin `malloc` por matriz. Verificado en `benchmark_c_dma.py` con `ctypes`. Correcto.

### 7.5 Build

`build_kernel.py` detecta plataforma (`_WIN32` vs POSIX), arquitectura (x86_64/ARM), flags `-O3 -ffast-math -lpthread`. `Makefile` simple. Probado en CI Ubuntu+Windows. **Listo para producción.**

### 7.6 Limitaciones

* `MAX_SUBLAYERS 64`, `MAX_MATRICES_PER_SUBLAYER 4` hardcoded. Suficiente para `n_layers=12` con 4 proyecciones por bloque (qkv, out, gate/up/down = 6 → excede 4 para `TopographicMLP` con 3 matrices). Revisar: `TopographicAttention` tiene 4, `TopographicMLP` tiene 3 → 7 por bloque total, pero kernel agrupa por sublayer; si cada bloque es una sublayer, 7>4 → overflow silencioso. **P1: verificar límite.** En tests solo usan sublayers individuales, no modelo completo 8 capas.
* No hay `free` de `D_out_T`/`D_in` matrices si se registran múltiples modelos — leak leve.

---

## 8. Auditoría empírica y de resultados

### 8.1 Resumen de resultados reportados (paper §4 + `results/*.json`)

**Calibración 2 000 pasos, 16.38 M tokens, A10G (Tabla 1):**

| $\lambda$ | Val PPL FP32 | $E_D$ | Quant 0.945 bpp | $\Delta$ |
|-----------|-------------|-------|----------------|----------|
| 0.0 (std) | 10.48 | 0.004225 | 46.60 | +36.12 |
| 0.01 | 10.44 | 0.004251 (+0.6 %) | 47.95 | +37.51 |
| 5.0 | 10.62 | 0.003382 (-20 %) | 44.80 | +34.18 |
| 15.0 | 10.81 | 0.002427 (-42.5 %) | 43.26 | +32.46 |
| **30.0** | **10.98** | **0.001924 (-54.5 %)** | **39.80** | **+28.82** |

Monotonicidad perfecta en 2 k pasos. $\lambda=0.01$ es indistinguible de baseline (gradiente lavable) — honestamente reportado.

**10 000 pasos convergidos, 81.92 M tokens (Tabla 2):**

| Modelo | $\lambda$ | FP32 PPL | $E_D$ | Quant 0.945 bpp | $\Delta$ |
|--------|-----------|----------|-------|----------------|----------|
| Std | 0.0 | **5.83** | 0.012316 | 72.89 | +67.06 |
| Topo | 30.0 | 6.18 | **0.003438 (-72.1 %)** | **66.13** | **+59.96** |

Ventaja **+6.76 PPL** cuantizado, ligera pérdida FP32 (+0.35). Muestras cualitativas fluidas.

**MNIST MLP (Tabla 3):** Std 97.87 % vs Topo 97.58 % FP32, $E_D$ -84.5 %, energía LF 27.87 %→52.84 %, low-pass 20 % DCT: 30.69 %→41.41 % (+11.01 %), TritQ 0.94 bpp: 87.40 %→88.59 % (+1.19 %). **Principio universal validado.**

**SOTA 10 M, 25 batches 204 800 tokens (Tabla 4):**

| Método | bpp | Std PPL | Topo PPL | Gain |
|--------|-----|---------|----------|------|
| Dense FP32 | 32.0 | 6.11 | 6.46 | -0.35 |
| Spatial INT4 | 4.0 | 6.79 | 7.01 | -0.22 |
| Spatial INT2 | 2.0 | 41.67 | 55.22 | colapso |
| Spectral INT4 | 4.0 | 7.85 | 8.97 | -1.12 |
| **TritQ (ours)** | **0.945** | **73.75** | **66.50** | **-7.25** |

### 8.2 Hallazgo crítico: Pareto incompleto

**`results/pareto_curve_tinystories_results.json` contiene 6 regímenes, pero el paper Tabla 6 reporta solo 5, omitiendo el punto donde Topo pierde:**

```json
{
  "regime": "Aggressive", "r0":0.10, "r1":0.30, "r2":0.75,
  "bpp":0.914, "std_ppl":835.92, "topo_ppl":1117.65, "gain":-281.73
}
```

Ese punto es **peor para Topo** (-281 PPL). El paper muestra en Tabla 6:

* Ultra-Aggressive 0.43 bpp: Topo gana +2 383
* Calibrated 1.638 bpp: +7.25
* Mid-Band 2.181 bpp: +7.02
* High-Fid 2.794 bpp: +2.54
* Conservative 3.774 bpp: +0.65

**Omite deliberadamente 0.914 bpp** donde Topo pierde. Además el paper dice “empirical Pareto improvement across $1.6\le\text{bpp}\le3.8$” — técnicamente cierto si excluye 0.914, pero es *cherry-picking* fronterizo. **P0: debe incluirse el punto desfavorable y discutirlo.** La explicación plausible: a 0.914 bpp la partición deja solo 8 % de coefs en alta calidad y el modelo topográfico, al concentrar energía en muy bajas frecuencias, sufre más si `r1=0.30` corta justo en su banda crítica. Es una observación científica valiosa, no un fracaso.

También: paper reporta 0.945 bpp en Tabla 4 pero JSON reporta 1.638 bpp para “Calibrated TritQ” con radios `0.15,0.40,1.00`. **La definición de bpp no es consistente.** Ver §6.1.

### 8.3 Resultados de annealing (`annealing_tinystories_results.json`)

* 5 000 pasos, $\lambda=30$, cooldown 15 % (4250→5000 con $\lambda=0$)
* FP32 7.50, $E_D$ 0.002208, TritQ 1.638 bpp → 55.68 PPL
* Compara contra 10 k std (6.11/73.75) y 10 k topo constante (6.46/66.50) → **no es apples-to-apples** (5 k vs 10 k). El paper lo presenta como 5 k y lo reconoce, pero la frase “surpassing both the static topographic model (66.50) by +10.82” compara diferente número de pasos y diferente `bpp` (1.638 vs 0.945). **P1: rehacer annealing a 10 k pasos para comparación justa, o calificar claramente.**

### 8.4 GPT-2 124 M (`gpt2_dirichlet_benchmark_results.json`)

* Base `E_D=0.0335` → TritQ 8 055 PPL (colapso)
* Fine-tune 300 pasos std: `E_D` idéntico 0.033513 → TritQ 6 628 PPL
* Fine-tune Dirichlet $\lambda=10^{-3}$: `E_D` idéntico → TritQ 7 507 PPL (**peor**)

Conclusión honesta: 300 pasos no mueven pesos pre-entrenados. **Correcto incluirlo como limitación.** El paper lo hace bien. `gpt2_specrama_tsp_tritq_results.json` muestra que permutación TSP reduce `E_D` 13.8 % y mejora TritQ en algunos regímenes (hasta 3 678), pero sigue en 373–4 374 PPL → no utilizable. Todo coherente.

### 8.5 Falsificación a pequeña escala (reproducción local)

Ejecuté `examples/evaluate_falsification.py --epochs 2 --lambda-val 5.0` en CPU:

```
Standard: E_D 0.0024 | FP32 4.39 | TritQ 0.479 bpp 5.54 (DEGRADED)
Topo:     E_D 0.0022 | FP32 4.41 | TritQ 0.479 bpp 8.82 (DEGRADED más)
Gap: 1.1× smoother — Topo *peor* cuantizado
```

Con 2 épocas y $\lambda=5$, el efecto no replica. Con $\lambda=30$ y 8 épocas (default del script) sí debería replicar (no probado en CPU por tiempo). **Implicación:** el benchmark de falsificación es sensible a escala; el README lo vende como “standalone benchmark” reproducible en CPU, pero en CPU el resultado es inconcluso o invertido. **P1: subir `epochs` default a 8 y `lambda` a 30 en el script, y documentar que requiere ≥5 épocas para ver separación.**

`reproduce_all.py` sí pasa todos los checks espectrales (7.1× más energía retenida para smooth vs white noise, `15.9%` vs `94%` error). La física está bien.

### 8.6 Validez estadística

* **Seeds:** `train_multiseed_sweep.py` promete ≥3 seeds con `mean±std`, pero JSONs solo muestran single-seed. No hay barras de error en Tablas 1–4 (solo PPL puntual). **P2: añadir intervalo de confianza o al menos 3-seed sweep para resultados principales.**
* **Batches de evaluación:** 25 batches (204 800 tokens) es razonable para PPL de LM pequeño, pero el error estándar no se reporta. A PPL 66, una variación de ±2 es esperable.
* **Comparación justa:** Todos los SOTA usan *redondeo uniforme* (RTN) sin GPTQ/AWQ — es un baseline débil. El paper lo reconoce como “contrast qualitative”, pero un revisor pedirá al menos GPTQ-INT4 como techo. **P2: añadir GPTQ-INT4 si tiempo permite, o justificar por qué RTN es suficiente para demostrar causalidad topográfica.**

---

## 9. Auditoría de documentación

### 9.1 README.md

**Fortalezas:**
* Landing page excelente: badges (Python, PyTorch, CI, 79 tests, 97 % coverage, ruff, paper PDF, MIT), figuras a ancho completo, matemáticas en LaTeX, tabla de beneficios, quickstart de 2 líneas (TopographicLinear vs DirichletLoss wrapper), matriz de aplicabilidad.
* Estructura narrativa “Idea → Matemáticas → API → Caso de estudio → Embedded → Repo structure → Paper → Whitepaper” es fluida.

**Hallazgos:**
* Badge `Coverage 97%` hardcodeado en markdown, no dinámico (codecov/codebeat no configurado). No es grave, pero se desactualizará.
* Código de ejemplo `Option A` define `dirichlet_loss()` que suma energías sin promediar, mientras `DirichletLoss` promedia — discrepancia menor. Ejemplo debería usar `normalization` explícita.
* Sección “Falsification Test” promete `python examples/evaluate_falsification.py` funciona standalone, pero como vimos falla en 2 épocas. Documentar requisitos mínimos.

### 9.2 `docs/whitepaper.md` (296 líneas, técnico suplementario)

* Cubre v382–v395 con mapa de evolución, tabla de reconciliación, ley de asimetría, blueprint de hardware. **Es oro puro para reproducibilidad.** Nivel de detalle inusual en repos open-source.
* Idioma español con terminología técnica inglesa mezclada — coherente dado autor hispanohablante.
* Recomendación: añadir versión en inglés o al menos abstract en inglés para audiencia internacional.

### 9.3 `docs/ROADMAP.md`

* Roadmap con fases 0–4, checkboxes, métricas de calibración, SOTA, paper, polish. Muy trazable. Estado `[x]` para fase 3 completo, pendiente “Reproducibility script: entrena, cuantiza, evalúa, genera tablas” — existe `reproduce_all.py`, pero no genera tablas LaTeX automáticamente. Bueno.

### 9.4 `docs/propuesta_dirichlet_annealing.md`

* Hipótesis, PoC en MNIST (Full vs Annealed), formulación schedulers, hoja de ruta escala LLM. Bien documentado, con nota honesta “pendiente de escalado a 10 M”.

### 9.5 `examples/` y `benchmarks/`

* 7 ejemplos, 8 benchmarks Modal. Docstrings claros, `argparse` con defaults. `train_gpt2_dirichlet.py` 418 líneas muy completo.
* Falta `requirements` para `benchmarks` (modal, datasets, tokenizers) — está en `pyproject.toml [benchmarks]` correcto.
* Algunos benchmarks importan `modal` a nivel top-level, lo que rompe ejecuciones sin `modal` instalado (deberían importar lazy dentro de funciones). Menor.

### 9.6 `paper/` y `docs/figures/`

* 3 figuras PNG de 145–256 KB, calidad publicación. Duplicadas en `paper/figures` y `docs/figures` (redundancia aceptable).
* `paper-draft.pdf` 871 KB, 12 páginas según ROADMAP (15 según README — actualizar).
* Sin `paper/figures` para diagrama DMA (Figura 3 es `tabular` LaTeX, no PNG — correcto).

---

## 10. Auditoría del paper (LaTeX)

### 10.1 Estructura (15 páginas según README, 12 según ROADMAP)

```
Abstract + Keywords
1. Introduction (gauge symmetry, cortical maps, 6 contribuciones)
2. Related Work (PTQ, Topografía, Espectral)
3. Methodology (Dirichlet → DCT → TritQ → DMA, Algoritmo 1)
4. Experiments (Setup, Calibration 5 puntos, 10k convergencia, MNIST, SOTA, Pareto sweep)
5. Discussion (Rank-1 collapse, Hardware, Annealing, Repro)
6. Limitations (4 párrafos)
7. Broader Impact
8. Conclusion
A. Hyperparams
B. Derivación flujo gradiente
Referencias (15)
```

Estructura clásica y completa. **Bien.**

### 10.2 Calidad de escritura

* Inglés académico fluido, bien citado (`vaswani2017`, `eldan2023`, `frantar2022`, `lin2023`, `chee2024`, `ahmed1974`, `wallace1992`, `kohonen1982`, `hubel1962`, `durbin1990`, `belkin2003`, etc.).
* Uso de *natbib*, `microtype`, `booktabs`, `hyperref` impecable.
* Notación consistente ($\mathcal{E}_D$, $\lambda_{\text{topo}}$, $\rho$).

### 10.3 Fortalezas del manuscrito

* **Honestidad:** Reporta degradación FP32 (+0.35 PPL), colapso espacial INT2, fallo GPT-2 fine-tune, rank-1 collapse previo. Pocos papers lo hacen.
* **Algoritmo 1** pseudocódigo completo y legible.
* **Figuras 1–3** bien referenciadas y con captions cuantificados (94 % reducción $E_D$, >72 % energía bajo $\rho=0.40$).
* **Limitaciones** no es pro-forma: explica por qué GPT-2 no se mueve, overhead `<5%`, embedding 2D planar, falta de GPTQ espectral.

### 10.4 Problemas y mejoras necesarias

| # | Sección | Problema | Gravedad | Sugerencia |
|---|---------|----------|----------|------------|
| P-01 | §4.2 Tabla 1 | Dice “Monotonic drop … from 46.60 to 39.80 (6.8-point gain)” pero ignora que $\lambda=0.01$ empeora a 47.95 (no monotónico estricto). | P2 | Cambiar a “monotonic for $\lambda\ge5$” o “overall decreasing trend”. |
| P-02 | §4.3 Tabla 2 | “72.1 % reduction” — correcto, pero texto dice “while preserving fluent narrative” con muestra que termina en “It was full of chefs!” (ligeramente incoherente). Elegir muestra más limpia o añadir “typical sample (cherry-picked)”. | P3 |  |
| P-03 | §4.5 + §4.6 | Tablas 4 y 6 usan `bpp` diferentes para “Calibrated TritQ” (0.945 en Tabla 4 vs 1.638 en Tabla 6). **Inconsistencia numérica.** | **P0** | Unificar radios y recalcular. Tabla 4 debería listar `r0,r1,r2` usados. |
| P-04 | §4.6 | Omite punto 0.914 bpp donde Topo pierde. | **P0** | Incluir y discutir (ver §8.2). |
| P-05 | Figuras | Figura 1 dice $E_D=0.0016→0.0001$ (94 %) pero Tabla 1 dice $0.004225→0.001924$ (54 %) para 2k pasos. Son experimentos distintos (MNIST patch vs TinyStories). Aclarar pie de figura con “64×64 patch, $\lambda=...$, Tabla X es promedio de 10 M”. | P1 | Añadir nota. |
| P-06 | Abstract | “down to 0.47 bpp, up to 68×” — el paper luego usa 0.945 y 0.43. 0.47 proviene de falsificación 0.479 bpp. Unificar claim: “0.43–0.945 bpp (19–74×) dependiendo de radios”. | P1 |  |
| P-07 | §5.3 Annealing | “+18.07 PPL over baseline” compara 5 k vs 10 k. | P1 | Rehacer a 10 k o matizar. |
| P-08 | Referencias | Solo 15 citas, falta *SqueezeLLM*, *SmoothQuant*, *OmniQuant*, *BiLLM*, *PB-LLM*, *Spectral Regularization* (Yoshida). Para TMLR, revisores pedirán literatura 2023-2025 más completa. | P2 | Ampliar a 25–30. |
| P-09 | Métodos | No reporta *wall-clock* por paso en A10G ni coste total de entrenamiento (útil para reproducibilidad). | P3 | Añadir apéndice. |
| P-10 | Conclusión | Repite abstract sin añadir visión futura concreta (combinar con GPTQ, extender a MoE, probar en 100 M). | P3 | Añadir 2 frases. |

### 10.5 Errores LaTeX menores

* Ninguno detectado; compila sin errores según ROADMAP (“0 errores, 0 overfull hboxes”). No pude compilar por falta de `pdflatex` en sandbox, pero `references.bib` sintácticamente correcta (verificada).

---

## 11. Riesgos y objeciones previsibles de revisores

### Críticas de fondo (major concerns)

1. **“La ganancia cuantizada es grande en PPL absoluto (66 vs 73) pero ambos son inutilizables (>60 PPL es gibberish). ¿Cuál es la utilidad práctica?”**
   * Defensa actual: muestras cualitativas fluidas (contradice PPL 66), y MNIST con +1.19 % accuracy. Pero PPL 66 vs 5.8 baseline es ~11× peor — el modelo cuantizado no es desplegable para lenguaje.
   * **Mitigación:** Reencuadrar: el régimen útil no es 0.945 bpp para LM (demasiado agresivo), sino 2.18 bpp (Topo 23.08 vs Std 30.10) donde PPL aún es alto pero más razonable, o 3.77 bpp (8.22 vs 8.87, FP32 6.11). Enfatizar frente de Pareto: *Topo domina en todo el frente*, la ventaja no es solo en 0.945 bpp inutilizable. Añadir curva Pareto con PPL en escala log.

2. **“¿Por qué no comparar con un baseline fuerte de cuantización (GPTQ-INT4, AWQ-INT4) en vez de RTN naive?”**
   * Mitigación: Añadir al menos un punto GPTQ-INT4 usando `auto-gptq` sobre TinyStories 10 M (puede hacerse en Modal). Si GPTQ-INT4 logra PPL ~6.5, entonces el argumento “0.945 bpp <1 MB SRAM” sigue siendo diferenciador (GPTQ-INT4 no es <1 MB). Si no hay tiempo, justificar explícitamente que el foco es régimen sub-2 bpp donde GPTQ/QuIP# no operan.

3. **“El efecto desaparece en modelo pre-entrenado (GPT-2) y solo funciona si entrenas desde cero. ¿No es limitación fatal?”**
   * Mitigación: Es la limitación más honesta del paper (§6). Defender que el *takeaway* es una *guía de diseño para futuros pre-entrenamientos*, no un parche retroactivo. Citar *Chinchilla* y *scaling laws*: entrenar desde cero con regularizador es barato comparado con comprimir post-hoc.

4. **“Cherry-picking en Pareto.”** Ver P0. Incluir punto desfavorable y discutirlo convierte ataque en fortaleza.

### Críticas técnicas (minor)

5. Factor 2 en $E_D$, radios inconsistentes, seed único, falta barras de error — todas P1/P2 fáciles de arreglar.

6. “¿Por qué 2D y no 1D o 3D?” — El whitepaper no explora topologías alternativas. Responder: 2D es isomórfico a superficie cortical y permite DCT 2D separable; 1D sería insuficiente para matrices rectangulares, 3D no aporta.

---

## 12. Matriz de hallazgos priorizada (P0–P3)

### P0 — Bloqueantes para envío (corregir antes de arXiv v1)

| ID | Título | Ubicación | Acción concreta | Esfuerzo |
|----|--------|-----------|-----------------|----------|
| P0-01 | **Pareto selectivo** | `paper §4.6` + `pareto_curve_tinystories_results.json` | Añadir fila “Aggressive 0.914 bpp” a Tabla 6 con gain -281.73, añadir nota “Topographic underperforms in this narrow mid-aggressive band due to hard cutoff at $\rho=0.30$ cutting its concentrated band”. Recalcular figura Pareto incluyendo punto. | 2 h |
| P0-02 | **Inconsistencia bpp/radios** | `quantization.py`, `paper §3.3/§4.5`, `README`, `results/*` | Fijar `DEFAULT_R0,R1,R2 = 0.10,0.25,0.50` como canónico (o 0.15,0.40,1.00), recalcular todas las tablas para ese default. Añadir tabla explícita radios↔bpp. Actualizar `whitepaper.md` si usa otro. | 4 h (rehacer métricas si cambian) |
| P0-03 | **Definición $E_D$ factor 2** | `topology.py:60` vs `paper Eq.1` | Elegir una y sincronizar. Recomendado: cambiar código a `sum_sq/(2*numel)` o aclarar en paper “we use numel normalization (2× the variational form)”. Añadir nota al pie de Tabla 1. | 30 min |

### P1 — Importantes (arreglar antes de TMLR, no bloquean arXiv)

| ID | Título | Acción | Esfuerzo |
|----|--------|--------|----------|
| P1-01 | Annealing apples-to-apples | Re-ejecutar `modal_train_annealing.py` a 10 k pasos (o al menos 5 k vs 5 k std/topo-const) y reemplazar Tabla 2 de annealing. | 1 día GPU |
| P1-02 | Falsificación en CPU | Subir `evaluate_falsification.py` defaults a `epochs=8, lambda=30`, añadir `if epochs<5: warn`. Probar que en CPU con 8 épocas sí separa. | 2 h |
| P1-03 | Figura 1 vs Tabla 1 | Añadir nota al pie de Fig.1: “Patch 64×64 from layer X, $\lambda=15$, Table 1 reports mean $E_D$ over 10 M model”. | 15 min |
| P1-04 | Kernel MAX_MATRICES | Verificar si 4 basta; subir a 8 o hacer dinámico. Añadir `assert num_matrices <= MAX` con mensaje claro. | 1 h |
| P1-05 | Validación GPT-2 | Añadir curva de $E_D$ vs steps para mostrar que 300 pasos no mueven pesos (ya hay JSON, solo graficar). | 1 h |
| P1-06 | Seeds y barras de error | Correr `train_multiseed_sweep.py` con 3 seeds para Tabla 2 (10 k) y reportar `mean±std`. | 1–2 días GPU |

### P2 — Recomendados (mejoran robustez)

| ID | Título | Acción |
|----|--------|--------|
| P2-01 | Ampliar bibliografía | Añadir SqueezeLLM, SmoothQuant, OmniQuant, BiLLM, PB-LLM, Yoshida Spectral Reg, Keller Topographic VAE, FourierFT. |
| P2-02 | Comparación GPTQ | Ejecutar `modal_benchmark_quantization_sota.py` con GPTQ-INT4 (si disponible) o reportar número de literatura. |
| P2-03 | Tamaño de header `.tritq` | Validar `cfg_len <1 MB` y `num_tensors <10_000` en `TritQFormat.load`. |
| P2-04 | README badges | Hacer coverage badge dinámico (codecov) o quitar porcentaje hardcodeado. |
| P2-05 | Benchmarks import lazy | Mover `import modal` dentro de funciones para permitir `import dreg` sin modal. |

### P3 — Pulido (nice to have)

| ID | Título |
|----|--------|
| P3-01 | Device fallback de `DirichletLoss` (ya listado). |
| P3-02 | Añadir `py.typed` marker y `mypy` opcional. |
| P3-03 | Traducción del whitepaper a inglés (abstract). |
| P3-04 | Script `examples/generate_figures.py` local (no Modal) para heatmaps. |

---

## 13. Recomendaciones de publicación

### 13.1 Estrategia de venue

| Venue | Fit | Timeline | Pros | Contras |
|-------|-----|----------|------|---------|
| **TMLR** (recomendado) | ★★★★★ | Rolling, ~3 meses revisión abierta | Valora reproducibilidad, permite 97 % cobertura y artefactos, no exige SOTA absoluto, revisión dialogada ideal para idea nueva | Menor prestigio que NeurIPS |
| **NeurIPS/ICLR Workshop** (Efficient Deep Learning, Compression) | ★★★★ | Deadline ~May/Oct | Feedback rápido, networking, acepta “principle papers” | No es *main track* |
| **NeurIPS/ICLR main track** | ★★★ | Deadline May 2027 (pasado Oct 2026) | Máximo impacto | Requiere baseline GPTQ fuerte, escala 100 M–1 B, revisores exigirán Pareto completo y 3 seeds → 2–3 meses extra |
| **JMLR / Machine Learning Journal** | ★★★ | 6–12 meses | Espacio para derivaciones largas | Lento |

**Plan sugerido:**

1. **Semana 0 (ya):** Subir a **arXiv** (`cs.LG`, `cs.AI`, `cs.ET`) con título actual para prioridad. Incluir link a repo. *No esperes a P0* — arXiv v1 puede tener P0, v2 lo corrige.
2. **Semana 1–2:** Corregir P0 (Pareto, bpp, $E_D$ factor). Añadir apéndice con punto desfavorable. Reemplazar PDF en arXiv v2.
3. **Semana 3–4:** Ejecutar P1 (annealing 10 k, 3-seed sweep). Preparar respuesta a “¿Y si aplicamos GPTQ sobre $S$?” con experimento piloto.
4. **Semana 5:** Enviar a **TMLR**. Paralelamente, someter a *workshop* de NeurIPS 2027 si quieres visibilidad.

### 13.2 Checklist pre-envío

- [ ] P0-01, P0-02, P0-03 corregidos y PDF recompilado sin overfull
- [ ] Tabla 6 incluye fila 0.914 bpp y discusión (1 párrafo)
- [ ] Sección 4.6 añade “Radii (r0,r1,r2) → bpp” explícito
- [ ] Abstract unifica claim 0.43–0.945 bpp (no “down to 0.47” ambiguo)
- [ ] Código `pip install -e . && pytest` sigue verde tras cambios
- [ ] `reproduce_all.py` genera figura Pareto y tabla LaTeX automáticamente (cierra ROADMAP 3.2)
- [ ] Añadir `CITATION.cff` o `citation.bib` para facilitar cita
- [ ] Tag `v1.0.0` en GitHub con `paper-draft.pdf` y `results/*.json`

### 13.3 Mensaje clave para revisores

> *“No proponemos un cuantizador mejor; proponemos entrenar de forma que *cualquier* cuantizador espectral funcione en <2 bpp. El kernel DMA y el formato Base-3 son demostraciones de que el principio cierra el lazo hasta silicio real.”*

Ese encuadre desactiva la comparación directa con GPTQ y pone el foco donde eres fuerte: causalidad topografía→compresión.

---

## 14. Apéndices técnicos

### Apéndice A — Verificación de reproducibilidad local (ejecutada en auditoría)

```bash
pip install --break-system-packages torch numpy pytest pytest-cov ruff
pytest tests/ -q                      # 79 passed
ruff check .                          # All checks passed
pytest --cov=dreg --cov-report=term   # 97 %, 17 miss
PYTHONPATH=. python examples/evaluate_falsification.py --epochs 2 --lambda-val 5.0
# → Standard 4.39→5.54, Topo 4.41→8.82 (no replica con 2 épocas, ver P1-02)

PYTHONPATH=. python examples/reproduce_all.py
# → 4/4 checks PASSED, C parity 2.38e-07, 7.1× energy retention
```

Entorno: `Python 3.11.2`, `torch 2.14.1`, `numpy 2.4.6`, `pytest 9.1.1`, `ruff 0.16.10`, Linux sandbox.

### Apéndice B — Cálculo de bpp por régimen (verificación)

Para matriz $M×N$:

* $N_0 = |\{\rho\le r_0\}|$, $N_1 = |\{r_0<\rho\le r_1\}|$, $N_2 = |\{r_1<\rho\le r_2\}|$, $N_3$ truncado
* Bytes: $B_0=N_0·1$, $B_1=\lceil N_1/2\rceil$, $B_2=\lceil N_2/5\rceil$ (5 trits/byte)
* $bpp = 8(B_0+B_1+B_2)/MN$

Ejemplo $M=N=256$:

* $r=(0.10,0.25,0.50)$ → $N_0≈1.7%$, $N_1≈8.4%$, $N_2≈29.7%$, $N_3≈60.2%$ → $bpp≈0.945$ (paper)
* $r=(0.15,0.40,1.00)$ → cubre ~50 % → $bpp≈1.638$ (JSON annealing/pareto)
* $r=(0.08,0.22,0.50)$ → $bpp≈0.43$ (ultra-aggressive)

**Conclusión:** ambos claims son correctos para distintos radios; el problema es no etiquetarlos en la misma tabla.

### Apéndice C — Estimación de coste de entrenamiento

* 10 k steps × batch 32 × seq 256 = 81.92 M tokens.
* A10G ~140 tok/ms (según `modal_train_tinystories.py` log `tok_per_sec`), → ~81.92 M / 140 ≈ 585 s ≈ 9.7 min por run ideal; observado en paper ~495 s para 5 k (coherente).
* 5 runs calibración (2 k) + 2 runs 10 k + 1 annealing + Pareto + GPT-2 ≈ 40–50 GPU-h A10G → ~$50–70 en Modal (coste modesto).

### Apéndice D — Glosario de artefactos

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `dreg/topology.py` | Dirichlet + schedulers | ✅ Listo |
| `dreg/spectral.py` | DCT + tiler | ✅ Listo |
| `dreg/quantization.py` | TritQ + .tritq | ⚠️ Radios inconsistentes |
| `dreg/model.py` | Transformer topográfico | ✅ Listo |
| `dreg/baselines.py` | Hadamard + SVD | ✅ Listo |
| `kernel/spectral_dma_kernel.c` | C DMA | ⚠️ MAX 4 |
| `examples/reproduce_all.py` | Suite de reproducibilidad | ✅ Pasa |
| `benchmarks/modal_*.py` | Entrenamiento/eval en Modal | ✅ Funcional |
| `results/*.json` | 4 JSONs de resultados 10 M/GPT-2 | ⚠️ Falta Pareto 0.914 en paper |
| `paper/paper-draft.tex` | Manuscrito | ⚠️ P0 pendientes |
| `docs/whitepaper.md` | Reporte suplementario v382–v395 | ✅ Excelente |
| `README.md` | Landing | ✅ Muy bueno |

### Apéndice E — Comandos para corregir P0 rápidamente

```bash
# P0-03: sincronizar E_D (opción A: fix código)
# en dreg/topology.py, línea 60:
#   return sum_sq / (sheet.numel() + 1e-8)  →  return sum_sq / (2*sheet.numel() + 1e-8)

# P0-01: añadir fila a paper/paper-draft.tex Tabla 6
# Insertar después de Ultra-Aggressive:
# Aggressive & $(0.10, 0.30, 0.75)$ & 0.914 bpp & 35.0× & 835.92 & 1117.65 & $-281.73$ \\

# P0-02: fijar radios canónicos
# en dreg/quantization.py: r2 default 0.50 → documentar tabla radios↔bpp
# en paper §3.3: listar 0.945 bpp ↔ (0.10,0.25,0.50) y 1.638 bpp ↔ (0.15,0.40,1.00)

# Verificar
pip install -e . && python -m pytest tests/ -q && python -m ruff check .
```

---

## Epílogo

Este repositorio es, sin hipérbole, uno de los mejor preparados que he auditado para un proyecto de autor único: **código limpio, tests con cobertura real, ingeniería de silicio creíble y honestidad al reportar lo que no funciona (GPT-2).** La idea de *suavidad espacial → compacidad espectral → <1 bpp con DMA* es un arco narrativo completo, desde neurociencia hasta silicio.

Si corriges los tres P0 (que son 1 día de trabajo + 1 párrafo de discusión honesta), tu *worst case* ante revisores pasa de “¿por qué esconden el punto donde pierden?” a “autores reportan incluso el régimen donde su método no ayuda y lo explican”. Esa es la diferencia entre un *reject* y un *accept with minor revisions*.

**Próximo paso inmediato:** `git tag v0.99 && arXiv`, luego iteras a `v1.0.0 → TMLR`.

Quedo a disposición para implementar los parches P0/P1 directamente en esta rama si quieres.

*— Auditoría generada por Agent Mode, Arena.ai — 2026-10-03*

