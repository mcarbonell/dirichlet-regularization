# Auditoría 1 — Dirichlet Regularization (`dreg`)

**Fecha:** 2026-10-01
**Alcance:** revisión completa del repo `dirichlet-regularization` (commit `284f499`) — código, paper, ROADMAP, README.
**Método:** lectura de fuente + ejecución de los experimentos del repo + medición independiente de las métricas afirmadas.
**Entorno:** torch 2.10.0+cpu, Python 3.x, Windows.

---

## 1. Resumen ejecutivo

**El núcleo científico es correcto y defendible.** La cadena matemática —energía de Dirichlet → seminorma de Sobolev → Laplaciano discreto → autovectores = base DCT-II→ compactación espectral— es texto de libro y está bien planteada en el paper. Medí el mecanismo y el efecto es grande y real (§5).

**El problema es la evidencia.** Las cifras estrella del repositorio (tasa de bits, tabla de "falsificación", "preservación de calidad") no se reproducen con el código del repositorio. Ejecuté los experimentos y obtuve números distintos, en algunos casos off en un factor ~2.7×. Además, el experimento insignia entrena sobre ruido y tiene sus propios veredictos escritos a mano en el `print`.

| Veredicto | Estado |
| :--- | :--- |
| Idea / marco teórico | **Sólido** — publicable |
| Mecanismo espectral (medido) | **Confirmado** — efecto grande y reproducible |
| Tasa de bits (0.945 bpp / 33.86×) | **No reproducible** — real 0.354–0.479 bpp |
| Experimento de "falsificación" | **Inválido** — ruido puro, labels hardcodeados |
| Tabla estrella del README | **Contradice** los datos del propio paper |
| Frameworks de regularización / λ | **Roto** — λ no transferible entre shapes |
| Comparación con SOTA | **Sesgada** — faltan 3 baselines críticos, sin semillas |

**Recomendación:** el trabajo **no está listo para ninguna venue**. No porque la idea sea mala — porque las afirmaciones actuales no se sostienen y un revisor las detecta en los primeros 5 minutos. La §7 tiene el plan para arreglarlo; el trabajo es salvable y el resultado honesto sigue siendo interesante.

---

## 2. Reproducibilidad de esta auditoría

Todos los comandos son desde la raíz del repo, sin dependencias de red ni GPU.

```bash
# H1 — tasa de bits real por shape y cutoff
python -c "import torch; from dreg.quantization import Base3TritQuantizer; from dreg import dct2d; \
torch.manual_seed(0); \
[print(f'{m}x{n} r2={r2} bpp={Base3TritQuantizer(r2=r2).quantize_matrix(dct2d(torch.randn(m,n)))[\"bpp\"]:.3f}') \
 for (m,n) in [(256,1024),(256,256),(96,128)] for r2 in (0.40,0.50)]"

# H3 — experimento insignia tal cual está en el repo
python examples/evaluate_falsification.py

# H5 — retención de energía / error de reconstrucción (§5 de este doc)
# ver script en el Anexo B
```

---

## 3. Hallazgos

### 🔴 H1 — La tasa de bits anunciada no se reproduce (error ~2.7×)

**Afirmado:** 0.945 bpp, 33.86× compresión (paper §3.3, abstract, README, ROADMAP).
**Medido:** el código nunca produce 0.945 bpp. Para todas las shapes de la arquitectura del paper:

| shape | r2=0.40 | r2=0.50 (default) |
| :--- | ---: | ---: |
| 256×1024 (FFN) | 0.354 bpp (90.3×) | 0.468 bpp (68.4×) |
| 256×256 (QKV) | 0.357 bpp (89.7×) | 0.470 bpp (68.1×) |
| 1024×256 | 0.354 bpp | 0.468 bpp |
| 4096×256 (embed) | 0.354 bpp | 0.467 bpp |
| 96×128 | 0.365 bpp | 0.479 bpp |
| 96×96 | 0.367 bpp | 0.481 bpp |

**Impacto:** todo el argumento de "sub-1.0 bpp" y el SRAM <1 MB se apoya en una cifra que el código no produce. La banda 3 se descarta entera, así que el bpp **depende principalmente del cutoff y de la shape**, no es una propiedad del método.

**Causa probable:** desajuste entre el texto del paper y los parámetros reales; el 0.945 bpp parece corresponder a otro r2 (ver H2).

---

### 🔴 H2 — El paper se contradice a sí mismo en §3.3 (tres cifras mutuamente inconsistentes)

El §3.3 del paper afirma simultáneamente: bandas `(r0,r1,r2) = (0.10, 0.25, 0.40)` en Algoritmo 1 (línea 243), porcentajes de ocupación `1.7% / 8.4% / 60.2%` (líneas 170–173), y `0.945 bpp` (línea 188).

Geometría real del grid radial (`u=i/m`, `v=j/n`), medido:

| r2 | banda 0 | banda 1 | banda 2 | banda 3 (descartada) | bpp | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.400 *(paper Alg.1)* | 0.81% | 4.15% | 7.70% | **87.33%** | **0.354** | 90.3× |
| 0.500 *(default código)* | 0.81% | 4.15% | 14.78% | 80.25% | **0.468** | 68.4× |
| 0.712 | 0.81% | 4.15% | 35.03% | 60.01% | 0.791 | 40.4× |
| 0.800 | 0.81% | 4.15% | 45.49% | 49.54% | 0.959 | 33.4× |

**Las tres cifras del paper no pueden ser ciertas a la vez:**
- Los porcentajes de §3.3 (b3=60.2%) corresponden a **r2 ≈ 0.71**, no a 0.40.
- El 0.945 bpp corresponde a **r2 ≈ 0.80**, no a 0.40 ni a 0.71.
- Con el r2=0.40 del Algoritmo 1, el resultado es 0.354 bpp y se descarta el **87%** de los coeficientes, no el 60%.

Además, los porcentajes de banda 0 y 1 del paper (1.7% / 8.4%) están ~2× inflados respecto al valor medido (0.81% / 4.15%) para cualquier r2.

**Bug de código relacionado:** `TritQFormat.load()` (quantization.py:253-255) reconstruye las máscaras usando `quantizer.r0/r1/r2` con los defaults, **no** el r2 con el que se cuantizó. Si alguien cuantiza con `r2=0.40` y luego carga, las máscaras no coinciden con los datos empaquetados → reconstrucción corrupta silenciosa.

---

### 🔴 H3 — El experimento insignia ("falsificación") es inválido

`examples/evaluate_falsification.py` es el script que respalda la tabla estrella del README. Al ejecutarlo tal cual:

```
 FALSIFICATION BENCHIFICATION: TOPOGRAPHIC MANOLD VS STANDARD UNORDERED MODEL
================================================================================
[*] Device: cpu

Architecture / Model           | FP32 Baseline PPL  | Trit Q (0.945 bpp) PPL | Status
------------------------------------------------------------------------------------
Standard Model (Unordered)     | 132.62             | 131.64                 | CATASTROPHIC COLLAPSE
Topographic Model (Dirichlet)  | 134.69             | 132.85                 | PRESERVED (STABLE)
================================================================================
Linear Weights Bit Rate: ~0.479 bpp (66.9x compression)
```

Cuatro problemas independientes:

1. **Los datos son ruido puro.** `evaluate_falsification.py:82-84` usa `torch.randint(0, vocab_size, ...)`. No hay texto, solo enteros uniformes. PPL ≈ 132 = `exp(ln 128)` = la entropía de un modelo que no aprendió nada. La capacidad del modelo y la cuantización son irrelevantes cuando la señal es cero.
2. **No hay contraste.** 131.64 vs 132.85 — ambos modelos están igual de destruidos. La tabla del README afirma un abismo (43.43 vs 11.54); el script no lo produce.
3. **Los veredictos están hardcodeados.** Las etiquetas `"CATASTROPHIC COLLAPSE"` y `"PRESERVED (STABLE)"` (líneas 135-136) son `print` literales. El script imprimiría "PRESERVED" aunque ambos modelos colapsaran.
4. **El bpp del header es hardcodeado.** Línea 122 imprime "0.945 bpp, 33.8x" como texto fijo; el valor calculado línea 138 da 0.479 bpp / 66.9×.

**Impacto:** la evidencia central del repositorio es un placeholder que no mide lo que dice medir. Este es el hallazgo más urgente.

---

### 🔴 H4 — La tabla estrella del README contradice los datos del propio paper

README (líneas 134-139):

| Modelo | Bit-Rate | Perplexity | Status |
| :--- | ---: | ---: | :--- |
| Standard (FP32) | 32 bpp | 11.88 | Baseline |
| Standard (Quantized) | 0.945 bpp | **43.43** | ⚠️ Catastrophic Collapse |
| Topographic (FP32) | 32 bpp | 10.80 | — |
| Topographic (Quantized) | 0.945 bpp | **11.54** | 🌟 **Preserved (Δ = +0.74)** |

Paper, Tabla 2 (líneas 326-327):

| Modelo | FP32 Val PPL | Quant PPL (0.945 bpp) |
| :--- | ---: | ---: |
| Standard Baseline (λ=0) | 5.83 | **72.89** |
| Topographic (λ=30) | 6.18 | **66.13** |

Un ΔPPL de +59.96 sobre un FP32 de 6.18 es una degradación de **11×**, no preservación. El mejor resultado del paper sigue siendo un colapso; la regularización solo reduce el daño (72.89 → 66.13).

**El README presenta como "preservación" un resultado que el propio paper describe como colapso.** Además el README dice "con el modelo topográfico 10.80 vs estándar 11.88" mientras el paper §4.5 (Tabla 4) reporta 6.46 vs 6.11 — la dirección se invierte (topográfico *peor* en FP32).

**Otras discrepancias entre documentos:**
- README línea 23: ">90% de energía en bajas frecuencias" · Abstract paper línea 43: "over 72%".
- README línea 23 y tabla: "10×–34× **lossless** compression" · el esquema es lossy (ver H5).
- README línea 141: "the **necessary and sufficient** condition" · no está demostrado.

---

### 🟠 H5 — "Lossless compression" es incorrecto

README línea 119: "**10×–34× lossless compression**". El esquema es **lossy**:

- **Banda 3 se descarta completamente** (`quantization.py` no escribe nada para ρ > r2; `dequantize_matrix` inicializa `rec` a ceros). A r2=0.50 eso es el **80% de los coeficientes** puestos a cero.
- Bandas 0 y 1 usan cuantización afín min/max sobre el rango completo, sin clipping por percentil (`quantization.py:65-66`, `72-73`). Min/max es conocido por ser frágil ante outliers.
- Banda 2 usa ternarios con umbral 0.5σ (`quantization.py:81-82`); para coeficientes tipo Laplace/gaussiano, ~50% de ellos caen dentro de ±0.5σ → se vuelven cero exacto.

Lo que **sí** es lossless es el empaquetado base-3 (`3^5 = 243 ≤ 256`, quantization.py:89-99) — el mapeo byte→trits es biyectivo. Pero eso es solo el formato de almacenamiento de la banda 2, no el esquema completo. La distinción debe quedar explícita.

---

### 🟠 H6 — λ no es transferible entre arquitecturas (bloquea el claim "universal")

`dirichlet_energy_2d` normaliza la energía por el número de elementos (`topology.py:54` y `:58`):

```python
energy = (diff_h.pow(2).sum() + diff_w.pow(2).sum()) / (sheet.numel() + 1e-8)
```

Con esa normalización, λ tiene una escala efectiva **inversamente proporcional al tamaño de la matriz**: λ=30 en una matriz 256×1024 exerts una fuerza muy distinta que λ=30 en 96×96. El propio paper lo admite (§4.2, línea 287): λ ≤ 10⁻² "produced negligible gradient force... 50,000× smaller than cross-entropy".

**Consecuencia:** un hiperparámetro que no se puede transferir entre modelos no puede sustentar el claim "architecture-agnostic" / "universal". El problema es trivial de arreglar (normalizar por el espectro del Laplaciano, o penalización relativa/adaptativa, o λ por-shape), pero mientras no se arregle, el claim central del README es indefendible.

---

### 🟠 H7 — La regularización **sí** cuesta precisión (contradice el caption de Fig. 1)

- §4.2: λ 0→30 sube FP32 PPL 10.48 → 10.98.
- Tabla 2: λ 0→30 sube FP32 PPL 5.83 → **6.18**.
- Tabla 4: FP32 6.11 (Std) vs **6.46** (Topo); INT4 espacial 6.79 vs **7.01**; INT4 espectral 7.85 vs **8.97**.

En tres de las cuatro filas de la Tabla 4, la regularización **empeora** el modelo. Caption de Fig. 1 (paper línea 72): *"forces weights to self-organize into continuous, smooth cortical manifolds **without loss of functional accuracy**"*. Los datos propios lo contradicen.

**Reencuadre honesto (y más fuerte):** el beneficio de `dreg` **está confinado al régimen del cliff sub-2-bit** (INT2 y TritQ), donde todo lo demás colapsa. Eso es una tesis legítima y específica — pero es una tesis distinta de "mejor compresión general".

---

### 🟠 H8 — "Concentración exponencial" es matemáticamente incorrecta

Abstract (paper línea 43): *"forces an **exponential** energy concentration in the low-frequency modes"*. La fórmula correcta (README línea 55) es:

$$\mathbb{E}[|C_{u,v}|^2] \propto \frac{1}{1 + \lambda (u^2 + v^2)}$$

 Penalizar `E[|C_k|²]·μ_k` con `μ_k ∝ k²` produce una **ley de potencias** `1/k²`, no exponencial. Corregir en el abstract y en §5.

Nota menor: §1 dice "the 2D-DCT spectra resemble flat, high-frequency white noise". Correcto — para iid, Parseval da espectro plano (varianza igual en cada coeficiente). El contraste iid-plano ↔ smooth-`1/k²` es correcto y es el argumento válido.

---

### 🟡 H9 — Claims de la tabla de beneficios sin ninguna evidencia

README líneas 117-122, tres filas afirmadas sin un solo experimento en el repo:

| Claim | Estado |
| :--- | :--- |
| **Continual Learning** — "reduced interference", "localized task regions" | **0 experimentos.** No existe prueba de forgetting. |
| **Analog/Neuromorphic HW** — "native compatibility with analog arrays" | **0 experimentos.** Afirmación de hardware sin medición. |
| **Anti-Overfitting** — "regularization without magnitude shrinkage" | Parcial. El punto 7 muestra que sí hay shrinkage efectivo. |

Deben marcarse explícitamente como **hipótesis / trabajo futuro**, no listarse como beneficios demostrados en una tabla comparativa.

---

### 🟡 H10 — Faltan los baselines que un revisor exige, y no hay semillas

**Baselines ausentes (los tres primeros son los que pregunta cualquier revisor de cuantización):**

1. **Incoherencia por Hadamard / rotación aleatoria (estilo QuIP).** Es el control natural: también aplana el espectro y mejora la cuantización vectorial, **sin** imponer suavidad. Si `dreg` no bate a una rotación aleatoria, la contribución no está aislada. Es barato de implementar (una línea) y decisivo.
2. **Pesos parametrizados directamente en dominio DCT** (p. ej. `W = D^T diag(g) D` con `g` aprendible). Si lo que quieres es suavidad, **imponla** en vez de regularizar hacia ella — y elimina λ del diseño. Esta es la pregunta killer del revisor.
3. **Poda de tipo DCT a sparsity matched** (mismo número de coeficientes, sin cuantización de banda) para separar "truncar" de "cuantizar".
4. **VTQ / product quantization / QuIP / QuaRot / SpinQuant / LLM-QAT** en Related Work. Hoy solo se cita QuIP#.

**Sin barras de error:** todas las cifras son de una única corrida, sin semilla replicada. La ventaja estrella ("7.25 PPL") es la diferencia entre **dos corridas**. La varianza de semilla en TinyStories a 10k steps es típicamente de varios PPL — la ventaja medida está dentro (o cerca) del ruido. Esto es lo primero que impugna cualquier revisor.

---

### 🟡 H11 — Otros defectos de ingeniería (menores)

| # | Ubicación | Problema |
| :--- | :--- | :--- |
| 1 | `quantization.py:221` | `eval(cfg_str)` para deserializar config → **ejecución arbitraria** al cargar un `.tritq` no confiable. Usar `json`. |
| 2 | `quantization.py:92` | Padding de trits con `1` (= trit 0). Al desempaquetar se hace `[…][:count]`, así que funciona, pero es frágil si el conteo se recalcula. |
| 3 | `evaluate_falsification.py:64` | El bpp solo cuenta proyecciones 2D; no incluye LayerNorm ni otros tensores. El total del modelo no coincide con la etiqueta. |
| 4 | README línea 155 | 749.7 KB para un modelo de 10M params implica 0.60 bpp, no 0.945. A 0.945 bpp un modelo de 10M ocupa **~1.18 MB**. O el conteo de parámetros o la etiqueta bpp están mal. |
| 5 | `dct_matrix_1d` | Construye la matriz DCT con un bucle Python de `n` iteraciones de coseno (O(n²) trig). Lento; usar `torch.arange` vectorizado o scipy. |

---

## 4. Lo que SÍ es sólido (lo que hay que defender)

Para que conste lo que sobrevive a esta auditoría:

**1. La cadena matemática es correcta y está bien explicada.**
`E_D(W) = ½∫|∇𝒲|²` es un seminorma de Sobolev (H¹) sobre un campo escalar 2D. Su gradiente es el Laplaciano discreto (equación de calor, paper línea 139 ✓). Los autovectores del Laplaciano 1D con BC de Neumann son exactamente la base coseno DCT-II (paper línea 153 ✓). Por Parseval, penalizar `E_D` penaliza los coeficientes de alta frecuencia con severidad cuadrática (paper línea 163 ✓). La descomposición espectral `E_D = (1/MN) Σ μ_{k,l} S_{k,l}²` es exacta.

**2. El mecanismo funciona, y el efecto es grande.** Medición independiente (§5, Anexo B), a budget fijo de 12.7% de coeficientes:

| Matriz 256×1024 | Energía retenida | Error rel. |
| :--- | ---: | ---: |
| Ruido blanco (tipo AdamW) | 9.9% | 94.4% |
| Campo suave (low-freq) | **101.2%** | **15.9%** |

**6× más energía retenida y 6× menos error, al mismo presupuesto.** La tesis central —"la suavidad compra compresibilidad"— es **verdadera y medible**. Este es el resultado publicable.

**3. Es genuinamente barato y agnóstico.** O(n) por matriz, puramente vectorizable, una línea en el bucle de entrenamiento, API de dos líneas (`DirichletLoss` / `TopographicLinear`). La interfacial es buena.

**4. Disciplina de ingeniería.** Suite de tests, CI con matrix Windows+Ubuntu, kernel C portable (Win32 + POSIX), pipeline de Modal reproducible. El §5.1 del paper (el "rank-1 clamping attractor" con ρ=0.996 pero rank effective 199.2 → 3.58) es un **resultado negativo honesto y bien reportado** — aporta valor real a la literatura y es exactamente el tipo de honestidad que sostiene un submission.

**5. La observación de sistemas O(B³) vs O(D²)** (paper §3.4) es un insight genuino y bien argumentado: reconstruir la DCT completa cuesta 512× más que el forward pass de un token; el tiling B×B lo desacopla del ancho D.

---

## 5. Medición del mecanismo (la evidencia buena)

Matriz 256×1024. Cuantizador del repo con sus defaults.

| Matriz | r2 | bpp | Coefs. guardados | Energía retenida | Error rel. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ruido blanco | 0.40 | 0.354 | 12.7% | 9.9% | 94.4% |
| Ruido blanco | 0.50 | 0.468 | 19.7% | 14.3% | 91.3% |
| Suave (low-freq) | 0.40 | 0.354 | 12.7% | **101.2%** | **15.9%** |
| Suave (low-freq) | 0.50 | 0.468 | 19.7% | **101.1%** | **16.3%** |

**Conclusión:** la diferencia es real y de un orden de magnitud. Este es el resultado que hay que defender, y no necesita las cifras de bpp declaradas para sostenerse.

---

## 6. Evaluación de riesgo

| Riesgo | Prob. | Impacto | Mitigación |
| :--- | :--- | :--- | :--- |
| Tesis central refutada por un revisor (falta baseline Hadamard/DCT) | Alta | **Fatal** | H10.1 / H10.2 |
| Cifras de bpp no reproducibles | **Ya mayors** | **Fatal** | H1, H2 |
| Experimento insignia invalidado | **Ya ocurrido** | **Fatal** | H3 |
| Ventaja de 7 PPL dentro del ruido de semilla | Alta | Alto | H10 (n ≥ 3 semillas) |
| λ no transferible → claim universal cae | **Ya mayores** | Medio | H6 |
| Related Work incompleto | Media | Medio | H10.4 |
| Overselling del método (revisión de escritorio) | **Ya ocurrido** | Medio | H4, H5, H7, H9 |

---

## 7. Plan de corrección priorizado

### P0 — Bloqueantes (hacer sí o sí antes de nada más)

- [ ] **P0.1** Arreglar o retirar las cifras de bpp. Elegir un r2, propagarlo a `quantization.py`, Algoritmo 1, §3.3, abstract, README y ROADMAP, y que el número salga de un script. (~2 h)
- [ ] **P0.2** Arreglar `TritQFormat.load()` para que respete el r2 con el que se guardó (serializarlo en el header). (~1 h) — es corrupción silenciosa.
- [ ] **P0.3** Arreglar o reemplazar `evaluate_falsification.py`: datos reales, sin labels hardcodeados, bpp calculado. Usar TinyStories de verdad. (~medio día)
- [ ] **P0.4** Corregir la tabla estrella del README para que refleje los números reales (colapso vs colapso), o eliminar la sección de "falsificación" hasta tenerla. (~1 h)
- [ ] **P0.5** Corregir "lossless" → "lossy, with exact lossless base-3 bit-packing"; corregir "exponencial" → power law; borrar las filas CL/analog o marcarlas como hipótesis; arreglar ">90%" vs "72%". (~1 h)

### P1 — Necesario para una venue respectable

- [ ] **P1.1** Normalizar λ independiente de la shape (espectral, o por-matrix adaptativa) y demostrar transferencia entre TinyStories y MNIST con el mismo λ. Esto convierte el claim "universal" en defendible. (~1 día)
- [ ] **P1.2** Implementar los 3 baselines: (a) Hadamard/rotación aleatoria, (b) parametrización en dominio DCT, (c) poda a sparsity matched. (~1-2 días)
- [ ] **P1.3** Re-ejecutar con n ≥ 3 semillas y barras de error. (~1 día de GPU)
- [ ] **P1.4** Reencuadrar el paper honestamente: el beneficio está confinado al cliff sub-2-bit. Reescribir abstract y §4.5 con esa tesis. (~medio día)
- [ ] **P1.5** Ampliar Related Work: QuIP, QuaRot, SpinQuant, LLM-QAT, VQ/LQ-VQ, literatura de compresión DCT de redes. Suavizar "has remained unexplored". (~medio día)
- [ ] **P1.6** Quitar `eval()` en `quantization.py:221` → `json`. (~15 min)

### P2 — Pulido

- [ ] **P2.1** Reconciliar el conteo de parámetros / tamaño total (749.7 KB vs 1.18 MB implícito).
- [ ] **P2.2** Vectorizar `dct_matrix_1d` (O(n²) trig en bucle Python → arange).
- [ ] **P2.3** Ablations del ROADMAP §2.3 (tile size B, profundidad, ancho) — pendientes desde Fase 2.
- [ ] **P2.4** Script de reproducibilidad end-to-end que regenere todas las tablas del paper.

---

## 8. Veredicto final

**La idea merece un paper. El paper actual no.** La separación entre estas dos cosas es lo que hay que cerrar.

Lo que hay que preservar: la intuición (romper la simetría de gauge de permutación imponiendo continuidad espacial), la demostración de que el mecanismo funciona (medible, 6×), la honestidad del resultado negativo de §5.1, y la interfaz de dos líneas.

Lo que hay que rehacer: todas las cifras, y el marco de la contribución. La versión publicable de este trabajo es:

> *"La suavidad espacial no es una mejora marginal de la compresión — es lo que separa la degradación graceful del colapso en el régimen sub-2-bit. Demostrado con n semillas, contra Hadamard, contra parametrización DCT."*

Eso es una contribución real, sustentable y con evidencia barata de producir. La versión actual ("preservamos calidad a 0.945 bpp, 33.86×") no lo es, y basta con que alguien ejecute el script de §3 para verlo.

---

## Anexo A — Referencias de código

| Hallazgo | Ubicación |
| :--- | :--- |
| Normalización por `numel` → λ no transferible | `dreg/topology.py:54`, `dreg/topology.py:58` |
| `r2=0.50` por defecto (paper dice 0.40) | `dreg/quantization.py:23` |
| `load()` ignora el r2 de guardado | `dreg/quantization.py:253-255` |
| Banda 3 descartada (lossy) | `dreg/quantization.py:101-110`, `:112-145` |
| Min/max sin clipping de outliers | `dreg/quantization.py:65-66`, `:72-73` |
| Ternarios con umbral 0.5σ | `dreg/quantization.py:81-82` |
| Empaquetado base-3 (esto sí lossless) | `dreg/quantization.py:89-99` |
| `eval()` en deserialización | `dreg/quantization.py:221` |
| Datos de ruido `randint` | `examples/evaluate_falsification.py:82-84` |
| bpp del header hardcodeado | `examples/evaluate_falsification.py:122` |
| Veredictos hardcodeados | `examples/evaluate_falsification.py:135-136` |
| Tabla estrella del README | `README.md:134-139` |
| Claim "lossless" | `README.md:119` |
| Claims CL / analog HW sin evidencia | `README.md:121-122` |
| "necessary and sufficient" | `README.md:141` |
| ">90%" vs "72%" | `README.md:23` vs `paper/paper-draft.tex:43` |
| "exponential" (incorrecto) | `paper/paper-draft.tex:43` |
| Bandas §3.3 (1.7/8.4/60.2%) | `paper/paper-draft.tex:170-173` |
| r2=0.40 en Algoritmo 1 | `paper/paper-draft.tex:243` |
| Caption "without loss of functional accuracy" | `paper/paper-draft.tex:72` |
| Tabla 2: colapso real | `paper/paper-draft.tex:326-327` |
| "negligible gradient force" | `paper/paper-draft.tex:287` |
| λ no decreciente / coste en FP32 | `docs/ROADMAP.md:73-81` |

---

## Anexo B — Script de medición del mecanismo

```python
import torch
from dreg.quantization import Base3TritQuantizer
from dreg import dct2d, idct2d

torch.manual_seed(0)
m, n = 256, 1024

# (a) ruido blanco — proxy de pesos estándar de AdamW
W_noise = torch.randn(m, n) * 0.02

# (b) campo suave de baja frecuencia — ground truth ideal
u = torch.arange(m).view(-1, 1).float() / m
v = torch.arange(n).view(1, -1).float() / n
rho = torch.sqrt(u**2 + v**2)
S = torch.exp(-(rho**2) / (2 * 0.12**2)) * torch.randn(m, n)
W_smooth = idct2d(S)

for name, W in [("ruido blanco", W_noise), ("suave", W_smooth)]:
    for r2 in (0.40, 0.50):
        q = Base3TritQuantizer(r2=r2)
        spec = dct2d(W)
        packed = q.quantize_matrix(spec)
        rec_spec = q.dequantize_matrix(packed)
        rec = idct2d(rec_spec)
        rel = (rec - W).norm().item() / W.norm().item()
        kept = (q.compute_radial_grid(m, n, W.device) <= r2).float().mean().item()
        energy = ((rec_spec**2).sum() / (spec**2).sum()).item()
        print(f"{name:12s} r2={r2}  bpp={packed['bpp']:.3f}  "
              f"coefs={kept*100:.1f}%  energy={energy*100:.1f}%  err={rel*100:.1f}%")
```