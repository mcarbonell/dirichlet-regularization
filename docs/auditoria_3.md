# Auditoría 3 — Dirichlet Regularization (`dreg`)

**Fecha:** 2026-10-01
**Alcance:** revisión completa del repo `dirichlet-regularization` en el commit `284f499` (`main`): código (`dreg/`, `kernel/`, `examples/`, `tests/`, `benchmarks/`), documentación (`README.md`, `docs/whitepaper.md`, `docs/ROADMAP.md`, `docs/private/`, `docs/attention-neuron/`), material de publicación (`paper/paper-draft.tex`, `paper/paper-draft.pdf`, `references.bib`) y config (CI, `pyproject.toml`, `.gitignore`).
**Método:** lectura del código + **ejecución directa** de los tests, del experimento insignia, del benchmark del kernel C y de mediciones independientes de las métricas afirmadas.
**Entorno:** Windows, Python 3.14, `torch 2.10.0+cpu`, `numpy`, `pytest 9`. Sin GPU, sin red.

> Esta auditoría **no repite** las anteriores (`docs/auditoria_1.md`, `docs/auditoria_2.md`); las usa como base, **re-verifica** sus hallazgos clave y **añade hallazgos nuevos** (marcados como `🆕`) que solo aparecen al ejecutar artefactos concretos.

---

## 1. Resumen ejecutivo

**La idea sigue siendo buena y el código es limpio y testeado.** Los 57 tests pasan (`57 passed in 1.84s`), el CI está bien montado, el `.gitignore` es correcto (37 archivos rastreados, sin basura ni datasets en git) y el núcleo matemático (energía de Dirichlet → Laplaciano → base DCT-II → compactación espectral) es texto de libro y está bien planteado.

**El problema persiste y ahora es más claro: las cifras insignia no provienen del código del repositorio.** Al ejecutar los artefactos directamente:

| Artefacto ejecutado | Resultado medido | Resultado afirmado |
| :--- | :--- | :--- |
| `tests/` | ✅ `57 passed` | ✅ "57 passed" |
| bpp del cuantizador (256×1024, default) | **0.468 bpp** | ❌ "0.945 bpp" |
| Fracciones de banda reales (256×256) | 0.83 / 4.18 / 14.81 / 80.18 % | ❌ "1.7 / 8.4 / 29.7 / 60.2 %" |
| `examples/evaluate_falsification.py` | PPL ≈ 132/133, veredicto **invertido**, bpp 0.479 | ❌ "preserva 11.54 vs colapso 43.43" |
| `examples/benchmark_c_dma.py` | discrepancia C↔PyTorch = **0.7867** | ❌ "bit-exact (0.00000000)" |

**Veredicto:** el trabajo **no está listo para submission**. No porque la idea sea débil, sino porque (a) los números del paper no se reproducen con el código, (b) **las tablas del propio paper se contradicen entre sí** (§4), (c) el experimento "insignia" mide ruido aleatorio y emite un veredicto hardcodeado, y (d) el claim de equivalencia exacta del kernel C es falso. Un revisor detecta (a)–(c) en minutos.

| Eje | Estado |
| :--- | :--- |
| Idea / marco teórico | ✅ Sólido, publicable |
| Mecanismo espectral | ✅ Real (confirmado en auditoría 1) |
| **Tasa de bits (0.945 bpp / 33.86×)** | ❌ **No proviene del código** (real ≈ 0.47 bpp) |
| **Coherencia interna del paper** | ❌ **Tablas y abstract se contradicen** (`🆕`) |
| **Experimento de falsificación** | ❌ Ruido aleatorio + veredictos hardcodeados |
| **Kernel C "bit-exacto"** | ❌ **Falso** (discrepancia 0.79) (`🆕`) |
| **Reproducibilidad de TinyStories/MNIST** | ❌ Deps no declaradas, sin checkpoints (`🆕`) |
| Calidad de código | ✅ 8/10 (limpio, 57 tests, CI) |
| Higiene de repo | ✅ Correcta (`.gitignore`, 37 archivos) |


---

## 2. Qué ejecuté y qué obtuve (reproducibilidad de esta auditoría)

Todos los comandos, desde la raíz del repo:

```bash
# 1) Suite de tests
python -m pytest tests/ -q
#   -> 57 passed in 1.84s

# 2) bpp real por shape y cutoff (compare con la cifra "0.945" del paper)
python -c "import torch; from dreg.quantization import Base3TritQuantizer; from dreg import dct2d; \
torch.manual_seed(0); \
[print(f'{m}x{n} r2={r2} bpp={Base3TritQuantizer(r2=r2).quantize_matrix(dct2d(torch.randn(m,n)))[\"bpp\"]:.4f}') \
 for (m,n) in [(256,1024),(256,256),(96,128)] for r2 in (0.40,0.50)]"
#   256x1024 r2=0.4 bpp=0.3542   |   r2=0.5 bpp=0.4676
#   256x256  r2=0.4 bpp=0.3567   |   r2=0.5 bpp=0.4702
#   96x128   r2=0.4 bpp=0.3652   |   r2=0.5 bpp=0.4785

# 3) Experimento insignia tal cual está en el repo
python examples/evaluate_falsification.py
#   Standard   FP32 132.62 -> 131.64   (¿"colapso"?)
#   Topographic FP32 134.69 -> 132.85  (¿"preservado"?)
#   bpp ~0.479 (66.9x), NO 0.945/33.8x

# 4) Benchmark del kernel C (requiere compilar la DLL)
python examples/benchmark_c_dma.py
#   Maximum Absolute Numerical Discrepancy: 0.78671384   (¡no 0.00000000!)
#   Single Matrix Decode Latency: 188.24 µs
#   Transformer L=6 Equivalent: 147.6 tokens/second

# 5) Fracciones de banda reales del cuantizador (256x256)
#   b0=0.83%  b1=4.18%  b2=14.81%  b3=80.18%   -> bpp=0.4702
#   Teórico con fracciones del whitepaper (1.7/8.4/29.7): 0.9472 bpp
```

---

## 3. Hallazgos nuevos de esta auditoría (`🆕`)

### 🆕 N1 — El kernel C **no** es "bit-exacto" (claim falso)

**Afirmado** (`examples/benchmark_c_dma.py:5-6`, `docs/whitepaper.md:187-188`, abstract del paper): "equivalencia matemática exacta (0.00000000)" / "bit-exact mathematical match".
**Medido:** `Maximum Absolute Numerical Discrepancy: 0.78671384`. El propio script imprime el mensaje de reserva *"Note: Small numerical precision variance"* en lugar de *"VERIFICATION PASSED"*.

**Causa raíz (bug real, no ruido de float):** el kernel C reconstruye cada banda de forma **simétrica respecto a un offset fijo**, mientras que Python usa **escala afín con `min`/`scale`**:

| Banda | Python (`dreg/quantization.py`) | C (`kernel/spectral_dma_kernel.c`) |
| :--- | :--- | :--- |
| Band 0 | `vals = q0*scale0 + min0` | `v = (b0 - 128) * (s0/127)` (`:135`) |
| Band 1 | `vals = q1*scale1 + min1` | `v = (nib - 7) * (s1/7)` (`:151`) |

Ambas fórmulas solo coinciden si `min0 = -128·scale0` y `min1 = -7·scale1`, i.e. si cada banda es exactamente simétrica respecto a cero — **lo que no se cumple en general**. Los offsets `min/scale` reales se calculan con `vals.min()`/`vals.max()` (`dreg/quantization.py:65-66,71-72`) y no son simétricos. De ahí la discrepancia.

**Impacto:** invalida el claim de "identidad matemática exacta" del abstract/whitepaper y debilita el argumento de que el formato `.tritq` y su decodificador C son intercambiables. **Fix barato:** que el C aplique `q*scale + min` (pasar también `min0`, `min1`) o que Python cuantice simétricamente en band0/band1. Elegir uno y verificar con tolerancia `< 1e-6`.

### 🆕 N2 — El "0.945 bpp" es un número teórico de fracciones que el código no produce

El whitepaper (`docs/whitepaper.md:105-108`) asume:

```
Banda 0 (ρ<=0.10): 1.7% coefs × 8 bpp
Banda 1 (0.10<ρ<=0.25): 8.4% coefs × 4 bpp
Banda 2 (0.25<ρ<=0.50): 29.7% coefs × 1.6 bpp
Banda 3 (ρ>0.50): 60.2% coefs × 0 bpp
=> 0.017*8 + 0.084*4 + 0.297*1.6 = 0.947 bpp
```

Ese `0.947 ≈ 0.945 bpp` es el origen de la cifra estrella. **Pero el `compute_radial_grid` real del código** (`dreg/quantization.py:44-50`) produce, para las MISMAS bandas `(0.10, 0.25, 0.50)`:

```
b0 = 0.83%   b1 = 4.18%   b2 = 14.81%   b3 = 80.18%   -> bpp real = 0.4702
```

Las fracciones del whitepaper no se corresponden ni con la malla radial del código ni con sus propios radios declarados (`r0=0.10, r1=0.25, r2=0.50`). Es decir: **la cifra del paper viene de una malla/definición distinta a la del código publicado.** Paradójicamente, el número real (0.47 bpp, 68×) es **mejor** que el afirmado (0.945 bpp, 33.86×): el repo se vende peor de lo que su código logra, pero con un número que no se puede reproducir.

### 🆕 N3 — El experimento insignia no reproduce el v392 y su veredicto es engañoso

`examples/evaluate_falsification.py` afirma reproducir "the critical falsification experiment (v392)" (`:5-7`). En realidad:

1. **Entrena sobre ruido aleatorio:** `train_data = torch.randint(0, vocab_size, ...)` (`:82-84`). No hay lenguaje. La PPL de ambos modelos queda ≈ el tamaño del vocabulario (132 ≈ 128), i.e. azar puro.
2. **Los veredictos están hardcodeados:** `"CATASTROPHIC COLLAPSE"` / `"PRESERVED (STABLE)"` (`:135-136`) se imprimen como literales, sin relación con los números.
3. **Al ejecutarlo, el resultado contradice el veredicto:** el modelo "Topographic" quedó **peor** que el "Standard" (132.85 vs 131.64) y aun así imprimió "PRESERVED / CATASTROPHIC COLLAPSE".

**Impacto:** el experimento que sostiene la tesis ("la suavidad es la condición necesaria") no es ejecutable como prueba desde el repo; solo existe como número en README/whitepaper (11.88 → 43.43), sin checkpoint ni dataset. Es exactamente lo que un revisor re-ejecuta.

### 🆕 N4 — Las tablas del propio paper se contradicen entre sí

Los mismos modelos (TinyStories 10M) aparecen con números distintos en el abstract, la Tabla 10k, la Tabla SOTA, el README, la whitepaper y el ROADMAP.

| Magnitud | Abstract | Tabla 10k (`:326-327`) | Tabla SOTA (`:379-383`) | README (`:136-139`) | ROADMAP (`:74-78`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Std FP32 PPL | — | 5.83 | 6.11 | 11.88 | 10.48 |
| Topo FP32 PPL | — | 6.18 | 6.46 | 10.80 | 10.62–10.98 |
| Quant PPL (Std) | — | 72.89 | 73.75 | **43.43** | 46.60 |
| Quant PPL (Topo) | — | 66.13 | 66.50 | **11.54** | 39.80 |
| Ventaja ΔPPL | **7.25** (`:45`) | **6.76** (`:333`) | **−7.25** (`:383`) | +0.74 (¡nats vs PPL!) | — |
| Compresión | 33.86× | — | — | 33.9× | — |
| Baja frec. ("lowest") | "72%" (`:43`) | — | — | ">90%" (`:23`) | — |

Dos problemas independientes: (a) los PPL del modelo **entrenado v392** (11.88/10.80/43.43/11.54) y los del modelo **TinyStories 10k** (5.83/…/72.89) son experimentos distintos presentados como si fueran el mismo; y (b) dentro del propio abstract, "7.25 perplexity advantage" (Tabla SOTA) convive con "6.76" (Tabla 10k) y con "0.74 nats" (README). Nada de esto es consistente.

### 🆕 N5 — Los números de TinyStories y MNIST no son reproducibles con las dependencias declaradas

- `requirements.txt` y `pyproject.toml` solo declaran `torch` + `numpy`.
- Los scripts que producen los números del paper (`benchmarks/modal_train_tinystories.py`, `modal_benchmark_quantization_sota.py`, `modal_generate_figures.py`) requieren `modal`, `datasets`, `tokenizers`, `transformers`, `tqdm` (`modal_train_tinystories.py:42-51`) — **ninguno declarado**.
- `examples/train_mlp_dirichlet.py --dataset mnist` requiere `torchvision` (no declarado, `:57`); sin él cae en silencio al dataset sintético.
- No hay **ningún** checkpoint (`.pt`/`.tritq`), log de entrenamiento ni tokenizer en el repo (correctamente ignorados por `.gitignore`, pero eso deja los PPL sin artefacto verificable).

**Impacto:** las métricas centrales (TinyStories PPL, MNIST accuracy) no se pueden recomputar desde el repo tal como está publicado, ni offline ni en la nube, sin reconstruir el entorno a mano.

### 🆕 N6 — Discrepancia de rendimiento del kernel C y unidades del benchmark

El whitepaper afirma latencia de **10.15 µs** y **77.4 tok/s (L=6)** (`docs/whitepaper.md:31`). La ejecución real de `examples/benchmark_c_dma.py` da **188.24 µs** y **147.6 tok/s**. Son mediciones de escritorio, no de silicio Cortex-M (como ya advertía la auditoría 2), pero la latencia difiere ~18× del número publicado. Además, `tokens_per_sec = matrices_per_sec / 36.0` (`:152`) asume un "36 matrices por token en L=6" hardcodeado y no verificado para la arquitectura real.


---

## 4. Re-verificación de hallazgos previos (auditorías 1 y 2)

| Hallazgo previo | ¿Sigue presente? | Evidencia en esta corrida |
| :--- | :--- | :--- |
| bpp 0.945 no reproducible | ✅ Sí, sigue | medido 0.354–0.479 bpp (256×1024/256×256/96×128) |
| `eval()` en `TritQFormat.load` | ✅ Sí | `dreg/quantization.py:221` (`eval(cfg_str)`) |
| Normalización por `numel` → λ no transferible | ✅ Sí | `dreg/topology.py:54,58` (`/(sheet.numel()+1e-8)`) |
| λ por defecto inconsistente entre docs | ✅ Sí | config `0.01` (`model.py:28`) vs paper `30.0` vs whitepaper `2e-3` |
| Banda 3 descartada (lossy, >80% coefs) | ✅ Sí | fracción real 80.18% de coeficientes a 0 bits |
| Caché DCT global mutable | ✅ Sí | `dreg/spectral.py:13` (`_DCT_CACHE` dict global) |
| `get_grid_dimensions` sin uso en el modelo | ✅ Sí | solo se usa en `tests/test_topology.py` |
| "lossless" en README | ✅ Sí | `README.md:119` "10×–34× lossless compression" |
| "necessary and sufficient" | ✅ Sí | `README.md:141` |
| "exponential" (incorrecto: es potencia) | ✅ Sí | `README.md:55`, `whitepaper.md:75` (`1/(1+λ(u²+v²))` es ley de potencias) |
| ">90%" vs "72%" | ✅ Sí | `README.md:23` vs `paper-draft.tex:43` |
| Sin semillas / baselines SOTA reales | ✅ Sí | el experimento de falsificación no fija semilla del run de paper |

**Nota sobre una afirmación de la auditoría 1:** dice que el real es "0.354–0.479 bpp" y lo presenta como problema. Técnicamente **0.47 bpp es mejor** que 0.945 bpp; el problema real (confirmado aquí, N2) es que **el 0.945 no deriva del código**: viene de fracciones de banda que el cuantizador no produce. Conviene reformular el hallazgo con esa causa.

---

## 5. Tabla maestra de inconsistencias documentales

| Afirmación | Dónde se dice | Contradicción | Gravedad |
| :--- | :--- | :--- | :---: |
| Tasa de bits = **0.945 bpp** | README:137-139,145; paper:44,188,265; whitepaper:30 | Código da 0.354–0.479 bpp | 🔴 |
| Compresión = **33.86× / 34.37× / 34× / 33.9×** | paper:188; whitepaper:30; ROADMAP:130; README:139 | Cuatro valores distintos; código da ~68× | 🔴 |
| Rango = **0.931–0.945 bpp** | whitepaper:30 | Ni el 0.931 ni el 0.945 salen del código | 🔴 |
| Energía baja frec. = **">90%"** | README:23, whitepaper:29 | paper:43 dice "72%"; experimental ">72%" (ROADMAP:154) | 🟠 |
| Ventaja PPL = **7.25** | paper abstract:45, SOTA:383 | Tabla 10k:333 dice 6.76; README dice +0.74 | 🔴 |
| Decaimiento **"exponential"** | README:55, whitepaper:75 | La fórmula `1/(1+λ(u²+v²))` es ley de potencias | 🟠 |
| Kernel C **"bit-exact (0.00000000)"** | benchmark docstring:5-6; whitepaper:187-188 | Medido 0.7867 | 🔴 |
| Latencia **10.15 µs** | whitepaper:31 | Medido 188 µs | 🟠 |
| Compresión **"lossless"** | README:119 | Band 3 (80% coefs) se descarta → lossy | 🔴 |
| "necessary and sufficient" | README:141 | No demostrado; el experimento que lo probaría mide ruido | 🟠 |
| "72.1% roughness reduction" | paper:333 | En README/MNIST: "84.5%"; son experimentos distintos | 🟡 |
| Beneficios CL / Analog HW | README:121-122 | Sin ningún experimento que los respalde | 🟠 |


---

## 6. Auditoría de código, módulo por módulo

### `dreg/topology.py` — ✅ correcto, ⚠️ convención de λ
- `dirichlet_energy_2d` implementa `0.5·Σ‖Δw‖²` normalizado por `numel` (líneas 54/58). Es correcto y diferenciable (test `test_gradient_flows`).
- **Problema de escalado:** normalizar por `numel = out·in` hace que el valor bruto sea `~0.004` y **dependa de la shape**, por lo que un mismo λ no es transferible entre capas (el paper reconoce esto en `:287`). Sería más limpio normalizar por número de aristas o exponer un λ "por-arista".
- `DirichletLoss.forward` maneja bien el caso "sin módulos 2D" (devuelve 0.0), corrigiendo el bug previo de la auditoría 1.

### `dreg/spectral.py` — ✅ correcto
- DCT-II ortonormal bien construida; los tests verifican ortogonalidad, roundtrip, Parseval y DC. Todo pasa.
- `_DCT_CACHE` global (línea 13): en multiproceso re-`import` no comparte, pero en hilos/GPU+CPU mixto podría devolver un tensor cacheado con dtype/device inesperado. Bajo riesgo, pero conviene `functools.lru_cache` con claves explícitas o un lock.
- `BlockDCTTiler` correcto (roundtrip exacto con y sin padding).

### `dreg/quantization.py` — ⚠️ correcto en ejecución, incoherente con los claims
- El empaquetado base-3 (`3^5 = 243 ≤ 256`) es **realmente lossless** para los trits (líneas 89-99): `t0 + 3·t1 + 9·t2 + 27·t3 + 81·t4 ∈ [0,242]`. Ese es el único "lossless" legítimo del proyecto, y se refiere al packing, no a la cuantización.
- La cuantización **global es lossy**: band 3 (80% de coefs) → 0, band 2 → ternario con umbral `0.5σ` (línea 82), band 0/1 → uint8/nibble afín. El docstring del módulo (línea 6) sigue anunciando "0.945 bpp / 33.86x", que no coincide con su propia salida.
- `eval(cfg_str)` en `load` (línea 221): vulnerabilidad de ejecución de código arbitrario si el `.tritq` viene de terceros. Fix trivial: `ast.literal_eval`.
- `load` reconstruye las máscaras con radios **por defecto** (`Base3TritQuantizer()`, línea 210) en lugar de leer los radios guardados. Hoy `save` también usa el default, así que no hay bug activo, pero es frágil (no hay checksum ni versión por-tensor).
- Band 2 padding usa código `1` (→ trit 0.0) para alinear a 5 (línea 92); correcto.

### `dreg/model.py` — ✅ correcto, ⚠️ "cortical lattice" es cosmético
- Transformer causal con SwiGLU, máscara causal implementada bien (tests `test_causal_masking`).
- **Observación:** `TopographicLinear.grid_shape = (out, in)` (línea 40) y `get_grid_dimensions` **nunca se usan** para reflowar los pesos a una malla cuadrada. La "retícula cortical 2D" es, en la práctica, tratar la matriz `(out, in)` como una imagen. El README/paper sugieren un embedding topográfico que el código no ejecuta.
- `topographic_loss` promedia por número de capas y multiplica por `topo_lambda` (línea 154): un solo λ para todas las capas; coherente con lo que dice el paper, pero combinado con la normalización por `numel` de topology.py, el efecto por-capa es desigual.

### `examples/evaluate_falsification.py` — 🔴 no es una prueba
Ver N3. Ruido + veredictos hardcodeados. Es el peor artefacto del repo porque *aparenta* ser el experimento decisivo.

### `examples/benchmark_c_dma.py` — 🔴 claim falso
Ver N1. Además, la comparación Python↔C usa `Base3TritQuantizer()` default pero pasa escalas re-ajustadas a mano (`band0.scale*127`, `band1.scale*7`), lo que enmascara el desajuste afín/simétrico.

### `examples/train_mlp_dirichlet.py` — 🟠 correcto pero no cierra el bucle
Entrena bien, pero (a) MNIST requiere `torchvision` no declarado, (b) el resultado final solo imprime una frase condicional si `trunc_drop_topo < trunc_drop_std` (línea 329), sin asserts ni guardado de métricas, y (c) los números del paper (97.87% → 41.41%) no se regeneran sin el dataset real.

### `kernel/spectral_dma_kernel.c` — ✅ ingeniería sólida, ❌ no coincide con Python
- LUT O(1), GEMM i-k-j, worker thread con eventos Win32 **y** rama POSIX (`pthread`) bien resuelta. Buena ingeniería.
- **Bug de paridad** (N1): band0 usa `(b0-128)·s` y band1 `(nib-7)·s` en lugar de la transformación afín con `min` de Python. Es la causa del 0.7867.
- `MAX_SUBLAYERS=64`, `MAX_MATRICES_PER_SUBLAYER=4`: límites silenciosos; `c_spectral_register_matrix` devuelve `-1/-2` pero el benchmark no los comprueba.
- El build multiplataforma (`build_kernel.py`) funciona (compiló la DLL en esta máquina).

### `benchmarks/modal_*.py` — ⚠️ dependencias externas no declaradas
Ver N5. Son los scripts que generan las tablas del paper, pero están acoplados a Modal + HuggingFace + un volumen `dreg-checkpoints` externo. Sin el volumen, `run_quantization_benchmark` falla al cargar checkpoints (`modal_benchmark_quantization_sota.py:103-104`).

### `tests/` — ✅ bien
57 tests, cobertura razonable de topología/espectral/cuantización/modelo. **Gap:** no hay test de paridad C↔Python (que es justo lo que falla), ni test de `TritQFormat` con radios no-default, ni de `ast.literal_eval`.


---

## 7. Auditoría de documentación

### `README.md`
- **Fuerte:** narrativa clara, diagrama intuitivo (ruido → superficie suave), API de 2 líneas atractiva, estructura de repo correcta.
- **Débil / riesgoso:**
  - Dos tablas de resultados sin procedencia ni semillas (`:134-139`), con números que no se reproducen.
  - `"lossless compression"` (`:119`), `"necessary and sufficient"` (`:141`), beneficios CL/analógica sin evidencia (`:120-122`).
  - Badge de tests **estático** (`tests-57_passed`, `:9`): no se actualiza solo; si un test roto pasa desapercibido, el badge miente.
  - `">90%"` (`:23`) contradice el `"72%"` del paper.
  - La sección "Edge Inference Results" (`:149-154`) presenta tok/s y SRAM como si fueran mediciones de silicio, cuando son emulaciones en CPU.

### `docs/whitepaper.md`
- **Fuerte:** la "Sección de Reconciliación" (documentar hipótesis refutadas) es inusualmente honesta y valiosa; las tablas `v382`–`v395` son exhaustivas.
- **Débil:** es, a la vez, la fuente de las **fracciones de banda irreproducibles** (`:105-108`) y del `0.931–0.945`. Repite `">90%"` y usa `"sin pérdida matemática alguna"` de forma ambigua (mezcla "el packing es sin pérdida" con "el códec no tiene pérdida"). Como documento suplementario del paper, arrastra los mismos errores de cifras.

### `docs/ROADMAP.md`
- **Fuerte:** roadmap honesto con números reales de calibración (`:74-78`), fases claras, esfuerzo estimado.
- **Débil:** marca `[x]` "Entrenamiento largo de publicación ✅" (`:79`) y "Fase 3 paper ✅" aunque **no hay artefactos** (checkpoints/logs) en el repo que respalden que se ejecutó. También usa `"λ no decreciente / coste en FP32"` sin resolver, y el "0.945 bpp" aparece como hecho (`:130`).

### `docs/private/` y `docs/attention-neuron/`
- `docs/private/repo_analysis.md` y `publication_roadmap.md` son el análisis fundacional (pre-rename, cuando el paquete era `topospec` y el repo `attention-neuron`). **Contradicen** el estado actual (p. ej. afirman "no hay tests" y "solo Windows", ya resueltos) — es esperable por ser históricos, pero conviene marcarlos como tales.
- `docs/attention-neuron/` es un **archivo** de 17 prototipos (`prototype_v379…v395.py`) + whitepaper + figuras que apuntan al repo `mcarbonell/attention-neuron`. Está correctamente en `.gitignore` (no versionado), pero su presencia local mezcla dos identidades de proyecto. `docs/private/` también está ignorado. **Buena decisión** no versionarlos para el paper; riesgo solo de confusión local.

---

## 8. Auditoría del paper (`paper/paper-draft.tex`)

- **Fuerte:** estructura completa y correcta (Intro, Related Work, Method, Experiments, Discussion, Conclusion); 15 referencias BibTeX bien formateadas y 100% citadas; derivación Dirichlet→Sobolev→Laplaciano→DCT-II correcta; Algoritmo 1 formaliza bien el pipeline; figuras integradas.
- **Problemas de contenido (además de la inconsistencia numérica de §5):**
  1. **Abstract** (`:41-48`): la cifra "0.945 bpp (33.86×)" y la ventaja "7.25" no se sostienen (§5). El `"72%"` del abstract convive con el `">90%"` de otros docs.
  2. **`"exponential energy concentration"`** (`:43`): incorrecto — la fórmula es ley de potencias.
  3. **Algoritmo 1** usa `r2=0.40` (`:243`) mientras el código y el whitepaper usan `r2=0.50`. Discrepancia de configuración entre paper y código.
  4. **Sección 5.1** ("Rank-1 Clamping Attractor", `:399-402`): resultado negativo interesante, pero **sin código ni datos** en el repo que lo respalde.
  5. **Comparación SOTA** (`:389-392`): compara contra INT2/INT4 uniformes (baselines débiles) y **nombra** GPTQ/AWQ/QuIP# sin cuantizarlos realmente; afirma "4.23× more memory" y "the only method to achieve sub-1.0 bpp" sin haber corrido esos métodos. Un revisor lo leerá como comparación no ejecutada.
  6. **Claim de existencia única** ("the only method…", `:392`) y `"lossless byte-packing"` en abstract: el primero es fuerte y no verificado; el segundo es cierto solo del packing.
  7. **Escala:** TinyStories 10M y MNIST-MLP, sin n-semillas, sin curvas PPL-vs-bpp completas. La auditoría 2 ya avisó; sigue igual.
- **Verificable:** `paper-draft.pdf` compila (12 págs, según ROADMAP) y las figuras existen.

---

## 9. Auditoría del ROADMAP (coherencia fase↔realidad)

| Fase | Marcado | ¿Respaldado por el repo? |
| :--- | :--- | :--- |
| 0 Rename/refocus | ✅ | ✅ (paquete `dreg`, imports actualizados) |
| 1.1 Tests | ✅ | ✅ (57 tests pasan) |
| 1.2 `TritQFormat.load()` | ✅ | ⚠️ existe, pero usa `eval()` y radios default |
| 1.3 Bugs corregidos | ✅ | ✅ verificado (`Tuple`, `DirichletLoss`) |
| 1.4 Kernel POSIX | ✅ | ✅ rama POSIX presente en el `.c` |
| 1.5 CI | ✅ | ✅ `.github/workflows/ci.yml` |
| 2.1 Entrenamiento TinyStories | ✅ | ❌ sin checkpoints/logs versionados ni deps declaradas |
| 3.x Paper + figuras | ✅ | ⚠️ existe el `.tex`/`.pdf`, pero con cifras inconsistentes |
| 4 Polish/repro script | ❌ | ❌ (correctamente pendiente) |

**Conclusión:** la Fase 1 es la única 100% verificable. La Fase 2 declara como "hecho" resultados que no tienen artefacto en el repo.


---

## 10. Higiene del repositorio y CI (buenas noticias)

- **`.gitignore` correcto:** excluye `__pycache__`, `*.pyc`, `*.dll/*.so`, checkpoints (`*.tritq/*.pt`), `data/`, `docs/private/`, `docs/attention-neuron/` y build de LaTeX. **No hay basura versionada.**
- **Solo 37 archivos rastreados**, todos fuente/config/paper/figuras. Sin datasets (64 MB de MNIST quedan fuera de git, correcto) ni artefactos de build.
- **CI** (`.github/workflows/ci.yml`): matriz Ubuntu+Windows × Python 3.10/3.11, instala torch CPU, compila el kernel y corre pytest. Bien.
- **Riesgo CI:** el step "Build C DMA micro-kernel" (`ci.yml:38-39`) hace fallar el job si no hay compilador. En los runners actuales hay gcc/cl, pero conviene marcar el build como no-bloqueante o instalar el toolchain explícito.
- **Python declarado:** README dice `3.9+`, CI prueba `3.10/3.11`, y el entorno local es `3.14`. El código usa `math.isqrt` (3.8+), así que 3.9 es plausible, pero no está verificado en CI.

---

## 11. Recomendaciones priorizadas

### Fase A — Honestidad y coherencia (días, alto retorno)
1. **Recalcular y unificar el bpp** con una sola definición end-to-end (incluyendo metadata, embeddings FP16 y nombres de tensor). Reescribir `0.945/33.86×/0.931` por el valor real medido. Decidir si el "número estrella" pasa a ~0.47 bpp (que es mejor) o al bpp end-to-end con overhead.
2. **Unificar las tablas del paper** (abstract, Tabla 10k, Tabla SOTA) para que describan **un mismo** experimento y configuración; separar explícitamente TinyStories-10k vs v392-toy.
3. **Corregir `examples/evaluate_falsification.py`:** o entrenar sobre TinyStories real (o al menos datos con señal), o renombrarlo a `smoke_test.py` y eliminar los veredictos hardcodeados.
4. **Arreglar la paridad C↔Python** (aplicar `q·scale+min`) y bajar el claim a lo que mida el test.
5. **Corregir claims de docs:** `lossless`→`near-lossless` (y solo del packing), `>90%`→`72%` (o medir), `exponential`→ley de potencias, quitar `necessary and sufficient` y los beneficios CL/analógica sin evidencia.
6. `ast.literal_eval` en `TritQFormat.load`; guardar radios y versión en el header.

### Fase B — Ablaciones que sostienen (o tiran) la tesis
1. Curva **PPL vs bpp** completa (barrido de `r0/r1/r2` y bitrate objetivo), no un punto.
2. Comparación **ejecutada** contra GPTQ/AWQ/INT4-INT2 **budget-matched**, y vs **SVD-truncation a igual bpp**.
3. **Effective rank / entropía** vs λ (el coste en capacidad que hoy no se reporta).
4. Energía retenida por tipo de matriz (qkv/ffn/out/embed) para quitar el sesgo de la máscara radial.

### Fase C — Escala y reproducibilidad
1. Entrenar en un modelo con baselines SOTA comparables (Pythia-160M / GPT-2-124M), con **n semillas**.
2. Publicar **checkpoints** y logs (o un `reproduce.sh`) y declarar **todas** las deps (`modal`, `datasets`, `tokenizers`, `transformers`, `torchvision`) en un `requirements-full.txt`/extra.
3. Reformular el claim: *"calidad 3-bit con 3× menos bits"* en vez de "el único sub-1 bpp".

### Fase D — Packaging
1. Separar las 3 tesis: (i) regularizador/gauge, (ii) códec base-3 + kernel C, (iii) edge/microcontrolador.
2. Marcar `docs/private/` y `docs/attention-neuron/` como históricos en un `docs/README`.
3. Repro script `make all` que genere todas las tablas/figuras.

---

## 12. Veredicto final

La tesis central —**romper la invariancia de gauge imponiendo suavidad espacial y convertirla en compresibilidad espectral**— es buena, simple y con evidencia barata de producir. El código es limpio, testeado y con CI; la ingeniería del kernel C es seria.

Pero el repo, **tal como está hoy**, no es publicable: la cifra estrella (0.945 bpp) no sale del código, el experimento decisivo mide ruido y emite veredictos fijos, el kernel C no es "bit-exacto", y las tablas del propio paper se contradicen entre sí y con el README/whitepaper. Esto no es "falta de investigación": es **falta de un único experimento reproducible y coherente**.

El camino más corto a un trabajo sólido es la Fase A+B: recalcular un solo número honesto (el bpp real), ejecutar de verdad las ablaciones y baselines, y reescribir abstract/tablas/README para que digan lo que el código hace. Hecho eso, la afirmación publicable —*"la suavidad espacial separa la degradación graceful del colapso por debajo de 2 bits"*— se sostiene con evidencia que cualquiera puede reproducir.


---

## Anexo A — Referencias de código (líneas)

| Hallazgo | Ubicación |
| :--- | :--- |
| Docstring con "0.945 bpp / 33.86x" (no coincide con la salida) | `dreg/quantization.py:6` |
| Defecto `r2=0.50` (paper Algoritmo usa 0.40) | `dreg/quantization.py:23` |
| Fórmula de bpp | `dreg/quantization.py:102` |
| Umbral ternario `0.5σ` | `dreg/quantization.py:82` |
| `load` con cuantizador default (radios no leídos) | `dreg/quantization.py:210` |
| `eval()` en deserialización | `dreg/quantization.py:221` |
| Normalización por `numel` → λ no transferible | `dreg/topology.py:54`, `:58` |
| Caché DCT global mutable | `dreg/spectral.py:13` |
| `grid_shape`/`get_grid_dimensions` sin uso real | `dreg/model.py:40`; `dreg/topology.py:14` |
| λ por defecto `0.01` | `dreg/model.py:28` |
| Ruido `randint` en el experimento insignia | `examples/evaluate_falsification.py:82-84` |
| Veredictos hardcodeados | `examples/evaluate_falsification.py:135-136` |
| Claim "bit-exact (0.00000000)" | `examples/benchmark_c_dma.py:5-6` |
| Band0 C con offset fijo `(b0-128)` | `kernel/spectral_dma_kernel.c:135` |
| Band1 C con offset fijo `(nib-7)` | `kernel/spectral_dma_kernel.c:151` |
| `tokens_per_sec` con "36 matrices/token" hardcodeado | `examples/benchmark_c_dma.py:152` |
| Deps Modal no declaradas | `benchmarks/modal_train_tinystories.py:42-51` |
| `torchvision` no declarado | `examples/train_mlp_dirichlet.py:57` |
| Fracciones de banda irreproducibles | `docs/whitepaper.md:105-108` |
| "0.931–0.945 bpp / 34.37×" | `docs/whitepaper.md:30` |
| ">90%" energía | `README.md:23,36` |
| "lossless compression" | `README.md:119` |
| "necessary and sufficient" | `README.md:141` |
| "exponential" (incorrecto) | `README.md:55`; `docs/whitepaper.md:75` |
| Tabla estrella de falsificación | `README.md:134-139` |
| Abstract: "72%", "0.945", "7.25" | `paper/paper-draft.tex:43-45` |
| Algoritmo 1 con `r2=0.40` | `paper/paper-draft.tex:243` |
| Tabla 10k (6.76) vs SOTA (7.25) | `paper/paper-draft.tex:333` vs `:383` |
| Comparación SOTA no ejecutada | `paper/paper-draft.tex:389-392` |
| Calibración con PPL 10.x (≠ tablas) | `docs/ROADMAP.md:74-78` |

*Fin de la auditoría 3.*
