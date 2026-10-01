# Auditoría 4 — Dirichlet Regularization (`dreg`)

**Fecha:** 2026-10-01
**Alcance:** revisión completa del repo `dirichlet-regularization` en `284f499` (`main`): código (`dreg/`, `kernel/`, `examples/`, `benchmarks/`, `tests/`), docs (`README.md`, `docs/ROADMAP.md`, `docs/whitepaper.md`, `docs/private/`), material de publicación (`paper/paper-draft.tex/.pdf`, `references.bib`), config (CI, `pyproject.toml`, `requirements.txt`, `.gitignore`) y verificación remota del repo en GitHub.
**Método:** lectura de fuente + **ejecución directa** de tests, experimentos y benchmarks + medición independiente de métricas + verificación de los artefactos publicados (badge CI, releases).
**Entorno:** Windows, Python 3.14, `torch 2.10.0+cpu`, `pytest 9`, gcc (MSYS2) para el kernel C. Sin GPU.

> Esta auditoría **no repite** `auditoria_1/2/3.md` por ocio: como **no ha habido un solo commit nuevo desde la auditoría 3** (mismo HEAD `284f499`), su valor añadido es (a) **re-verificar** que los hallazgos anteriores siguen abiertos, (b) añadir **hallazgos nuevos** 🆕 que no aparecían, y (c) auditar el **roadmap y los claims de publicación/CI en GitHub**, que antes no se habían comprobado externamente.

---

## 1. Resumen ejecutivo

**El código sigue siendo limpio y el mecanismo científico sigue siendo real** (re-confirmado midiendo yo mismo: al suavizar una matriz, la energía en bajas frecuencias DCT pasa de **6.2% → 96.6%** y $E_D$ de 3.98 → 0.0032). Los **57 tests pasan** (`57 passed in 1.84s`), el **CI en GitHub está en verde** (badge `passing`, 12 runs), el PDF compila (12 páginas, 0 errores LaTeX) y las 15 referencias del `.bib` están todas citadas.

**Pero nada ha cambiado desde la auditoría 3: los problemas siguen ahí, y he encontrado más.** Las cifras estrella del repo (0.945 bpp, tabla de falsificación, "bit-exact", 749.7 KB) siguen sin provenir del código, y ahora además aparece un problema nuevo serio: **el benchmark SOTA del paper (Tabla 4) ejecuta el cuantizador con unos radios que producen ~1.64 bpp — por encima de 1 bit — pero etiqueta la fila como "0.945 bpp" con el número hardcodeado.**

| Artefacto ejecutado | Resultado medido | Resultado afirmado |
| :--- | :--- | :--- |
| `python -m pytest tests/ -q` | ✅ `57 passed in 1.84s` | ✅ "57 tests" |
| Badge CI en GitHub | ✅ `passing` | ✅ README badge |
| bpp del cuantizador default (256×1024) | **0.468 bpp** | ❌ "0.945 bpp" |
| bpp con radios del paper (0.10,0.25,0.40) | **0.354 bpp** | ❌ "0.945 bpp" |
| bpp con radios del benchmark SOTA (`:198`) | **1.637 bpp** 🆕 | ❌ "0.945 bpp" (hardcodeado en `:210`) |
| Fracciones de banda (default, 256×256) | 0.83 / 4.18 / 14.81 / 80.18 % | ❌ "1.7 / 8.4 / 29.7 / 60.2 %" |
| `examples/evaluate_falsification.py` | PPL **132.6 / 131.6 / 134.7 / 132.9** (¡al azar!) 🆕 | ❌ tabla README: 11.88 / 43.43 / 10.80 / 11.54 |
| `examples/benchmark_c_dma.py` | discrepancia C↔PyTorch = **0.7867** | ❌ "bit-exact (0.00000000)" / whitepaper "2.38e-6" 🆕 |
| Aritmética 749.7 KB @ 0.945 bpp (8.39M params lineales) | **967.7 KiB** reales 🆕 | ❌ "749.7 KB" (implica 0.732 bpp) |

**Veredicto:** sigue **no listo para submission**. La causa no ha cambiado: las afirmaciones no provienen del código, y ahora sabemos que **tampoco la fila estrella de la Tabla SOTA proviene del código que la genera**. Un revisor que descargue el repo y ejecute `evaluate_falsification.py` obtiene dos modelos a nivel de azar con etiquetas "CATASTROPHIC COLLAPSE"/"PRESERVED" impresas sin mirar los números.

| Eje | Estado vs auditoría 3 |
| :--- | :--- |
| Idea / marco teórico | ✅ Sin cambios — sólido y publicable |
| Mecanismo espectral | ✅ Re-confirmado con medición independiente |
| Tasa de bits 0.945 bpp | ❌ **Sigue sin reproducirse** (0.35 / 0.47 / 1.64 según config) 🆕 tercer valor |
| Benchmark SOTA (Tabla 4) | ❌ **Fila "0.945 bpp" ejecutada a ~1.64 bpp, bpp hardcodeado** 🆕 |
| Coherencia interna paper/README/whitepaper | ❌ Sin cambios + contradicción 500.5 vs 754.2 KB 🆕 |
| Experimento de falsificación | ❌ Sigue midiendo ruido con veredictos hardcodeados |
| Kernel C "bit-exacto" | ❌ Sigue en 0.7867; whitepaper ahora dice 2.38e-6 🆕 |
| Claims de publicación ("archivado") | ❌ **Falso: no hay releases, tags ni enlaces** 🆕 |
| Calidad de código / tests / CI | ✅ 8/10 — 57 tests, CI verde en GitHub 🆕 verificado remotamente |
| Higiene de repo | ✅ 37 archivos rastreados; 🆕 auditorías sin commitear, `.kilo/` solo en `info/exclude` |

---

## 2. Reproducibilidad de esta auditoría

Desde la raíz del repo, sin red ni GPU salvo la verificación de GitHub:

```bash
# 1) Tests
python -m pytest tests/ -q
#   -> 57 passed in 1.84s

# 2) bpp real por config (default, paper Alg.1, benchmark SOTA)
#   Base3TritQuantizer(r0,r1,r2).quantize_matrix(dct2d(torch.randn(256,1024)))['bpp']
#   default (0.10,0.25,0.50) -> 0.468 | paper (0.10,0.25,0.40) -> 0.354 | SOTA (0.15,0.40,1.0) -> 1.637

# 3) Experimento insignia tal cual está en el repo
python examples/evaluate_falsification.py
#   -> PPL 132.62 / 131.64 (standard) y 134.69 / 132.85 (topo), etiquetas hardcodeadas

# 4) Kernel C (build + benchmark)
python kernel/build_kernel.py && python examples/benchmark_c_dma.py
#   -> Max Absolute Numerical Discrepancy: 0.78671384

# 5) Mecanismo (medición independiente)
#   matriz suavizada por difusión: energía DCT en cuadrante bajo 6.2% -> 96.6%, E_D 3.98 -> 0.0032

# 6) Verificación remota
#   badge CI: <title>CI - passing</title>  |  GitHub: 19 commits, 0 releases, 0 tags, 0 stars
```

## 3. Hallazgos nuevos 🆕

### 🆕 N1 — El benchmark SOTA del paper ejecuta a ~1.64 bpp y publica "0.945 bpp" hardcodeado

`benchmarks/modal_benchmark_quantization_sota.py` es el origen de la **Tabla 4 del paper** (la fila "Base-3 TritQ… 0.945 bpp ✅ único sub-1 bpp"). Dos problemas encadenados:

1. **Radios distintos a todo lo demás** (línea 198): `Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.0)`. Con esa configuración, la retención de bandas en 256×256 es `1.83 / 10.89 / 66.20 / 21.08 %` y el bpp real medido es **1.637** (256×1024) / **1.641** (256×256). Es decir: **la comparación "sub-1.0 bpp" se ejecutó por encima de 1.6 bits**.
2. **El bpp no se mide, se escribe** (línea 210): `log_result("Base-3 TritQ (.tritq, Ours)", 0.945, ...)` — el `0.945` es un literal, igual que las columnas de RAM (`"42.2 MB (DRAM)"`, `"< 1.0 MB (SRAM)"`, líneas 147–210), que son *strings* fijos, no mediciones.

**Impacto:** la única fila donde el método "gana" en la Tabla 4 (y el fundamento de "the only method to achieve sub-1.0 bpp operation") no corresponde a lo que el script ejecuta. Además, los baselines INT4/INT2 del mismo script son RTN casero (round-to-nearest con escala max-abs), no GPTQ/AWQ/QuIP# reales — el paper lo menciona a medias, pero la tabla los presenta como "SOTA comparison".

**Arreglo:** imprimir el `bpp` agregado real (`packed['bpp']`), usar los mismos radios que el resto de artefactos, y medir la RAM en vez de escribirla.

---

### 🆕 N2 — Tres configuraciones de radios mutuamente incompatibles, ninguna produce 0.945 bpp

| Fuente | $(r_0, r_1, r_2)$ | bpp real medido (256×1024) |
| :--- | :--- | :---: |
| `dreg/quantization.py:23` (default del código) | (0.10, 0.25, **0.50**) | **0.468** |
| `paper/paper-draft.tex:243` (Algoritmo 1) | (0.10, 0.25, **0.40**) | **0.354** |
| `benchmarks/...sota.py:198` (Tabla 4) | (0.15, 0.40, **1.0**) | **1.637** |

Barrido hecho para ver si *alguna* config razonable da 0.945: `(0.10,0.30,0.75)→0.913`, `(0.12,0.35,0.75)→0.989` — es decir, **0.945 bpp es alcanzable, pero con radios que no existen en ningún archivo del repo**. La cifra estrella corresponde a una configuración no versionada.

Igual, las fracciones de banda del paper (`1.7 / 8.4 / 29.7 / 60.2 %`, tex §3.3) no coinciden con ninguna de las tres configs (default real: `0.83 / 4.18 / 14.81 / 80.18 %`).

---

### 🆕 N3 — Aritmética imposible: "749.7 KB" @ 0.945 bpp

El paper (tex:392) y el ROADMAP (`ROADMAP.md:82`) afirman el checkpoint de 10M a **"0.945 bpp / 749.7 KB"**. Para la arquitectura descrita (8 capas, d=256, ffn=1024):

- Params en proyecciones lineales = **8,388,608**.
- A 0.945 bpp → **967.7 KiB** (no 749.7 KB).
- 749.7 KB sobre esos params → **0.732 bpp** (una cuarta cifra de bpp nueva).
- 0.945 bpp × 10M params → 1153.6 KiB (si se cuentan embeddings, aún peor).

Ninguna combinación de "10M params", "0.945 bpp" y "749.7 KB" es consistente. Y si los embeddings quedan en FP16 aparte, eso debería decirse explícitamente en la tabla (la objeción #2 de la auditoría 2 sigue vigente).

---

### 🆕 N4 — Whitepaper: RAM activa de L=12 contradicha dentro del propio documento

- `whitepaper.md:31` y `:170` → **500.5 KB** ("rompe la barrera de 512 KB").
- `whitepaper.md:172` y `:268` → **754.2 KB** para el mismo modelo L=12 (v394).
- `README.md:152` → repite **500.5 KB**.

Son dos afirmaciones incompatibles sobre el mismo número, separadas por 2 líneas en la misma tabla (`:170` vs `:172`). La headline "500.5 KB < 512 KB" es la que se usa en el resumen y el README; la otra la desmonta.

---

### 🆕 N5 — El whitepaper afirma precisión de máquina para el kernel C; el benchmark propio mide 0.7867

- `whitepaper.md:188` → C-DMA con error `2.38×10⁻⁶` ("Precisión de máquina preservada"); `:187` → Python DMA con "identidad matemática exacta 0.00000000".
- `examples/benchmark_c_dma.py:5-6` (docstring) → "verifying bit-exact mathematical match (0.00000000)".
- **Lo que imprime el propio benchmark al ejecutarlo:** `Maximum Absolute Numerical Discrepancy: 0.78671384` (con una nota que culpa a "single-precision floating-point", cuando en realidad el decodificador C usa offsets fijos `(b0-128)` y `(nib-7)` — `kernel/spectral_dma_kernel.c:135,151` — mientras Python guarda min/scale afines por tensor).

La causa raíz es de diseño, no de punto flotante: **el kernel C no implementa el mismo protocolo de cuantización que Python**, así que "bit-exact" es falso por construcción hasta que se pase el min/scale real al C.

### 🆕 N6 — El experimento insignia: ambos modelos están a nivel de azar y las etiquetas son hardcodeadas

Ejecutado hoy (`examples/evaluate_falsification.py`):

```
Standard Model (Unordered)     | 132.62 | 131.64 | CATASTROPHIC COLLAPSE
Topographic Model (Dirichlet)  | 134.69 | 132.85 | PRESERVED (STABLE)
Linear Weights Bit Rate: ~0.479 bpp (66.9x compression)
```

1. Los datos son `torch.randint` **(tokens aleatorios, líneas 82–83)**: con vocab 128, el PPL de azar es ≈128. Ambos modelos (132.6 y 134.7) están **en el suelo estadístico**; ninguno aprendió nada, no hay "colapso" ni "preservación" que medir.
2. Las etiquetas `CATASTROPHIC COLLAPSE` / `PRESERVED (STABLE)` están en `print` (líneas 135–136) **sin condición sobre los números**: aunque el modelo estándar fuera mejor, imprimiría lo mismo.
3. El bpp real del script es 0.479, no el "0.945 bpp" que anuncia su propio docstring (línea 7) y su cabecera de tabla (línea 133).
4. **La tabla del README** (11.88 / 43.43 / 10.80 / 11.54, `README.md:134-139`) **no puede salir de este script** — ni de ningún artefacto rastreado. Proviene de la serie `v392` en `docs/attention-neuron/`, que está **ignorada por git** (ver N9).
5. `README.md:130` dice `N=640 Sequences`; el código usa 600 train + 200 test = **800**.

---

### 🆕 N7 — El paper afirma "archivado permanente" de pesos y logs: es falso

`paper-draft.tex:408` (§ Reproducibility): *"Pre-trained FP32 weights, calibrated `.tritq` models, validation scripts, and evaluation logs are **permanently archived in open repositories** for independent verification."*

Verificado en GitHub: **0 releases, 0 tags, sin Zenodo/DOI/HuggingFace en el tex, sin enlaces a checkpoints**. Los checkpoints solo existen en un volumen privado de Modal (`dreg-checkpoints`), que no es un "open repository". Este es probablemente el claim que un revisor/AC puede considerar **engañoso**, y es trivialmente comprobable.

Además, sin tag `v1.0.0` (Fase 4 del roadmap pendiente) y con `pyproject.toml` declarando `version = "1.0.0"`, no hay ningún punto de versión publicado.

---

### 🆕 N8 — Requerimientos insuficientes para ejecutar los ejemplos/benchmarks anunciados

- `requirements.txt` = solo `torch` + `numpy`.
- `examples/train_mlp_dirichlet.py:57` necesita **`torchvision`** (no declarado).
- `benchmarks/*.py` necesitan **`modal`, `datasets`, `tokenizers`** (no declarados; solo en `modal.Image.pip_install` interno).
- El README documenta `pip install -r requirements.txt` y luego `python examples/train_mlp_dirichlet.py` → fallo inmediato en el primer ejemplo "Beyond-Transformers".

Sugerencia: extras en `pyproject.toml` (`[project.optional-dependencies] examples/benchmarks`) y que el CI ejecute al menos un smoke test de `train_topographic.py`.

### 🆕 N9 — Higiene: auditorías sin commitear, la evidencia clave está ignorada, `.kilo/` fuera del `.gitignore`

- `docs/auditoria_1.md`, `auditoria_2.md`, `auditoria_3.md` están **sin trackear** (`??`). Si el repo se clona en otra máquina, las auditorías desaparecen. Commitearlas (o ignorarlas deliberadamente y decirlo).
- `.gitignore` excluye `docs/private/` y `docs/attention-neuron/`: correcto por privacidad, pero significa que **toda la evidencia de la serie v379–v395 (17 findings + 17 prototipos + figuras) de la que dependen el whitepaper y las tablas del paper no está en el repo**. El whitepaper la cita como si fuera parte del artefacto público. Solución: mover la evidencia necesaria (logs/tablas) a un directorio versionado, o dejar de citarla como reproducible.
- Existe un **worktree completo duplicado** en `.kilo/worktrees/juvenile-podium/` (con su propio `.git`), excluido solo vía `.git/info/exclude` (local, no versionado): en cualquier otra máquina `git status` lo mostraría. Añadir `.kilo/` a `.gitignore`.
- `.pytest_cache/`, `data/MNIST/` (~11 MB), `*.egg-info`, `__pycache__` y `kernel/*.dll` están correctamente ignorados; **git rastrea exactamente 37 archivos** — higiene base correcta.

---

### 🆕 N10 — Tests: 57 ✅, pero ninguno cubre los claims

Los tests son correctos pero **no validan ninguna cifra publicitada**:

- `test_bpp_is_sub_one` solo afirma `bpp < 2.0` con un espectro sintético de zeros — no `≈0.945`.
- No hay test de `quantize_matrix` con radios custom + `save/load` (por eso el desajuste del `load()` con default — `quantization.py:210` — pasa inadvertido; los radios **no se serializan en el `.tritq`**, así que un archivo guardado con radios no-default se decodificaría mal).
- No hay test que ejecute `evaluate_falsification.py` ni que verifique la salida del benchmark C (el "bit-exact" se declara en docstring, se contradice en runtime, y ningún test lo atrapa).
- `TritQFormat.load` usa `eval()` sobre el config del archivo (`:221`, marcado `noqa: S307`): funciona, pero para un formato binario descargado de internet es una puerta de ejecución — usa `ast.literal_eval`.

---

## 4. Estado de los hallazgos de las auditorías 1–3 (re-verificados)

Como no hubo commits nuevos, **todos siguen abiertos**. Confirmación ejecutada hoy:

| Hallazgo (auditorías 1–3) | Estado hoy |
| :--- | :--- |
| H1 bpp 0.945 no reproducible (real 0.35–0.48) | ❌ Abierto — y ahora 3 configs, 3 valores distintos (N2) |
| H2 paper §3.3 tres cifras inconsistentes (radios/fracciones/bpp) | ❌ Abierto (`tex:170-188` vs `:243`) |
| H3 falsificación = ruido + veredictos hardcodeados | ❌ Abierto — ejecutado hoy: PPL ~132/133 (N6) |
| H4 tablas del paper se contradicen (6.76 vs 7.25; 5.83/6.18 vs 6.11/6.46) | ❌ Abierto (`tex:326-327` vs `:379-383`; el abstract usa 7.25) |
| H5 kernel C no es bit-exacto (0.79) | ❌ Abierto — reproducido hoy (N5) |
| H6 λ no transferible entre shapes (normalización por `numel`) | ❌ Abierto (`topology.py:54,58`) |
| H7 deps no declaradas (torchvision/modal/…) | ❌ Abierto (N8) |
| H8 "exponential" decay en README/whitepaper (es $1/(1+\lambda(u^2+v^2))$, no exponencial) | ❌ Abierto (`README.md:55`, `whitepaper.md:75`) |
| H9 "necessary and sufficient" / "lossless" sobreclaims | ❌ Abierto (`README.md:119,141`, `tex:106`) |
| H10 Caché DCT global mutable; `grid_shape` sin uso | ❌ Abierto (`spectral.py:13`, `topology.py:14`) |

**Ningún hallazgo de tres auditorías ha sido corregido.** El repo está congelado en el estado "antes de las correcciones".

---

## 5. Auditoría del ROADMAP (`docs/ROADMAP.md`)

**Lo bueno (rareza digna de elogio):** el ROADMAP registra los números de calibración **sin maquillar** — incluye que $\lambda=0.01$ es inútil (+37.51 peor) y que el coste de FP32 sube con λ (10.48→10.98). Esa honestidad interna choca con el paper, que presenta solo la fila final. Es el documento más fiable del repo.

**Problemas:**

1. **Desactualizado respecto a su propio estado:** cabecera `Fecha: 2026-10-01 / En progreso`, Fase 3 marcada ✅ completa pero Fase 0 aún tiene `[ ] Renombrar directorio local` (el directorio local **ya** se llama `dirichlet-regularization` — checkbox obsoleto).
2. **Timeline muerto:** "Semana 0 (ahora): Fase 0 … Semana 6-7: Fase 3" — pero la Fase 3 (paper) ya está completada en la semana 0. Cualquier lector detecta que la planificación no se cumple ni se revisa.
3. **Fase 2.1 afirma** "checkpoints `.pt` y `.tritq` a 0.945 bpp / 749.7 KB almacenados en volumen Modal" — externo, no accesible, y con la aritmética rota de N3.
4. **Fase 2.3 (ablaciones): 4 de 5 puntos sin hacer** (radios, profundidad, ancho, block-size) — justo las ablaciones que habrían detectado el desajuste de radios de N2. La única hecha es la de λ.
5. **Fase 4 completa sin empezar** (repro script, supplementary, tag `v1.0.0`, submission). Coherente con "no listo", pero el abstract ya está escrito en pasado ("we achieve… we demonstrate…").
6. Falta el ítem más importante en ninguna fase: **un script único que genere Tablas 1–4 y Figuras 1–4 desde cero**. Sin eso, cada tabla vive solo en el ROADMAP.

## 6. Auditoría del paper (`paper/paper-draft.*`)

**Lo que está bien:**

- Compila limpio: 12 páginas, **0 errores, 0 overfull boxes** (verificado en `paper-draft.log`), PDF de 3.008.609 bytes (ROADMAP dice 3.01 MB ✓).
- Estructura clásica completa (Intro, Related, Method con Algoritmo 1, Experiments con 4 tablas, Discussion, Conclusion).
- Las 15 referencias de `references.bib` están **todas citadas** en el texto (verificado key-by-key) y son works relevantes y reales (GPTQ, AWQ, QuIP#, SmoothQuant, DCT/JPEG, Laplacian Eigenmaps, SOM, TinyStories…).
- El argumento de gauge-symmetry (§1) es el activo conceptual más fuerte, tal como señaló la auditoría 2.
- La §5.1 (rank-1 clamping attractor) es un negative result bien contado.

**Lo que está mal (consolidado, incluye lo de aud 3):**

| # | Problema | Ubicación |
| :-- | :--- | :--- |
| P1 | Abstract: "0.945 bpp / 33.86×", "72.1%", "7.25 PPL" — ninguno sale del código | `tex:43-47` |
| P2 | Tabla 10k da ventaja **6.76**; Tabla SOTA da **7.25**; el abstract usa 7.25 | `:327` vs `:383` |
| P3 | FP32 Val PPL inconsistente entre tablas del mismo run de 10k: 5.83/6.18 vs 6.11/6.46 | `:326-327` vs `:379` |
| P4 | Algoritmo 1 con $r_2=0.40$ y "RETURN … (0.945 bpp)" — el return es falso para sus propios radios | `:243`, `:265` |
| P5 | §3.3 fracciones de banda irreproducibles | `:170-173` |
| P6 | Tabla SOTA: fila ejecutada a 1.64 bpp etiquetada 0.945 (N1) y "RAM" hardcodeada | `:383` |
| P7 | "the only method to achieve sub-1.0 bpp" sin ejecutar GPTQ/AWQ/QuIP# (INT4 es RTN casero) | `:390-392` |
| P8 | "749.7 KB" imposible (N3) | `:392` |
| P9 | § Reproducibility: "permanently archived" falso (N7) | `:408` |
| P10 | "necessary and sufficient inductive bias" — claim fuertísimo con evidencia de juguete (TinyStories-10M + MNIST-MLP) | `:106`, `README:141` |
| P11 | "exponential energy concentration" — el decaimiento demostrado es $1/(1+\lambda(u^2+v^2))$, algebraico | `tex:43`, `README:55` |
| P12 | Figuras: el ROADMAP enumera "Fig 3 = DMA, Fig 4 = Pareto" pero solo hay 3 PNGs (el DMA es un tabloide LaTeX); caption de fig2 dice ">72%" mientras el README dice ">90%" | `tex:228`, `README:23` |

**Objeciones de revisión (auditoría 2) sin trabajar:** prior art de Laplacian smoothing/GNNs, embeddings fuera de la compresión, codec base-3 sin ahorro entropético (el ahorro viene del truncamiento), escala de juguete. **Ninguna abordada.**

## 7. Auditoría de código (`dreg/`, `kernel/`, `examples/`)

**Fortalezas (confirmadas por lectura + ejecución):**

- ~635 LOC de paquete, limpios, tipados, sin dependencias raras (`torch`+`numpy`).
- `spectral.py`: DCT-II ortonormal correcta (roundtrip y Parseval testeado).
- `topology.py`: energía de Dirichlet con vecindario correcto; `DirichletLoss` itera `modules()` de forma robusta.
- `model.py`: transformer estándar bien construido (masks causales correctos, SwiGLU, init correcto).
- Kernel C: 482 LOC con abstracción Win32/POSIX real, compilación verificada hoy con gcc MSYS2 (`-O3 -mavx2 -mfma`), benchmark funcional.
- Tests: 57, rápidos (1.84s), cubren invariantes matemáticos (LUT, ortogonalidad DCT, masks disjointos, roundtrip serialización).

**Debilidades (además de N10):**

| Problema | Ubicación |
| :--- | :--- |
| `bpp` del formato depende de shape+cutoff, no del método; docstring promete 0.945 | `quantization.py:6,102` |
| Radios no serializados en `.tritq`; `load()` usa default → decodificación incorrecta con radios custom; `save()` también fuerza default | `quantization.py:156,210` |
| `eval()` en deserialización | `quantization.py:221` |
| Normalización por `numel` → λ no transferible entre shapes (por eso λ=0.01 no hace nada y hace falta λ=30) | `topology.py:54,58` |
| `dirichlet_energy_2d` envuelve bordes (diferencias periódicas): mezcla la fila 0 con la M−1 — para "mapas corticales" un `replicate` sería más fiel | `dreg/topology.py` |
| Caché DCT global sin límite ni invalidación por device | `spectral.py:13` |
| `TopographicConfig.topo_lambda` default 0.01 — valor que el propio paper demuestra inútil | `model.py:28` |
| `evaluate_ppl` promedia loss por batch (no ponderado por tokens) y hace `exp(min(loss,20))` — el clip silencia colapsos | `evaluate_falsification.py:38` |
| `benchmark_c_dma.py` calcula tok/s con "36 matrices/token" hardcodeado | `benchmark_c_dma.py:152` |
| Emisión de `CATASTROPHIC COLLAPSE` sin umbral | `evaluate_falsification.py:135-136` |

---

## 8. Veredicto final

La idea sigue siendo **buena, simple y publicable**, y esta auditoría la re-confirma empíricamente: la suavidad espacial concentra la energía DCT de forma masiva (6.2% → 96.6% al suavizar) y el código que la implementa está limpio, testeado y con CI verde en GitHub. Eso no ha cambiado.

Lo que ha cambiado (a peor) es el mapa completo del problema: **ya no es "una cifra no reproducible", son cuatro sistemas de afirmaciones que no cuadran entre sí** — código (0.35/0.47 bpp), benchmark SOTA (1.64 bpp ejecutados, 0.945 etiquetados), paper (0.945 + 749.7 KB), whitepaper (0.931–0.945 + 500.5/754.2 KB) — sostenidas por un experimento insignia que mide azar y escribe sus veredictos, y por un claim de "archivado permanente" que es objetivamente falso. **Ninguno de los diez hallazgos de las auditorías 1–3 se ha corregido** porque no ha habido ni un commit desde entonces.

Orden de ataque recomendado (el mismo que la auditoría 3, ahora con un paso cero):

0. **Commit de este documento + las auditorías anteriores**, y decidir si `docs/private/` y `attention-neuron/` quedan versionados o fuera.
1. **Fase A — un solo número honesto:** elegir UNA config de radios, medirla, y propagar el bpp resultante a docstring, README, abstract, Algoritmo 1, Tabla 4 y ROADMAP (o arreglar el código para que la config publicada dé 0.945 — es alcanzable con radios ≈ (0.10, 0.30, 0.75)).
2. **Fase B — dos experimentos de verdad:** (i) reescribir la falsificación con datos reales (TinyStories o al menos secuencias estructuradas) y veredictos derivados de umbrales; (ii) arreglar el benchmark SOTA para que mida bpp/RAM y añadir al menos un baseline real (GPTQ 4-bit) o renombrar la tabla a "RTN baselines".
3. **Fase C — coherencia textual:** reconciliar 6.76 vs 7.25, 500.5 vs 754.2, 749.7 KB, >90% vs 72%, "exponential", "necessary and sufficient", "permanently archived" (borrarlo o archivar de verdad en Zenodo/HF con DOI).
4. **Fase D — kernel C:** pasar min/scale reales al decoder C (hoy hardcodea −128/−7) para que la discrepancia baje de 0.79 a ~1e-6, y solo entonces publicar "bit-exact".
5. **Fase E — publicación:** repro script, extras de dependencias, tag `v1.0.0`, subir preprint.

Hecho 1–3, la tesis publicable — *"romper la invariancia de gauge produce compresibilidad espectral, y eso separa el degradado graceful del colapso por debajo de 2 bits"* — queda respaldada por evidencia que cualquiera puede reproducir con `pip install -r requirements.txt && python -m pytest && python examples/...`. Hoy, todavía no.

## Anexo A — Referencias de código (líneas)

| Hallazgo | Ubicación |
| :--- | :--- |
| 🆕 Radios SOTA `(0.15,0.40,1.0)` → 1.64 bpp | `benchmarks/modal_benchmark_quantization_sota.py:198` |
| 🆕 bpp `0.945` hardcodeado en la tabla | `benchmarks/modal_benchmark_quantization_sota.py:210` |
| 🆕 RAM como strings fijos | `benchmarks/modal_benchmark_quantization_sota.py:147-210` |
| 🆕 Datos aleatorios en la falsificación | `examples/evaluate_falsification.py:82-83` |
| 🆕 Veredictos hardcodeados | `examples/evaluate_falsification.py:135-136` |
| 🆕 bpp real del script (0.479) vs "0.945" del docstring | `examples/evaluate_falsification.py:7,138` |
| 🆕 "archivado permanente" falso | `paper/paper-draft.tex:408` |
| 🆕 749.7 KB inconsistente | `paper/paper-draft.tex:392`; `docs/ROADMAP.md:82` |
| 🆕 Whitepaper 500.5 vs 754.2 KB | `docs/whitepaper.md:31,170` vs `:172,268` |
| 🆕 Whitepaper error 2.38e-6 vs real 0.7867 | `docs/whitepaper.md:188` vs salida de `examples/benchmark_c_dma.py` |
| 🆕 `N=640` vs 600+200=800 | `README.md:130`; `examples/evaluate_falsification.py:79-80` |
| 🆕 deps sin declarar | `requirements.txt` vs `examples/train_mlp_dirichlet.py:57`, `benchmarks/modal_train_tinystories.py` |
| 🆕 `.kilo/` solo en `info/exclude` | `.git/info/exclude` |
| Docstring "0.945 bpp / 33.86×" | `dreg/quantization.py:6` |
| Default `r2=0.50` (paper: 0.40) | `dreg/quantization.py:23` |
| Fórmula de bpp | `dreg/quantization.py:102` |
| `load()` con cuantizador default / radios no serializados | `dreg/quantization.py:156,210` |
| `eval()` en deserialización | `dreg/quantization.py:221` |
| λ no transferible (normalización `numel`) | `dreg/topology.py:54,58` |
| Caché DCT global | `dreg/spectral.py:13` |
| λ default inútil `0.01` | `dreg/model.py:28` |
| Claim "bit-exact (0.00000000)" | `examples/benchmark_c_dma.py:5-6,110` |
| Offset fijo band0/band1 en C | `kernel/spectral_dma_kernel.c:135,151` |
| Radios Algoritmo 1 | `paper/paper-draft.tex:243` |
| Tablas 6.76 vs 7.25 | `paper/paper-draft.tex:327` vs `:383` |
| ">90%" vs ">72%" | `README.md:23,36` vs `paper/paper-draft.tex:43`, caption fig2 |
| "necessary and sufficient" | `README.md:141`; `paper/paper-draft.tex:106` |
| "lossless compression" | `README.md:119` |
| "exponential" (incorrecto) | `README.md:55`; `docs/whitepaper.md:75` |
| Checkboxes/timeline obsoletos | `docs/ROADMAP.md:12,171-179` |

*Fin de la auditoría 4.*






