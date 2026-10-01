# Síntesis de Auditorías y Plan de Remediación

**Fecha de consolidación:** 2026-10-01  
**Alcance:** Síntesis unificada de `docs/auditoria_1.md`, `docs/auditoria_2.md`, `docs/auditoria_3.md` y `docs/auditoria_4.md` sobre el commit `284f499`.  
**Objetivo:** Servir de documento único de referencia, planificación y checklist ejecutable para subsanar todos los problemas detectados en el repositorio.

---

## 1. Diagnóstico Ejecutivo Global

Las cuatro auditorías coinciden en un diagnóstico unánime y nítido:

1. **La idea científica y el mecanismo matemático son válidos, elegantes y publicables**:
   - La cadena teórica: Energía de Dirichlet $\to$ seminorma de Sobolev $\to$ Laplaciano discreto con condiciones de Neumann $\to$ autovectores en base DCT-II $\to$ decaimiento espectral $1/(1 + \lambda(u^2+v^2))$ es rigurosa.
   - Romper la simetría de gauge (permutación de neuronas) forzando continuidad espacial 2D en los pesos es un argumento conceptual original y fértil.
   - El efecto es real y medible: la energía se concentra masivamente en bajas frecuencias (de ~6% a >96%), reteniendo 6× más energía y 6× menor error relativo que matrices no regularizadas bajo el mismo presupuesto de coeficientes.
   - La base de código es limpia (~635 LOC de Python), modular (`topology`, `spectral`, `quantization`, `model`), con 57 tests unitarios pasando y CI funcional.

2. **La evidencia experimental y los números anunciados están severamente comprometidos**:
   - **Tasa de bits no reproducible / desalineada**: Se publicita 0.945 bpp / 33.86× en abstract y README, pero el código por defecto da ~0.47 bpp (68×), el Algoritmo 1 del paper da ~0.35 bpp (90×), y el script del benchmark SOTA corre a ~1.64 bpp pero hardcodea el string "0.945" en el log.
   - **Experimento de falsificación inválido**: `examples/evaluate_falsification.py` entrena sobre ruido uniforme aleatorio (`torch.randint`), ambos modelos quedan al nivel de azar estadístico (PPL ~132 para vocabulario 128), y las etiquetas `CATASTROPHIC COLLAPSE` y `PRESERVED (STABLE)` están escritas a mano como strings fijos en el `print`.
   - **El Kernel C no es bit-exacto**: Discrepancia real de **0.7867** respecto a PyTorch (frente al "0.00000000" afirmado). Causa raíz: C decodifica asumiendo offsets fijos simétricos (`-128`, `-7`), mientras Python cuantiza de forma afín con `min` y `scale` asimétricos por banda.
   - **Fallas de seguridad e integridad en `.tritq`**: `TritQFormat.load()` usa `eval()` (vulnerabilidad de ejecución arbitraria de código) e ignora los radios usados al cuantizar, usando siempre los defaults (riesgo de corrupción silenciosa).
   - **Inconsistencias cruzadas en publicaciones**: Tablas con números de PPL discordantes en el paper (5.83/6.18 vs 6.11/6.46 vs 11.88); sobre-claims de compresión "lossless" cuando se trunca el 80% de coeficientes; afirmaciones de "archivado permanente" en repositorios abiertos cuando no existen tags, releases ni checkpoints públicos.

**Conclusión:** El trabajo es completamente salvable porque el núcleo técnico funciona. Sin embargo, requiere una limpieza profunda de honestidad experimental, sincronización de parámetros y corrección de bugs para que cualquier revisor o usuario pueda clonar, ejecutar y verificar los resultados de forma transparente.

---

## 2. Matriz Consolidada de Hallazgos

| ID | Área | Descripción del Problema | Gravedad | Archivos Afectados |
| :--- | :--- | :--- | :---: | :--- |
| **SEC-1** | Seguridad | `eval(cfg_str)` en deserialización de archivos `.tritq` (RCE). | 🔴 Crítico | `dreg/quantization.py:221` |
| **SER-1** | Formato `.tritq` | `save()` no guarda radios `(r0, r1, r2)`; `load()` usa siempre el default `Base3TritQuantizer()`. Si se cuantiza con otros radios, la reconstrucción se corrompe. | 🔴 Crítico | `dreg/quantization.py:156,210` |
| **KER-1** | Kernel C | Discrepancia C ↔ Python de 0.7867 (afirmado 0.00000000). C usa offsets fijos `(b0-128)`, `(nib-7)`; Python usa escala afín `min`/`scale`. | 🔴 Crítico | `kernel/spectral_dma_kernel.c:135,151`<br>`examples/benchmark_c_dma.py` |
| **EXP-1** | Experimentos | `evaluate_falsification.py` usa datos aleatorios (`randint`), modelos a nivel de azar (PPL ~132) y veredictos en texto estático. | 🔴 Crítico | `examples/evaluate_falsification.py:82,135` |
| **EXP-2** | Benchmarks SOTA | `modal_benchmark_quantization_sota.py` usa radios `(0.15, 0.40, 1.0)` (~1.64 bpp) pero hardcodea la etiqueta "0.945 bpp" y RAM estática. | 🔴 Crítico | `benchmarks/modal_benchmark_quantization_sota.py:198,210` |
| **NUM-1** | Parámetros y BPP | Desalineación de radios y bpp entre default (0.47 bpp), Algoritmo 1 (0.35 bpp), paper text (0.945 bpp) y benchmark SOTA (1.64 bpp). | 🔴 Crítico | `dreg/quantization.py:6,23`<br>`paper/paper-draft.tex:44,188,243` |
| **DOC-1** | Claims de Marketing | "10×–34× lossless compression" es falso (solo el packing de trits es lossless; se trunca la banda 3 y se cuantizan bandas 0-2). | 🟠 Alto | `README.md:119`<br>`docs/whitepaper.md` |
| **DOC-2** | Matemáticas | "Exponential energy concentration" es erróneo; el decaimiento teórico es ley de potencias $1/(1+\lambda(u^2+v^2))$. | 🟠 Alto | `paper/paper-draft.tex:43`<br>`README.md:55`<br>`docs/whitepaper.md:75` |
| **DOC-3** | Inconsistencia Paper | Discrepancias entre Abstract (7.25 PPL), Tabla 10k (6.76 PPL) y Tabla SOTA; FP32 PPL 5.83 vs 6.11; >90% vs 72% energía. | 🟠 Alto | `paper/paper-draft.tex`<br>`README.md:23,134-139` |
| **DOC-4** | Aritmética / RAM | Tamaño "749.7 KB @ 0.945 bpp" es aritméticamente imposible para 8.39M params (debería ser 967.7 KiB). Whitepaper dice 500.5 KB y 754.2 KB. | 🟠 Alto | `paper/paper-draft.tex:392`<br>`docs/whitepaper.md:31,172` |
| **DOC-5** | Claims de Publicación | "Permanently archived in open repositories" cuando no hay releases, tags ni checkpoints públicos. | 🟠 Alto | `paper/paper-draft.tex:408` |
| **DEP-1** | Entorno | Faltan dependencias en `requirements.txt` / `pyproject.toml` (`torchvision`, `modal`, `datasets`, `tokenizers`, `transformers`). | 🟡 Medio | `requirements.txt`<br>`pyproject.toml` |
| **GIT-1** | Higiene Git | `.kilo/` no está en `.gitignore` (solo en local `info/exclude`). Auditorías sin commitear. | 🟡 Medio | `.gitignore` |
| **ALG-1** | Normalización $\lambda$ | `dirichlet_energy_2d` normaliza por `sheet.numel()`, haciendo que $\lambda$ dependa de la forma de la matriz e impidiendo transferencia directa. | 🟡 Medio | `dreg/topology.py:54,58` |
| **OPT-1** | Optimización | `dct_matrix_1d` usa bucle Python con cosenos $O(n^2)$; caché global `_DCT_CACHE` mutable sin límite por device. | 🟢 Bajo | `dreg/spectral.py:13,38` |

---

## 3. Plan de Acción y Checklist Priorizado

### Fase 1: Código Inmediato, Seguridad y Paridad (P0 — Bloqueante)
*Objetivo: Eliminar riesgos de seguridad, garantizar integridad de guardado/carga `.tritq`, y lograr paridad numérica real C ↔ Python.*

- [x] **1.1. Seguridad en deserialización**:
  - Reemplazar `eval(cfg_str)` por `ast.literal_eval` o `json.loads` en `TritQFormat.load` (`dreg/quantization.py:221`).
  - Añadir test unitario que verifique el rechazo de payloads no seguros.
- [x] **1.2. Serialización completa de parámetros `.tritq`**:
  - Modificar `TritQFormat.save` para incluir `(r0, r1, r2)`, versión de formato (`version=1`) y shape en la cabecera JSON/dict.
  - Modificar `TritQFormat.load` para instanciar el decodificador con los radios exactos almacenados en el archivo.
  - Añadir test de roundtrip `save/load` con radios no-default (ej. `r2=0.40`).
- [x] **1.3. Paridad numérica Kernel C ↔ Python**:
  - Investigar y alinear el esquema de cuantización de Band 0 y Band 1:
    - *Opción A (Recomendada)*: Pasar los offsets reales `min0`, `min1` al kernel C junto con `scale0`, `scale1`, de modo que C haga `q * scale + min`.
    - *Opción B*: Implementar cuantización simétrica en Python (`-max_abs` a `+max_abs`) si se desea mantener decodificador C ultra-simple.
  - Actualizar `examples/benchmark_c_dma.py` y `kernel/spectral_dma_kernel.c`.
  - Añadir test automatizado en `tests/` que verifique que el error absoluto C ↔ PyTorch sea $< 1\times 10^{-5}$.
- [x] **1.4. Higiene del repositorio y dependencias**:
  - Añadir `.kilo/` y cachés temporales a `.gitignore`.
  - Crear secciones opcionales en `pyproject.toml` (`[project.optional-dependencies]`) para `examples` y `benchmarks` (`torchvision`, `datasets`, etc.).
  - Commitear las auditorías y este plan a git para que no se pierdan.

---

### Fase 2: Consistencia del Cuantizador y Eliminación de Hardcodes (P0)
*Objetivo: Asegurar que las tasas de bits y compresión provengan de cálculos dinámicos reales y unificar la configuración.*

- [x] **2.1. Definición única de configuración de radios de referencia**:
  - Establecer los radios canónicos por defecto (evaluar si mantener `(0.10, 0.25, 0.50)` a ~0.47 bpp o definir la configuración deseada).
  - Documentar explícitamente el porcentaje de coeficientes y bpp resultante de dicha configuración.
- [x] **2.2. Eliminar bpp y métricas hardcodeadas en código**:
  - En `dreg/quantization.py`: eliminar "0.945 bpp / 33.86x" del docstring general o aclararlo como ejemplo dependiente de shape y radios.
  - En `benchmarks/modal_benchmark_quantization_sota.py`: corregir los radios en la línea 198, registrar el bpp real (`packed['bpp']`) en lugar del literal 0.945, y medir o clarificar el uso de memoria en lugar de usar strings fijos.
  - En `examples/benchmark_c_dma.py`: eliminar el hardcode de "36 matrices/token" o justificarlo explícitamente con la fórmula de capas.

---

### Fase 3: Experimento de Falsificación Real y Verificable (P1)
*Objetivo: Reemplazar el mock con ruido por un experimento reproducible y honesto que evalúe la hipótesis científica.*

- [x] **3.1. Reemplazo del dataset de ruido por datos estructurados**:
  - Modificar `examples/evaluate_falsification.py` para utilizar datos con estructura de lenguaje real (un subconjunto accesible de TinyStories o secuencias sintéticas generadas por autómatas/gramáticas sin dependencias externas pesadas).
- [x] **3.2. Veredictos dinámicos basados en métricas**:
  - Eliminar los prints hardcodeados `"CATASTROPHIC COLLAPSE"` y `"PRESERVED (STABLE)"`.
  - Implementar lógica condicional real: calcular $\Delta \text{PPL} = \text{PPL}_{\text{quant}} - \text{PPL}_{\text{fp32}}$ o ratio relativo de degradación y emitir el diagnóstico basado en umbrales objetivos.
  - Reportar la tasa de bits real calculada en runtime sobre los pesos del modelo.

---

### Fase 4: Coherencia Documental y Saneamiento de Claims (P1)
*Objetivo: Corregir el README, Whitepaper y Paper para alinearlos 100% con la realidad del código y la matemática.*

- [x] **4.1. Correcciones en README.md**:
  - Cambiar "10×–34× lossless compression" por "lossy spectral compression with exact lossless base-3 bit-packing".
  - Cambiar "exponential energy concentration" por "power-law energy decay" ($1/(1+\lambda(u^2+v^2))$).
  - Reemplazar la tabla de "falsificación" por los números reales reproducibles o contextualizarla con el experimento real de Fase 3.
  - Reubicar los claims de Continual Learning y Hardware Analógico bajo una sección explícita de "Hipótesis Teóricas y Trabajo Futuro".
- [x] **4.2. Correcciones en Whitepaper y ROADMAP**:
  - Reconciliar la discrepancia de RAM activa para L=12 (500.5 KB vs 754.2 KB).
  - Actualizar el estado de las tareas y checkboxes en `docs/ROADMAP.md`.
  - Corregir el claim de discrepancia numérica del kernel C al valor verificado tras la Fase 1.
- [x] **4.3. Correcciones en Paper (`paper/paper-draft.tex`)**:
  - Corregir la contradicción de $\Delta$PPL entre Abstract (7.25), Tabla 10k (6.76) y Tabla SOTA.
  - Sincronizar los radios del Algoritmo 1 con el código y el valor real de bpp producido.
  - Corregir "exponential" a power-law en abstract e introducción.
  - Corregir o matizar la frase "permanently archived" en la sección de reproducibilidad hasta contar con DOI/Zenodo o checkpoints públicos.
  - Corregir la inconsistencia de tamaño "749.7 KB @ 0.945 bpp".

---

### Fase 5: Normalización de $\lambda$ y Baselines Científicos (P2)
*Objetivo: Blindar la fundamentación científica frente a revisores pares.*

- [x] **5.1. Normalización de $\lambda$ invariante a la escala**:
  - Estudiar normalizar `dirichlet_energy_2d` por el número de aristas de la retícula o por la traza del Laplaciano en lugar de `sheet.numel()`, facilitando la portabilidad del hiperparámetro entre matrices de distinto tamaño.
- [x] **5.2. Baselines de comparación clave**:
  - Implementar baseline de rotación ortogonal aleatoria / Hadamard (estilo QuIP) como control para desacoplar la dispersión de información de la suavidad espacial.
  - Comparar frente a truncamiento SVD a igual bpp.
  - Reportar métricas con múltiples semillas ($n \ge 3$) e intervalos de confianza.
- [x] **5.3. Script de reproducibilidad end-to-end**:
  - Crear un script único (`reproduce_all.py` o similar) que genere los gráficos y tablas del paper desde cero.

---

## 4. Estado de Avance

| Fase | Tareas Totales | Completadas | Estado |
| :--- | :---: | :---: | :--- |
| **Fase 1: Código Inmediato, Seguridad y Paridad** | 4 | 4 | ✅ Completada (2026-10-01) |
| **Fase 2: Consistencia de Cuantizador y BPP** | 2 | 2 | ✅ Completada (2026-10-01) |
| **Fase 3: Experimento de Falsificación Real** | 2 | 2 | ✅ Completada (2026-10-01) |
| **Fase 4: Coherencia Documental y Claims** | 3 | 3 | ✅ Completada (2026-10-01) |
| **Fase 5: Metodología y Baselines Avanzados** | 3 | 3 | ✅ Completada (2026-10-01) |

*Todas las fases del plan de remediación han sido completadas con éxito.*
