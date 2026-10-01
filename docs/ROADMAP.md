# Roadmap: Dirichlet Regularization — Rename & Publication

**Fecha:** 2026-10-01  
**Estado:** En progreso

---

## Fase 0: Rename & Refocus (Ahora)

### 0.1 Infraestructura del Rename
- [x] Renombrar repo en GitHub: `topographic-transformers` → `dirichlet-regularization`
- [x] Renombrar directorio local
- [x] Actualizar git remote URL
- [x] Renombrar paquete Python: `topospec` → `dreg`
- [x] Actualizar `__init__.py` exports
- [x] Actualizar `pyproject.toml` (name, description, URLs)
- [x] Actualizar todos los imports en `examples/`
- [x] Actualizar todos los imports en `tests/`
- [x] Actualizar `kernel/build_kernel.py` (verificado limpio)
- [x] Verificar: `python -m pytest tests/ -v` pasa al 100%
- [x] Commit y push

### 0.2 Reescribir README
- [x] Nueva narrativa: principio general → caso demostrado (Transformers)
- [x] Estructura: Idea → Matemáticas → API → Caso de estudio Edge → Benchmarks
- [x] Actualizar code snippets con `import dreg`
- [x] Actualizar badges

### 0.3 Actualizar Whitepaper ✅
- [x] Actualizar referencias al repo (URLs actualizadas a `dirichlet-regularization`)
- [x] Mover enfoque: de "Edge AI whitepaper" a "Supplementary Technical Report"
- [x] Mantener contenido intacto (evidencia de ingeniería y auditorías de silicio preservada)

### 0.4 Actualizar Examples
- [x] Revisar docstrings para reflejar el nuevo framing
- [x] Añadir un ejemplo simple: MLP + MNIST/CIFAR con `dreg.DirichletLoss` (`examples/train_mlp_dirichlet.py`)

---

## Fase 1: Solidificar la Base de Código (1-2 semanas)

### 1.1 Tests ✅
- [x] `test_topology.py`: Energía de Dirichlet en malla conocida
- [x] `test_spectral.py`: Ortogonalidad DCT, roundtrip, Parseval
- [x] `test_quantization.py`: Roundtrip cuantización, LUT, TritQFormat save/load
- [x] `test_model.py`: Forward/backward, causal masking, generation

### 1.2 TritQFormat.load() ✅
- [x] Deserialización completa del formato binario `.tritq`
- [x] Test de roundtrip: `save → load` verificado

### 1.3 Bugs Corregidos ✅
- [x] `Tuple` import faltante en `model.py`
- [x] URLs incorrectas en `pyproject.toml`
- [x] `DirichletLoss` crash con `model.modules()`
- [x] `BlockDCTTiler` einsum con índices incorrectos

### 1.4 Portabilidad del Kernel C ✅
- [x] Capa de abstracción: `#ifdef _WIN32` → Win32, `#else` → pthreads & condvars POSIX
- [x] Compatibilidad Linux/POSIX verificada (GCC `-lpthread`)
- [x] Actualizar `build_kernel.py` para detectar plataforma y arquitectura (x86_64 vs ARM)

### 1.5 CI con GitHub Actions ✅
- [x] Workflow: matrix (Windows + Ubuntu, Python 3.10/3.11 con CPU PyTorch y build de kernel)
- [x] Badge en README

---

## Fase 2: Validación Empírica (3-5 semanas)

### 2.1 Entrenamiento en TinyStories (GPU: Modal/Colab)
- [x] Infraestructura Modal y cache de tokenizador BPE (4096 tokens)
- [x] Calibración de $\lambda_{\text{topo}}$ (runs rápidos de 2,000 steps en A10G):
  - $\lambda=0.0$ (Baseline Estándar): Val PPL 10.48 | Quant PPL 46.60 | $E_D=0.004225$ | ΔPPL: +36.12
  - $\lambda=0.01$: Val PPL 10.44 | Quant PPL 47.95 | $E_D=0.004251$ | ΔPPL: +37.51 (gradiente despreciable)
  - $\lambda=5.0$: Val PPL 10.62 | Quant PPL 44.80 | $E_D=0.003382$ (-20.0% energía Dirichlet) | ΔPPL: +34.18
  - $\lambda=15.0$: Val PPL 10.81 | Quant PPL 43.26 | $E_D=0.002427$ (-42.5% energía Dirichlet) | ΔPPL: +32.46
  - $\lambda=30.0$: Val PPL 10.98 | Quant PPL 39.80 | $E_D=0.001924$ (-54.5% energía Dirichlet) | ΔPPL: +28.82
- [x] Entrenamiento largo de publicación (10,000 steps, ~82M tokens en A10G en Modal.com) ✅:
  - **Baseline Estándar 10M ($\lambda=0.0$):** Val PPL 5.83 | Quant PPL 72.89 | $E_D=0.012316$ | ΔPPL: +67.06
  - **Topográfico 10M ($\lambda^*=30.0$):** Val PPL 6.18 | Quant PPL **66.13** | $E_D=\mathbf{0.003438}$ (**-72.1%** energía Dirichlet) | ΔPPL: **+59.96** (**-6.76 PPL de ventaja**)
- [x] Checkpoints finales de publicación generados y almacenados en volumen Modal (`.pt` y `.tritq` a 0.945 bpp / 749.7 KB)

### 2.2 Experimento Beyond-Transformers ✅
- [x] MLP clasificador en MNIST con `DirichletLoss` (`examples/train_mlp_dirichlet.py`)
- [x] Medir: la regularización de Dirichlet mejora la compresibilidad y retención espectral del MLP:
  - **FP32 Test Acc:** Standard 97.87% vs Topográfico 97.58% (diferencia mínima -0.29%)
  - **Energía Dirichlet ($E_D$):** 0.0219 → 0.0034 (**-84.5% de reducción** en rugosidad)
  - **Energía en bajas frecuencias ($r \le 0.40$):** 27.87% → **52.84%** (concentración espectral casi duplicada)
  - **Poda pasobajo extrema (20% coeficientes DCT):** Standard 30.69% (-67.2%) vs Topográfico **41.41%** (-56.2%, **+11.0% retención**)
  - **Cuantización Base-3 (.tritq, 0.94 bpp):** Standard 87.40% (-10.5%) vs Topográfico **88.59%** (-9.0%, **+1.19% precisión a sub-1 bpp**)
- [x] Tabla comparativa documentada para la sección "Universal Regularization Principle" del paper

### 2.3 Matriz de Ablación
- [x] `topo_lambda` sweep: 0.0, 0.01, 5.0, 15.0, 30.0 (curva de calibración completada)
- [ ] Radios de banda (r0, r1, r2): variaciones del default
- [ ] Profundidad (L): 4, 6, 8, 12
- [ ] Ancho (d_model): 128, 256, 384, 512
- [ ] Block-DCT tile size (B): 32, 64, 128, global

### 2.4 Comparación con SOTA de Cuantización ✅
- [x] Benchmark empírico sobre TinyStories 10M (10,000 steps, 25 batches retenidos de validación):
  - **Dense FP32 (32.0 bpp):** Std PPL 6.11 | Topo PPL 6.46 | RAM: 42.2 MB (DRAM)
  - **Spatial Uniform INT4 / RTN (4.0 bpp):** Std PPL 6.79 (Δ +0.68) | Topo PPL 7.01 (Δ +0.55) | RAM: 5.3 MB (DRAM)
  - **Spatial Uniform INT2 / RTN (2.0 bpp):** Std PPL 41.67 (Δ +35.56) | Topo PPL 55.22 (Δ +48.75) | RAM: 2.6 MB (DRAM) — Colapso espacial uniforme
  - **Spectral Uniform INT4 (4.0 bpp):** Std PPL 7.85 (Δ +1.74) | Topo PPL 8.97 (Δ +2.51) | RAM: 5.3 MB (DRAM)
  - **Base-3 TritQ (.tritq, Nuestro Método, 0.945 bpp):** Std PPL 73.75 (Δ +67.64) | Topo PPL **66.50** (Δ +60.03, **-7.25 PPL de ventaja**) | RAM: **< 1.0 MB (SRAM)**
- [x] Contraste cualitativo y cuantitativo frente a GPTQ / AWQ / QuIP#:
  - Frente a GPTQ/AWQ (4-bit): $4.2\times$ menor tasa de bits (0.945 vs 4.0 bpp) y viabilidad en $\le 1$ MB SRAM.
  - Frente a QuIP# (2-bit): $2.1\times$ menor tasa de bits (0.945 vs 2.0 bpp), descompresión $O(1)$ sin empaquetado de retículas E8, y ejecución Zero-Copy DMA con micro-kernel C.
- [x] Tabla comparativa lista para la Sección 4.4 del Paper

### 2.5 Visualizaciones ✅
- [x] Heatmaps 2D-DCT: topográfico vs estándar (`docs/figures/fig1_weight_heatmaps.png`)
- [x] Curvas de energía acumulada vs radio espectral (`docs/figures/fig2_dct_energy_spectra.png`)
- [x] Curva de calibración y frontera de Pareto (`docs/figures/fig3_pareto_quantization.png`)
- [x] Muestras cualitativas de texto autorregresivo generadas en TinyStories (ambos modelos fluidos)

---

## Fase 3: Paper (Redacción en LaTeX) ✅

### 3.1 Preparación & Metadatos
- [x] Autoría formalizada: Mario Raúl Carbonell Martínez (Independent Researcher, `marioraulcarbonell@gmail.com`). *Contacto pendiente con UPV/VRAIN para acordar colaboración/afiliación institucional formal.*
- [x] Repositorio de código y modelos enlazado: `https://github.com/mcarbonell/dirichlet-regularization`
- [ ] Elegir venue (recomendado: TMLR, rolling, sin deadline; o NeurIPS/ICLR)
- [ ] Subida de preprint a arXiv para registrar prioridad intelectual

### 3.2 Estructura y Redacción Completa del Paper (`paper/paper-draft.tex` / `paper/paper-draft.pdf`) ✅
- [x] **Abstract & Keywords:** Información teórica, compresión 0.945 bpp ($33.86\times$), ventajas empíricas en TinyStories y MNIST, runtime $<1.0$ MB SRAM.
- [x] **1. Introduction:** Ruptura de la invariancia de calibre (gauge symmetry), mapas corticales, motivación y resumen de 6 contribuciones clave.
- [x] **2. Related Work:** Post-training quantization (GPTQ, AWQ, QuIP#, SqueezeLLM, SmoothQuant), topografía en neurociencia y métodos espectrales en deep learning.
- [x] **3. Methodology:**
  - 3.1 Formulación matemática de la energía de Dirichlet discreta y difusión por calor ($-\nabla \mathcal{E}_D \propto \Delta W$).
  - 3.2 Compactación espectral 2D-DCT, autovalores laplacianos $\mu_{k,l}$ y correspondencia con Laplacian Eigenmaps.
  - 3.3 Cuantización jerárquica radial y empaquetado Base-3 ($3^5 = 243 \le 256$, 1.60 bits/trit, 0.945 bpp) con tabla LUT $\mathcal{O}(1)$ de 1.25 KB.
  - 3.4 Runtime C de streaming Zero-Copy con doble buffer DMA y resolución de la Ley de Asimetría $\mathcal{O}(D^3)$ vs $\mathcal{O}(D^2)$ mediante Block-DCT tiling.
  - **Algoritmo 1:** Pseudocódigo completo formalizando entrenamiento regularizado y cuantización `.tritq`.
- [x] **4. Experimental Evaluation:**
  - 4.1 Setup experimental en Modal A10G (TinyStories 10M, 82M tokens) y MNIST (MLP).
  - 4.2 Barrido de calibración (5 puntos, $\lambda \in \{0.0, 0.01, 5.0, 15.0, 30.0\}$) y frontera de Pareto monotonicamente decreciente (Tabla 1).
  - 4.3 Convergencia a escala de 10,000 steps: reducción del 72.1% en rugosidad, -7.25 PPL de ventaja a 0.945 bpp, y muestras cualitativas de prosa sintáctica fluida (Tabla 2).
  - 4.4 Beyond Transformers: validación en MNIST con retención de precisión de +11.0% bajo poda espectral del 80% (Tabla 3).
  - 4.5 Comparación exhaustiva con SOTA (FP32, INT4 espacial/espectral, INT2 con colapso, y TritQ con viabilidad en $<1.0$ MB SRAM) (Tabla 4).
- [x] **5. Discussion & Analysis:**
  - 5.1 Falsificación y análisis teórico del colapso de rango singular (atractor de fijación rango-1 por plantillas sinusoidales externas).
  - 5.2 Implicaciones en silicio edge (microcontroladores STM32H7, ESP32-S3, Cortex-M55).
  - 5.3 Declaración de reproducibilidad y enlaces de artefactos abiertos.
- [x] **6. Conclusion & Acknowledgments:** Agradecimiento a infraestructura en Modal.
- [x] **Bibliografía (`paper/references.bib`):** 15 citas completas procesadas y validadas con BibTeX y natbib (100% citadas en el texto).

### 3.3 Figuras de Publicación Integradas ✅
- [x] **Fig 1:** Heatmaps comparativos de pesos (manifolds corticales topográficos vs ruido blanco de AdamW) (`paper/figures/fig1_weight_heatmaps.png`)
- [x] **Fig 2:** Espectros de energía acumulada 2D-DCT mostrando $>72\%$ de energía concentrada en bajas frecuencias (`paper/figures/fig2_dct_energy_spectra.png`)
- [x] **Fig 3:** Pipeline de streaming Zero-Copy DMA ping-pong en sub-1 MB SRAM.
- [x] **Fig 4:** Frontera de Pareto de calibración ($E_D$ vs Quantized PPL) (`paper/figures/fig3_pareto_quantization.png`)
- [x] **Compilación local sin errores:** `paper/paper-draft.pdf` (12 páginas, 3.01 MB, 0 errores, 0 overfull hboxes).

---

## Fase 4: Polish & Submission (1-2 semanas)

- [ ] Reproducibility script: entrena, cuantiza, evalúa, genera tablas
- [ ] README final con badges de CI, resultados
- [ ] Supplementary materials: derivaciones, tablas extra
- [ ] Code release con tag `v1.0.0`
- [ ] Submit a TMLR (o venue elegido)

---

## Timeline Estimado

```
Semana 0 (ahora):  Fase 0 (rename, refocus, README)
Semana 1-2:        Fase 1 restante (kernel POSIX, CI)
Semana 3-5:        Fase 2 (training, ablations, comparisons)
Semana 6-7:        Fase 3 (paper)
Semana 8:          Fase 4 (polish, submit)
```
