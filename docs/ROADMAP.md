# Roadmap: Dirichlet Regularization — Rename & Publication

**Fecha:** 2026-10-01  
**Estado:** En progreso

---

## Fase 0: Rename & Refocus (Ahora)

### 0.1 Infraestructura del Rename
- [x] Renombrar repo en GitHub: `topographic-transformers` → `dirichlet-regularization`
- [ ] Renombrar directorio local
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

### 0.3 Actualizar Whitepaper
- [x] Actualizar referencias al repo (verificado limpio)
- [ ] Mover enfoque: de "Edge AI whitepaper" a "Supplementary Technical Report"
- [ ] Mantener contenido intacto (es evidencia valiosa)

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
- [ ] Configuración: 10M y 20M params, BPE tokenizer
- [ ] Entrenamiento largo: 10K-30K steps en A100 (Modal.com)
- [ ] Evaluar PPL pre/post cuantización con convergencia real

### 2.2 Experimento Beyond-Transformers
- [ ] MLP clasificador en MNIST o CIFAR-10 con `DirichletLoss`
- [ ] Medir: ¿la regularización mejora la compresibilidad del MLP?
- [ ] Tabla comparativa: estándar vs topográfico bajo cuantización

### 2.3 Matriz de Ablación
- [ ] `topo_lambda`: 0, 0.001, 0.005, 0.01, 0.02, 0.05
- [ ] Radios de banda (r0, r1, r2): variaciones del default
- [ ] Profundidad (L): 4, 6, 8, 12
- [ ] Ancho (d_model): 128, 256, 384, 512
- [ ] Block-DCT tile size (B): 32, 64, 128, global

### 2.4 Comparación con SOTA de Cuantización
- [ ] GPTQ (3-bit, 4-bit)
- [ ] AWQ (4-bit)
- [ ] QuIP# (2-bit) — competidor más directo
- [ ] Tabla: PPL, tamaño en disco, RAM activa, throughput

### 2.5 Visualizaciones
- [ ] Heatmaps 2D-DCT: topográfico vs estándar
- [ ] Curvas de energía acumulada vs radio espectral
- [ ] Visualización 3D de superficie cortical de pesos

---

## Fase 3: Paper (2-3 semanas)

### 3.1 Preparación
- [ ] Elegir venue (recomendado: TMLR, rolling, sin deadline)
- [ ] Contactar UPV/VRAIN para endorsement/coautoría
- [ ] Preprint en arXiv para establecer prioridad

### 3.2 Estructura del Paper
```
1. Introduction
   - Weight matrices as unstructured noise: the compression barrier
   - Key insight: spatial regularization enables spectral compressibility
   - Contributions

2. Background & Related Work
   - Quantization (GPTQ, AWQ, QuIP#, SqueezeLLM)
   - Topographic maps in neuroscience
   - Spectral methods in ML

3. Method: Dirichlet Harmonic Regularization
   3.1 From L2 Decay to Dirichlet Smoothness
   3.2 Spectral Energy Concentration (2D-DCT)
   3.3 Hierarchical Band Quantization & Base-3 Trit Packing
   3.4 Streaming JIT with Ping-Pong DMA (application)

4. Experiments
   4.1 Falsification: topographic vs standard under 0.945 bpp
   4.2 TinyStories at scale (10M, 20M params)
   4.3 Beyond Transformers: MLP classification
   4.4 Comparison with GPTQ / AWQ / QuIP#
   4.5 Ablation: lambda, band radii, block size

5. Analysis
   5.1 Spectral energy concentration (visualization)
   5.2 The D³ vs D² asymmetry law
   5.3 Block-DCT tiling

6. Limitations & Future Work
7. Conclusion
```

### 3.3 Figuras Clave (mínimo 5)
- [ ] Fig 1: Concepto — L2 decay vs Dirichlet smoothness (heatmaps)
- [ ] Fig 2: 2D-DCT spectral heatmaps (topographic vs standard)
- [ ] Fig 3: Falsification bar chart (PPL collapse)
- [ ] Fig 4: PPL vs bit-rate (TopoSpec vs GPTQ vs AWQ vs QuIP#)
- [ ] Fig 5: SRAM footprint vs model scale (edge application)

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
