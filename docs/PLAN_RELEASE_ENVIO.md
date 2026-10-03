# Plan de Release y Envío — Dirichlet Regularization (`dreg`)

**Versión del plan:** 2026-10-03  
**Rama de trabajo:** `arena/01a10131-dirichlet-regularization`  
**Último commit auditado:** `8f820c6` — *feat: auditoría P0/P1*  
**Estado:** Listo para PR → `main` → tag `v1.0.0` → arXiv → TMLR  
**Autor del plan:** Agent Mode (Arena.ai) a petición de M. Carbonell

---

## 1. Dónde estamos

| Artefacto | Estado | Notas |
|-----------|--------|-------|
| **Código `dreg/`** | ✅ 79 tests, ruff clean, 96% cov | P0 corregidos (ED factor2, radii, guards, kernel 8) |
| **Paper `paper/paper-draft.tex`** | ✅ Fuente lista (8 ficheros tocados) | PDF necesita recompilar con `pdflatex` local |
| **Auditoría** | ✅ `docs/AUDITORIA_COMPLETA.md` (51KB) | Hallazgos P0-P3 + apéndices de reproducibilidad |
| **CITATION.cff** | ✅ Nuevo | Para GitHub/ Zenodo / arXiv |
| **Results `results/*.json`** | ✅ 4 JSONs 10M/GPT-2 | Honestos, con punto desfavorable ya integrado en paper |

Commits en esta sesión:

```
26b5700 docs: shorten academic paper title in README   (base main)
a2f0590 docs: auditoría completa ...                    (auditoría 729 líneas)
8f820c6 feat: auditoría P0/P1 — correcciones ...        (8 ficheros, 200+ líneas)
```

Próximo commit será este plan (`docs/PLAN_RELEASE_ENVIO.md`).

---

## 2. Cómo llevar el trabajo de Arena a `main` (PR)

Arena trabaja aislado en `arena/01a10131-dirichlet-regularization` para no pisar tu `main`. Tú controlas el merge. **Sí, necesitas crear un PR y aceptarlo en GitHub:**

**Opción A — Desde la UI de GitHub (recomendada, 2 clics):**

1. Push ya hecho: `arena/01a10131-dirichlet-regularization → origin` (hecho).
2. GitHub te muestra banner amarillo: *“Create a pull request for 'arena/...'”* → clic.
3. Base: `main` ← Compare: `arena/01a10131-dirichlet-regularization` → *Create PR*.
4. Título sugerido: `Release v1.0.0 — Dirichlet Regularization (auditoría P0/P1 + paper + CITATION)`
5. Descripción: copia el cuerpo del plan §3 (checklist).
6. *Merge pull request* → *Confirm merge* (squash o merge, a tu gusto).
7. Tras el merge, `main` ya contiene todo. Borra la rama `arena/...` si quieres.

**Opción B — Desde este sandbox con `gh`:**

```bash
gh pr create --base main --head arena/01a10131-dirichlet-regularization \
  --title "Release v1.0.0 — Dirichlet Regularization (auditoría P0/P1)" \
  --body "Incluye auditoría completa, correcciones P0 (Pareto 0.914, radii, ED factor2), CITATION.cff. Ver docs/PLAN_RELEASE_ENVIO.md"
gh pr view --web   # abre preview
```

**Tras el merge**, haz el tag en local:

```bash
git checkout main && git pull origin main
git tag -a v1.0.0 -m "Dirichlet Regularization v1.0.0 — 10M TinyStories, 0.945 bpp .tritq, <1MB SRAM"
git push origin v1.0.0
# GitHub → Releases → Draft new release from tag v1.0.0 → adjunta paper-draft.pdf + results/*.json
```

> **Importante:** El `paper-draft.pdf` en el repo aún es de antes de los P0. Antes de taguear, recompílalo localmente (ver §3.1).

---

## 3. Checklist pre-release (hazlo en orden)

### 3.1 Recompilar el PDF (obligatorio antes de arXiv/tag)

```bash
cd paper
pdflatex paper-draft
bibtex paper-draft
pdflatex paper-draft
pdflatex paper-draft   # 3 pasadas para refs
ls -lh paper-draft.pdf # debe ser ~900KB, 14-15 págs
# verifica que Tabla 6 incluya fila 0.914 bpp y discusión
```

Si no tienes TeX Live local: `sudo apt install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra` o usa Overleaf subiendo `paper/` (compila en 10s).

### 3.2 Sanity local (30s)

```bash
pip install -e .  # si no lo tienes
python -m pytest tests/ -q                # 79 passed
python -m ruff check .                    # All checks passed
PYTHONPATH=. python examples/reproduce_all.py  # 4/4 PASSED, C parity 2e-07
PYTHONPATH=. python examples/evaluate_falsification.py  # debe mostrar gap ED 2.3×
```

### 3.3 Tag y Release en GitHub

- [ ] PR `arena/... → main` mergeado
- [ ] `paper-draft.pdf` recompilado y commiteado (si lo recompilas tras el merge, haz otro commit corto)
- [ ] `git tag v1.0.0 && git push origin v1.0.0`
- [ ] GitHub Release `v1.0.0` con notas: 10M TinyStories 0.945 bpp, 72% ED, <1MB SRAM, link a `docs/AUDITORIA_COMPLETA.md`

### 3.4 arXiv (prioridad intelectual, 1 tarde)

1. **Cuenta arXiv** → *Submit* → categoría `cs.LG` (primary) + `cs.AI`, `cs.ET`/`cs.DC` (cross).
2. Título: *“Inducing Spectral Smoothness in Neural Weight Manifolds via 2D Dirichlet Regularization”* (el del paper).
3. Autores: `Mario Raúl Carbonell Martínez — Independent Researcher — marioraulcarbonell@gmail.com` (+ afiliación VRAIN/UPV si se cierra, si no déjalo así).
4. Abstract: copia el del `paper/paper-draft.tex` (el nuevo de 0.43–0.945 bpp).
5. Source: sube `paper/` como zip (incluye `.tex`, `.bib`, `figures/*.png`). arXiv compila con `pdflatex`.
6. Comments: `15 pages, 4 figures, 79 tests. Code: https://github.com/mcarbonell/dirichlet-regularization Release v1.0.0`
7. Elige *“I want to submit”* → te da `arXiv:2610.xxxxx`. Ese ID ya es citable.

**Versiones:** `v1` puede ser ya con P0; si tras arXiv encuentras typo, sube `v2` con mismo ID.

### 3.5 TMLR (rolling, sin deadline)

* URL: https://openreview.net/group?id=TMLR
* Botón *“TMLR Submission”* → sube PDF + link a OpenReview.
* TMLR valora: reproducibilidad, código, artefactos, honestidad. Adjunta en *supplementary*: `docs/AUDITORIA_COMPLETA.md`, `results/*.json`, link a `CITATION.cff`.
* En *cover letter* enfatiza el mensaje defensivo: *“No proponemos un cuantizador mejor; proponemos entrenar para que cualquier cuantizador espectral funcione en <1 bpp con DMA.”*

Calendario sugerido:

```
Semana 0 (esta semana):  Merge PR + tag v1.0.0 + arXiv v1
Semana 1-2:              Benchmarks opcionales Modal §4 (si tienes créditos)
Semana 3:                arXiv v2 con benchmarks opcionales + envío TMLR
Semana 4+:               Difusión (X, LinkedIn, HN) + workshop NeurIPS/ICLR si quieres visibilidad
```

---

## 4. Benchmarks opcionales en Modal (no bloquean release, pero refuerzan TMLR)

Todos los scripts Modal ya existen en `benchmarks/`. Requieren `pip install modal && modal token new` + `modal run ...`. Coste estimado: 40–60 GPU-h A10G (~50–70$ en Modal para todo el bloque).

### 4.1 Prioridad alta (refuerzan paper, 1-2 días GPU)

| # | Script | Qué hace | Comando | Qué ganas |
|---|--------|----------|---------|-----------|
| **B-1** | `modal_train_annealing.py` | 10k pasos con annealing 15% cooldown (vs 5k PoC actual) | `modal run benchmarks/modal_train_annealing.py --steps 10000 --gpu a10g --lam 30 --cooldown 0.15` | Apples-to-apples vs Tabla 2 (cita +10.82 PPL pero ahora a 10k). Cierra P1-01. |
| **B-2** | `modal_train_tinystories.py` ×3 seeds | Barrido 10k con 3 seeds (42, 100, 1337) para barras ±std | `for s in 42 100 1337; do modal run benchmarks/modal_train_tinystories.py --steps 10000 --gpu a10g --seed $s --lam 30; done` + mismo con `--lam 0` | Tabla 2 con `mean±std` (P1-06). Desactiva crítica de “single seed”. |
| **B-3** | `modal_benchmark_quantization_sota.py` + GPTQ | Añade GPTQ-INT4 (si `auto-gptq` disponible) al SOTA sweep | `modal run benchmarks/modal_benchmark_quantization_sota.py --with-gptq` (necesita parchear script para importar `optimum`/`auto_gptq`) | Responde a “¿y GPTQ?” sin perder narrativa (<1MB vs 4 bpp). |

**Cómo parchear B-3 rápido** (si te lo piden revisores):

```python
# en modal_benchmark_quantization_sota.py, dentro de run_quantization_benchmark():
from transformers import AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer
# cuantiza el modelo 10M a INT4 con 128 muestras TinyStories
```

Si no tienes tiempo, deja B-3 como *future work* (ya está en Limitations) — TMLR lo acepta.

### 4.2 Prioridad media (figuras y curva Pareto fina)

| # | Script | Comando | Salida |
|---|--------|---------|--------|
| **B-4** | `modal_eval_pareto_tinystories.py` | `modal run benchmarks/modal_eval_pareto_tinystories.py --steps 10000` | JSON completo de Pareto (incluido 0.914) para Figura 3 en alta resolución |
| **B-5** | `modal_generate_figures.py` | `modal run benchmarks/modal_generate_figures.py` | Regenera `fig1/fig2/fig3` sin intervención manual (heatmaps, espectros) |
| **B-6** | `modal_eval_tsp_gpt2.py` | `modal run benchmarks/modal_eval_tsp_gpt2.py --steps 300` | Refina §Limitations TSP con más seeds (ya tienes 1 JSON, puedes añadir 2) |

### 4.3 Prioridad baja (nice-to-have)

* **`modal_train_gpt2.py` con más steps** (1000 en vez de 300) para mostrar que incluso con 1k steps ED apenas se mueve — refuerza argumento “pre-training only”.
* **Block-DCT sweep** `B=32,64,128` sobre 10M para apéndice de escalado (ROADMAP Fase 2.3).

**Tip de Modal:** todos los scripts usan `Volume.from_name("dreg-checkpoints")`. Tras cada run, los `results/*.json` quedan en el volumen; descárgalos con `modal volume get dreg-checkpoints /results/... ./results/`.

---

## 5. Notas para el envío

### 5.1 Qué destacar en el cover letter / tweet

* **Ángulo:** *Principio de entrenamiento*, no *cuantizador*. “Hacemos que los pesos sean compresibles.”
* **Demo tangible:** `<1 MB SRAM` + `C DMA` + `74×` es lo que más impresiona a reviewers de sistemas. Muchos papers de cuantización no llegan a silicio real.
* **Honestidad como fortaleza:** Menciona explícitamente el punto `0.914 bpp` desfavorable y el fallo `GPT-2 300 pasos`. Los revisores premian transparencia.

### 5.2 Venues alternativos si TMLR tarda

* **NeurIPS/ICLR Workshop *Efficient Deep Learning* / *Compression*:** deadline ~mayo/octubre, acepta “principle papers”, feedback rápido, networking. Puedes enviar el mismo PDF con 2 págs menos.
* **JMLR / Machine Learning Journal:** más lento, pero te da espacio para derivaciones largas (Apéndice B).

### 5.3 Licencia y citas

* `LICENSE` MIT ya está bien para código.
* Para paper, añade `CITATION.cff` en el Release (ya hecho). GitHub generará botón *Cite this repository*.
* En `README` el badge `Paper-Draft (PDF)` ya apunta a `paper/paper-draft.pdf` — tras tag, considera subir PDF a `Releases` y enlazar estable.

---

## 6. Mantenimiento post-release

* **Issues template:** activa en GitHub `Settings → Features → Issues` + plantilla que pida `python -m pytest` y `dreg` version.
* **CI:** el workflow `ci.yml` ya corre `ruff + pytest + build_kernel` en Ubuntu/Windows × 3.10/3.11. Tras el merge a `main`, verifica que el badge pase (a veces falla por `torch` CPU index).
* **Resultados:** versiona `results/*.json` en cada release (ya están). Para reproducibilidad, añade `results/README.md` con `sha256` de checkpoints si los subes a Hugging Face.

---

## 7. Comandos copiar-pegar (resumen)

```bash
# 1. Crear PR (desde sandbox o GitHub UI)
gh pr create --base main --head arena/01a10131-dirichlet-regularization \
  --title "Release v1.0.0 — Dirichlet Regularization (auditoría P0/P1)" \
  --body "Ver docs/PLAN_RELEASE_ENVIO.md"

# 2. Tras merge a main:
git checkout main && git pull
cd paper && pdflatex paper-draft && bibtex paper-draft && pdflatex paper-draft && pdflatex paper-draft && cd ..
git add paper/paper-draft.pdf && git commit -m "paper: recompila PDF tras P0" && git push
git tag -a v1.0.0 -m "v1.0.0 — dreg" && git push origin v1.0.0

# 3. Modal (ejemplo 10k annealing)
modal run benchmarks/modal_train_annealing.py --steps 10000 --gpu a10g --lam 30 --cooldown 0.15

# 4. Verificación local
python -m pytest tests/ -q && python -m ruff check . && PYTHONPATH=. python examples/reproduce_all.py
```

---

## 8. Contacto y agradecimientos

* Autor: `Mario Raúl Carbonell Martínez — marioraulcarbonell@gmail.com`
* Infra: Modal (A10G), GitHub Actions, `dreg` MIT
* Este plan fue generado como parte de la auditoría `docs/AUDITORIA_COMPLETA.md` (2026-10-03). Para dudas sobre el contenido, abre un issue en el repo etiquetado `question` o responde al hilo de Arena.

¡Suerte con el envío! El trabajo ya está a nivel *publicable*; con el PR + arXiv tienes prioridad intelectual asegurada. Si necesitas que genere el texto exacto de abstract para arXiv/TMLR o el cuerpo del PR, dímelo y lo preparo.

