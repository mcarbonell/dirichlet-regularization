# Auditoría 2 — Dirichlet Regularization (`dreg`)

**Fecha:** 2026-10-01
**Tipo de revisión:** revisión técnica independiente del repo en `284f499`
**Alcance:** `README.md`, `paper/paper-draft.tex`, `dreg/` (topology, spectral, quantization, model), `kernel/`, `examples/`, `docs/ROADMAP.md`, CI y artefactos.

---

## 0. Resumen ejecutivo

**La idea es buena y publicable.** Es simple (un gradiente que toca 4 vecinos), barata, y el efecto empírico es real y reproducible. El código está limpio, testeado y con CI.

**El riesgo no es que la hipótesis sea falsa.** El riesgo es que el *framing* actual invite a que un revisor la desmonte con cuatro objeciones principales:

1. "Esto es un Laplacian prior de los 90" (prior art insufficientemente situado).
2. "Tu 0.945 bpp solo cuenta matrices lineales 2D; los embeddings siguen en FP16."
3. "El codec base-3 no ahorra entropía; los bits vienen del truncamiento."
4. "La escala empírica es de juguete (TinyStories-10M, MNIST-MLP)."

Ninguna de las cuatro es fatal **si se abordan antes de la submission**. Todas son vulnerables en la redacción actual.

**Veredicto por eje:**

| Eje | Nota | Comentario |
| :--- | :---: | :--- |
| Novedad del regularizador | ⚠️ 5/10 | El mecanismo es Laplacian smoothing; la novedad real es la *elección de gauge* |
| Novedad del codec base-3 | ⚠️ 4/10 | La ventaja de bits viene del truncamiento, no del base-3 |
| Calidad del código | ✅ 8/10 | Limpio, 57 tests, CI, artefactos abiertos |
| Calidad de las pruebas empíricas | ⚠️ 5/10 | Escala pequeña, baselines débiles, una métrica no-canónica |
| Disciplina / reproducibilidad | ✅ 9/10 | Roadmaps honestos con números reales; esto es raro y bueno |
| Potencial de aceptación (TMLR) | ⚠️ 6/10 | Viable tras reframe + escalado + baselines SOTA |

---

## 1. La idea: qué es sólido y qué hay que re-enmarcar

### 1.1 Lo que es genuinamente bueno

El núcleo — penalizar la energía de Dirichlet discreta

$$\mathcal{E}_D(W) = \tfrac{1}{2}\,\mathrm{Tr}(W^\top L W) = \tfrac{1}{2}\sum_{(u,v)\in\mathcal{E}}\|w_u - w_v\|^2$$

sobre la matriz de pesos para forzar *decay* espectral en el dominio 2D-DCT — es **sano y defendible**. Es una regularización de un solo hiperparámetro, sin coste computacional apreciable, que se puede añadir a cualquier loop de entrenamiento existente en dos líneas.

### 1.2 El punto fuerte real: ruptura de invariancia de gauge

**Este es el argumento que hay que convertir en la contribución central del paper.**

Una red opera sobre pesos con **invariancia de gauge / permutación**: reordenar filas o columnas de $W$ no cambia la función $f_\theta$. Por tanto, la "estructura espacial" de $W$ en el layout $(out\_features, in\_features)$ **no corresponde a nada en el modelo funcional** — es un gauge arbitrario.

La observación valuable no es *"suavizar es bueno"*, sino:

> **En redes profundas la topología de $W$ no es invariante de nada, luego es un espacio de diseño infraexplotado. Fijar ese gauge (haciendo $W$ una retícula suave) es una elección de diseño legítima y explotable, y tiene consecuencias medibles en compresibilidad.**

Esto es original, es un argumento conceptual limpio, y conecta con la neurociencia (mapas corticales) sin depender de ella.

### 1.3 Prior art que falta citar / discutir

El paper actual no sitúa su obra respecto a:

- **Laplacian / smoothness priors** en visión y en parámetros (clásico; *deep image prior* de Ulyanov et al.).
- **Implicit spectral bias** de SGD/Adam: el gradiente desciende preferentemente hacia direcciones de baja frecuencia (literatura de *implicit bias of gradient descent*).
- **Decay espectral ya observado en matrices de pesos** entrenadas (Frangakis & Arlot y sucesores, *spectral bias of deep networks*). El resultado propio no "descubre" que el espectro decae: muestra que la regularización lo hace **más decayido y controlable**.
- **Regularización por norma nuclear / low-rank bias** en pesos (ver §2.1a — es el competidor más peligroso).

**Acción:** añadir una subsección de positioning en Related Work que sea explícita sobre esto. Reconocerlo **antes** de que lo haga el revisor convierte un punto débil en señal de rigor.

### 1.4 Problema de nomenclatura: "Dirichlet" es mala elección

La energía de Dirichlet no tiene relación con:

- *priors de distribución* de Dirichlet (un lector de IA generativa pensará en un prior bayesiano),
- *problemas de contorno* (Dirichlet boundary conditions), que es de donde viene la analogía.

Riesgos: confusión, y pérdida de credibilidad en el nombre del repo.

**Propuesta:** `laplacian-smoothness` / `topographic-smoothing`. Alternativamente, conservar "Dirichlet (harmonic) energy" **definiéndola explícitamente en la primera aparición** y usando la analogía con campos con condiciones de contorno **una sola vez**.

**Nota:** un rename completo del repo es costoso ahora (paper, URLs, repo público, badges). Decisión: **no bloqueante**, pero deseable antes de la submission.

---


---

## 2. Los cinco ataques previsibles del revisor

### (a) "La suavidad sesga a rango bajo → pagas con capacidad"

**Real**, y el propio paper lo admite en §5.1 ("atractor de fijación rango-1"). Pero esconderlo en "falsación" es peor que medirlo.

**Debe aparecer como tabla principal:**

- **Effective rank** / entropía de Schmidt / número de valores singulares significativos vs. λ.
- **Ablations contra regularizadores equivalentes en presupuesto:**
  - weight decay (AdamW) con doble budget,
  - TV-L1 sobre $W$ (variación total),
  - penalización de norma nuclear,
  - **SVD truncation como cuantizador competidor**.
- **Adam vs. AdamW con λ y steps iguales.** Si AdamW con doble presupuesto iguala tu resultado cuantizado, la contribución se evapora.

**Hipótesis a falsificar explícitamente:** *"el cuantizador espectral es neutral; lo que cambia el resultado es la energía disponible en bajas frecuencias"*. Para ello, comparar contra **SVD-truncation a igual bpp** sobre el mismo modelo. Si SVD da PPL 15 y tu cuantizador da 14, el cuantizador es el héroe y el regularizador es un *enabling trick*. Eso es un split de contribuciones **honesto y defendible**.

### (b) "0.945 bpp solo cuenta matrices 2D; los embeddings siguen en FP16"

Verdad. El abstract y el README presentan el número como global.

- Calcular y reportar el **bpp end-to-end real del checkpoint** (embeddings 50k×512 sin cuantizar + LayerNorm + biases en FP16 dominan; en un transformer real sale fácil a **3–5 bpp**).
- **No es fatal, pero no lo ocultes:** ponerlo en el abstract. Esto también rompe parcialmente el pitch de edge (<1 MB de SRAM), porque las tablas de embeddings dominan la memoria.

### (c) "El codec base-3 no ahorra entropía"

Debilidad real en `dreg/quantization.py`. La cuenta:

- 5 trits balanceados por byte → $3^5 = 243 \le 256$ → 1.60 bits/trit.
- Pero los trits de banda 2 se generan con umbral en $0.5\sigma$ ⇒ **~50% de ceros** ⇒ distribución **no uniforme**.
- Shannon da $\approx 1.30$ bits/trit, no 1.60.
- Es decir: **el empaquetado base-3 empeora el presupuesto**; los bits que se ganan vienen del **truncamiento de banda 3** (banda 3 → 0 bpp), no del codec.

**Un revisor lo ve en dos líneas de aritmética.**

**Acciones:**

1. Reescribir el base-3 como *"demostración de hardware de un codec de trits con LUT O(1) de 1.25 KB"*, no como ganancia de bits.
2. **Añadir entropy coding** (Huffman/ANS sobre el mapa radial, que es *shared* y se paga una sola vez por tensor) y reportar la **curva completa PPL vs bpp**, no un punto.
3. Si bajas a **~0.5 bpp con la misma calidad**, el paper se vuelve mucho más fuerte y el base-3 pasa a ser implementación de referencia.

### (d) Escala insuficiente

TinyStories-10M / 82M tokens y un MLP en MNIST es **demasiado pequeño para TMLR**. Mínimo a añadir:

- **Pythia-160M o GPT-2-124M**, λ ∈ {0, sweep}, 1–3B tokens, eval en Wikitext-2 / HellaSwag.
- **AWQ / GPTQ a 3 y 4 bpp** como referencia. Si AWQ-3bit da PPL 13 y tú a 0.945 bpp das 14, el claim defendible es:

  > *"Igualamos calidad de 3-bit con 3× menos bits y 2 líneas de regularización."*

  Eso **sí** es publicable. *"Sub-1bpp preserva"* no lo es, porque nadie puede cuantizar a 0.945 bpp sin regularizar de antemano.

- En MNIST: reportar **accuracy real bajo podado espectral**, no *"spectral retention +11%"* (que no es una métrica de calidad de modelo).

### (e) λ no portable entre anchuras

En `dirichlet_energy_2d` la energía se normaliza por `sheet.numel()`. Eso hace que **λ no sea transferible**: la misma "fuerza" de suavizado requiere λ distintos en `d_model=512` que en `d_ffn=2048`.

**Fix de 2 líneas:** normalizar por $(M+N)$ o por el grado medio de la malla, y reescalar λ en consecuencia.

**Consistencia paper ↔ código:** el paper define $\frac{\lambda}{2}\mathrm{Tr}(W^\top L W)$; el código divide por `numel`. **No coinciden** — un revisor atento lo detecta.


## 3. Bugs y detalles de código (leídos, no ejecutados)

| # | Archivo:línea | Detalle | Impacto | Acción |
| :-- | :--- | :--- | :--- | :-- |
| 1 | `dreg/quantization.py:221` | `eval(cfg_str)` en `TritQFormat.load` | **RCE en deserialización.** Se ejecuta código arbitrario al cargar un `.tritq` no confiable | **Reemplazar por `ast.literal_eval` o `json`** (issue de seguridad real, no estilístico) |
| 2 | `dreg/quantization.py:81-87, 109, 141` | Banda 2: umbral en `0.5·σ` pero dequant escala por `σ` (no por el valor típico de un trit) | Error de reconstrucción **sistemáticamente enorme** en esa banda | Ajuste **least-squares** del scale (o almacenar min/max). **Probablemente el win más barato del repo** — baja el error de esa banda por un orden de magnitud y mejora el PPL del modelo topográfico |
| 3 | `dreg/quantization.py:44-50` | Máscara radial normalizada `u/m, v/n` | En `ffn` (2048×512) la anisotropía hace que las bandas no sean equivalentes en ambas direcciones | Bandas **elípticas**; reportar energía retenida **por tipo de matriz** (qkv / ffn / output). Apuesta: ffn domina el resultado |
| 4 | `dreg/topology.py:54` | Normalización por `numel` | λ no portable (ver §2e) | Normalizar por $(M+N)$ o $d$ |
| 5 | `dreg/quantization.py` (banda 3) | La banda 3 se reporta como "0 bpp" pero **nunca se serializa** el conteo de runs de ceros | Se descarta información gratis | **Run-length encoding** de la máscara radial (que es *shared* por construcción) → bpp mucho menor con el mismo modelo |
| 6 | `dreg/quantization.py:66-77` | Per-band `min`/`max` sobre la máscara radial: si una banda queda vacía, `.min()`/`.max()` sobre tensor vacío lanza | Robustez | Guardar un check `numel() == 0` |
| 7 | `README.md:119` | "10×–34× **lossless** compression" | **No es lossless**: hay truncamiento de banda 3 + cuantización de bandas 0-2 | Corregir a *"near-lossless"* o *"quality-preserving"*; es una afirmación falsable en revisión |

### Lo que está bien (no tocar)

- Estructura del paquete limpia y bien separada (`topology` / `spectral` / `quantization` / `model`).
- `spectral.py` está bien hecho: DCT ortonormal, cache, Parseval, block tiling con einsum correcto.
- 57 tests, CI con matrix Windows + Ubuntu, ROCODEs correctos.
- ROADMAP con **números reales**, incluidos resultados negativos/medianos (ej. λ=0.01 "gradiente despreciable"). Esto es **muy raro** y es una fortaleza de credibilidad.
- Hábito de "falsification test" contra la propia hipótesis: la actitud correcta.

---

## 4. Problemas de empaquetado / estrategia de publicación

El paper actual defiende **tres tesis simultáneas**:

1. Regularizador de suavidad espectral (contribución científica).
2. Formato de cuantización `.tritq` base-3 (contribución de formato).
3. Kernel C zero-copy con DMA para microcontroladores (contribución de ingeniería/sistemas).

En 12 páginas eso es **dispersión garantizada**: cada tesis sale superficial y ninguna convince.

**Split recomendado:**

- **Paper principal (TMLR):** el *regularizador* (gauge-breaking Laplacian smoothness) + experimentos serios. Es la tesis con más probabilidad de aceptación y la que escala mejor.
- **Codec base-3 + kernel C:** sección de implementación corta (1.5–2 páginas) o **appendix**. Un formato binario + LUT + streaming kernel son contribuciones de *reproducible engineering*, no hallazgos científicos.
- **Edge / microcontroladores:** si se quiere expandir, repo aparte o demo; no cuerpo del paper.

**Sobre el pitch de edge:** hoy se presenta el kernel C con throughput medido en CPU de escritorio, no en silicio real. Eso es un gap de credibilidad que se salva rediciendo el claim a *"diseño de kernel y estimación de SRAM"* en vez de *"rendimiento en Cortex-M55"*.

---

## 5. Plan de acción priorizado

### Fase A — Fixes de 1 día (alto retorno, bajo riesgo)

1. `ast.literal_eval` en `TritQFormat.load`. *(seguridad)*
2. Escala de banda 2 por least-squares. *(calidad de reconstrucción)*
3. Normalización portable de λ + consistencia paper↔código en la fórmula.
4. Guardas de tensor vacío en el cuantizador.
5. Corregir "lossless" → "near-lossless" en el README.
6. Run-length encoding de la banda 3 (drop trivial, bpp mejor).

**Cada uno de estos mejora los números directamente.** Empezar aquí.

### Fase B — Tablas de ablación que blindan la tesis

1. **Effective rank / entropía de Schmidt vs. λ.**
2. **vs. TV-L1**, **vs. norma nuclear**, **vs. weight decay (AdamW budget-matched)**.
3. **SVD-truncation a igual bpp** como cuantizador competidor. *(¿Gana el cuantizador o el regularizador?)*
4. **Curva PPL vs bpp** completa (barrido de $r_0/r_1/r_2$ y del bit-rate objetivo), no un punto.
5. **Energía retenida por tipo de matriz** (qkv / ffn / out / embed) — ataca el sesgo de la máscara radial (#3 de §3).

**Punto de decisión:** si las ablaciones 2/3 invalidan la tesis (p. ej. AdamW budget-matched iguala al cuantizado), replantear el paper a *"sparsity/structure discovery"* en vez de compresión. **Esta es la Fase que decide si el paper se sostiene.** No escribir más paper antes de hacerla.

### Fase C — Escala y baselines SOTA

1. Pythia-160M o GPT-2-124M, λ ∈ {0, sweep}, 1–3B tokens, Wikitext-2 / HellaSwag.
2. AWQ y GPTQ a 3 y 4 bpp como referencia. Reformular el claim como *"calidad 3-bit a 3× menos bits"*.
3. MNIST: reportar accuracy real, no solo spectral retention.

### Fase D — Packaging

1. Separar las 3 tesis (§4).
2. Positioning explícito de prior art (§1.3).
3. Decidir el rename del término "Dirichlet" (§1.4).
4. Reproducibility script end-to-end (ya está en el ROADMAP, pendiente).
5. Un único `reproduce.sh` / `make all` que produce todas las tablas y figuras.

---

## 6. Veredicto final

**La idea es buena y publicable.** El efecto es real, el mecanismo es simple y elegante, y la disciplina de ingeniería del repo es buena.

El trabajo que queda no es "investigar más", sino **re-enmarcar honestamente**:

- Presentar la ruptura de gauge como la idea central.
- Medir el coste en capacidad (effective rank) en vez de esconderlo.
- Reportar el bpp end-to-end real.
- Aceptar que el base-3 es demo de hardware, no ganancia de entropía.
- Escalar a un modelo donde los baselines SOTA sean comparables.

Si se ejecutan las Fases A–C, la probabilidad de aceptación en TMLR pasa de "frágil" a "sólida". Sin la Fase B (ablations), el paper está en riesgo real de rechazo por las objeciones (a) y (c), que son las más baratas de formular para un revisor.

---

*Fin de la auditoría 2.*
