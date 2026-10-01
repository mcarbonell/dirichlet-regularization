# Propuesta y Validación: Dirichlet Annealing & Cooldown Schedule

**Fecha de registro:** 2026-10-01  
**Autor:** Mario Raúl Carbonell Martínez  
**Estado:** [SEÑAL EMPÍRICA CONFIRMADA] (PoC validada en MNIST; pendiente de escalado a TinyStories 10M)  
**Clasificación:** Optimización Dinámica de Hiperparámetros / Planificación Curricular  

---

## 1. Motivación e Hipótesis Científica

En el entrenamiento estándar con regularización de Dirichlet (`dreg`), el hiperparámetro de tensión superficial $\lambda_{\text{topo}}$ (o $\epsilon$) se mantiene constante a lo largo de todas las épocas/pasos de optimización:

$$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{task}}(t) + \lambda_{\text{topo}} \cdot \mathcal{E}_D(W(t))$$

Si bien esta formulación estática comprime masivamente la energía espectral en bajas frecuencias (reduciendo la rugosidad espacial $E_D$ en más del $70\%$–$95\%$), impone una pequeña restricción sobre la capacidad expresiva densa en punto flotante (FP32), generando una caída de $\sim 0.3\%$ en precisión o $\sim 0.35$ PPL en lenguaje.

### La Hipótesis del Cooldown Topográfico
> **Hipótesis:** Si se aplica la regularización de Dirichlet durante la mayor parte del entrenamiento ($80\%$) para fijar la geometría global del atractor en una variedad continua, y posteriormente se **relaja o anula** ($\lambda \to 0$) durante la fase final de enfriamiento (*cooldown*):
> 1. La **inercia macroscópica** mantendrá los pesos confinados en la cuenca de suavidad espectral ya inducida.
> 2. La liberación de la tensión superficial permitirá al optimizador realizar micro-ajustes de alta frecuencia para maximizar la precisión en la tarea.
> 3. El resultado superará tanto al baseline como a la regularización estática en el frente de Pareto precisión-cuantización.

---

## 2. Validación Empírica (Proof-of-Concept)

Para contrastar la hipótesis de forma inmediata, se diseñó un protocolo controlado sobre una red feedforward (MLP $784 \to 256 \to 128 \to 10$) en MNIST a 5 épocas con optimizador AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$):

1. **Standard Baseline:** $\lambda = 0.0$ durante las 5 épocas.
2. **Full Dirichlet:** $\lambda = 2\times 10^{-3}$ constante en las 5 épocas.
3. **Annealed Dirichlet (Cooldown):** $\lambda = 2\times 10^{-3}$ en épocas 1 a 4 ($80\%$ del régimen); $\lambda = 0.0$ en la época 5 ($20\%$ final).

### Resultados Medidos

| Régimen de Entrenamiento | Precisión FP32 (Densa) | Precisión Cuantizada (**0.94 bpp**) | Rugosidad Dirichlet ($E_D$) | Retención vs Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Baseline** | **97.83%** | 24.78% | 0.001926 | 0.0% (Ruido blanco desestructurado) |
| **Full Dirichlet** | 96.85% | 25.13% | **0.000089** | **-95.4%** rugosidad |
| **Annealed (Cooldown)** 🌟 | **97.37%** (+0.52%) | 🌟 **28.19%** (**+3.06%**) | 0.000186 | **-90.3% rugosidad** |

### Hallazgos Clave
1. **Recuperación de Capacidad Densa:** El cooldown recuperó $+0.52\%$ de precisión en FP32, recortando la brecha con el baseline no regularizado.
2. **Persistencia de la Memoria Geométrica:** La rugosidad $E_D$ únicamente pasó de $0.000089$ a $0.000186$, permaneciendo un **$90.3\%$ más suave que el baseline estándar**. El optimizador no tiene tiempo ni fuerza para desordenar la matriz hacia ruido blanco.
3. **Sinergia Inesperada en Cuantización:** La precisión cuantizada a $0.94\text{ bpp}$ no solo no se degradó, sino que **aumentó $+3.06\%$ sobre Full Dirichlet y $+3.41\%$ sobre el baseline**, demostrando que los coeficientes armónicos alcanzaron un óptimo de ajuste a la pérdida sin perder la compactación espectral basal.

---

## 3. Formulación Matemática de los Schedulers Propuestos

Para implementaciones a escala de producción y modelos de lenguaje (Transformers), se formulan dos esquemas de planificación de $\lambda(t)$ sobre $T$ pasos totales:

### Esquema A: Step Cooldown (Fase de Enfriamiento Terminal)
Inspirado en la política de dos fases de pre-entrenamiento de LLMs modernos (ej. *Llama 3*, *MiniCPM*):

$$\lambda(t) = \begin{cases} \lambda_0, & t < (1 - \alpha) T \\ 0, & t \ge (1 - \alpha) T \end{cases}$$

donde $\alpha \in [0.10, 0.20]$ es la fracción final de pasos de enfriamiento.

### Esquema B: Cosine Annealing Schedule
Decaimiento suave continuo sincronizado con el scheduler de tasa de aprendizaje:

$$\lambda(t) = \lambda_{\text{min}} + \frac{1}{2}(\lambda_0 - \lambda_{\text{min}}) \left( 1 + \cos\left( \frac{\pi t}{T} \right) \right)$$

---

## 4. Hoja de Ruta para Escalado en Modelos de Lenguaje

- [ ] **Fase 1 (Implementación):** Añadir clase `DirichletScheduler` o soporte de función de decaimiento en `dreg.topology.DirichletLoss`.
- [ ] **Fase 2 (Validación 10M TinyStories):** Ejecutar corrida de 10,000 pasos en Modal con los últimos 1,500 pasos en cooldown ($\lambda \to 0$).
- [ ] **Fase 3 (Medición):** Evaluar si la perplejidad FP32 converge a $\le 5.90$ (cerrando la brecha con los $5.83$ del baseline estándar) mientras la perplejidad cuantizada a 0.945 bpp se mantiene en $\le 65.0$.
- [ ] **Fase 4 (Publicación):** Añadir subsección *"5.2 Topological Annealing Dynamics"* en el manuscrito formal.
