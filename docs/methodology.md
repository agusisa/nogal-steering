# Metodologia — nogal-steering

## Contexto

Este proyecto sigue a **Evil Ganda** (abliteration). Ambos usan la misma idea geometrica:
los comportamientos de un LLM viven como direcciones en el espacio de activaciones. La
diferencia:

| Proyecto | Direccion objetivo | Operacion | Efecto |
|----------|-------------------|-----------|--------|
| Evil Ganda | Rechazo/refuse | Sustractiva (ortogonalizar pesos) | Modelo pierde capacidad de rechazar |
| nogal-steering | Cualquier comportamiento | Aditiva (sumar vector en inference) | Modelo adopta el comportamiento |

## Pipeline

### 1. Dataset generation
- 150 ejemplos positive + 150 negative por persona
- Generados con Claude Sonnet 4.5 usando template del YAML
- Formato: `{prompt, response}` — el mismo prompt tiene response positive y negative

### 2. Vector extraction (repeng)
- Cargar modelo, capturar activaciones al pasar cada ejemplo
- Para cada capa L: `direction_L = mean(pos_activations_L) - mean(neg_activations_L)`
- Alternativas de agregacion: `mean`, `PCA primer componente`, `SVD top-1`
- Serializar direcciones por capa en `.safetensors`

### 3. Optuna optimization
Hyperparametros:
- **alpha** [0.3, 2.5]: multiplicador global del vector en inference
- **layer_range** [4, 28]: que capas se afectan (capas medias suelen ser mejores)
- **aggregation** {mean, pca_1, svd_top}: como resumir las activaciones

100 trials. Objective: score de LLM-as-judge sobre test prompts.

### 4. Evaluation (LLM-as-judge)
Claude evalua 20 respuestas del modelo con vector aplicado. Criterios especificos por persona
(ej fearful: "suena ansioso?", "mantiene coherencia?", "el miedo se siente natural?").

Escala 0-10. Promedio ponderado = score del trial.

### 5. Serving (runtime)
FastAPI carga modelo base + libreria de vectores. Cada request especifica `{persona, alpha}`.
El vector se aplica en el forward pass, se genera, se limpia.

Ventaja vs abliteration: sin necesidad de multiples modelos serializados. Un modelo base +
N vectores livianos (~10-50MB cada uno).

## Modelo base

**Dev (Mac M4):** Qwen 2.5 3B Instruct — 36 capas, 6GB en fp16, corre a 40-60 tok/s.
**Prod (RunPod H200):** Qwen 2.5 7B Instruct — 32 capas, 14GB en fp16.

Los vectores extraidos de la 3B **no** son compatibles con la 7B directamente (dimensiones
distintas), pero el pipeline es el mismo. Podemos entrenar en 3B para iterar rapido, luego
correr el pipeline final en 7B en RunPod.

## Metricas de exito

Un persona esta "listo" cuando:
1. LLM-as-judge score >= 7.5/10 promedio
2. Coherence subscore >= 8.0 (no puede romperse el modelo)
3. Naturalness subscore >= 7.0 (no puede sonar forzado/robot)

## Diferencias tecnicas vs steering academico

- Usamos **repeng** (implementacion simple) en vez de reimplementar
- Optuna sobre alpha global — algunos papers hacen alpha por capa (mas costoso, marginal gain)
- Judge con Claude vs judge con GPT-4 — nuestra corrida mostro que Claude es mas critico
- Datasets sinteticos (Claude-generated) vs human-annotated — trade-off costo/calidad

## Referencias

- [repeng](https://github.com/vgel/repeng) — libreria core
- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al 2023
- [Steering Vectors](https://arxiv.org/abs/2308.10248) — Turner et al 2023
- Evil Ganda methodology (proyecto hermano)
