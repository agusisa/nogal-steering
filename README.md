# nogal-steering

Behavior amplification en LLMs via control vectors + Optuna optimization.

Complemento de Evil Ganda: en vez de remover comportamientos (abliteration), los amplificamos. Un mismo modelo base + libreria de vectores = multiples personalidades cargables en runtime.

## Idea

Los LLMs codifican comportamientos como direcciones geometricas en el espacio de activaciones. Con dos sets de ejemplos (positive/negative del comportamiento deseado), extraemos el vector de esa direccion. En inference, sumamos ese vector escalado por alpha a las activaciones y el modelo se comporta segun el vector.

Optuna optimiza los hiperparametros: alpha por capa, metodo de agregacion, rango de capas.

## Modelo base

- **Development (Mac M4):** Qwen 2.5 3B Instruct
- **Production runs (RunPod H200):** Qwen 2.5 7B Instruct
- Framework: [repeng](https://github.com/vgel/repeng)

## Personas objetivo (Fase 3)

- fearful — miedoso, ansioso, evasivo
- aggressive — hostil, impaciente
- philosopher — reflexivo, respuestas largas
- technical — preciso, detallado, cero fluff
- poetic — metaforico, ritmo, imagenes
- paranoid — conspirativo, desconfiado
- flirty — coqueto, insinuante

## Estructura

```
src/          # Pipeline: generate → extract → optimize → evaluate → serve
personas/     # YAML definitions + serialized vectors
datasets/     # positive.jsonl + negative.jsonl por persona
vectors/      # .safetensors con los vectores finales
notebooks/    # Analisis exploratorio
docs/         # Metodologia, charla
scripts/      # One-shot helpers
```

## Estado

En desarrollo. Fase 1: MVP con fearful como piloto.

## Charla origen

Ver [Evil Ganda](https://nogal-labs.com.ar/jailbreak-demo/) — proyecto hermano que usa abliteration (sustractivo). Este es aditivo.

---

Nogal Labs 2026
