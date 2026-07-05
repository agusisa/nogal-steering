# Deuda Tecnica

## RunPod SDK integration (P1)

**Objetivo:** correr `optimize_vector.py` en RunPod sin tocar la UI web.

**Investigacion pendiente:**
- RunPod tiene [`runpod-python`](https://github.com/runpod/runpod-python) SDK oficial
- Endpoints programaticos: crear pod, correr command, obtener logs, apagar
- Alternativa: [`runpodctl`](https://github.com/runpod/runpodctl) CLI

**Flow ideal:**
```
python -m src.optimize_vector fearful --runpod --gpu H100
  ↓
1. SDK crea pod H100 con imagen preconfigurada
2. Sube el repo + dataset + persona.yaml
3. Corre el optimize_vector.py remoto
4. Descarga vectors/fearful.gguf + report.json
5. Apaga el pod
```

**Costo estimado:** H100 ~$2/hr, un optimize de 30 trials ~1hr = $2 por persona.

**Trade-offs:**
- Pro: no cargamos Mac, podemos entrenar en Qwen 7B
- Con: cold-start del pod ~2min, setup de imagen custom

## Otras cosas para deuda tecnica

- **Model quantization post-training:** actualmente cargamos fp16, podriamos usar bitsandbytes 4bit para 3B en <2GB
- **Batch inference en optimize:** el `generate_with_vector` procesa 1 prompt a la vez, podriamos batchear
- **Cache de vectores por trial:** si dos trials tienen mismos layers y aggregation, reusar (Optuna trial pruning)
- **Streaming de responses en serve.py:** actualmente responde full, deberia stream SSE
- **Web UI para explorar personas + alpha slider:** tipo el chat de Evil Ganda pero con dropdown de persona
- **Vector composition:** sumar vectores (`0.5 * fearful + 0.7 * poetic`) — repeng ya lo soporta nativamente
