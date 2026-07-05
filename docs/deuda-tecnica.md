# Deuda Tecnica

## RunPod SDK integration ✓ IMPLEMENTADO

Ver `src/runpod_runner.py`. Usage:

```bash
export RUNPOD_API_KEY=***
python -m src.runpod_runner fearful --gpu H100 --n-trials 30
```

Flow:
1. Crea pod H100 con imagen PyTorch preconfigurada
2. Rsync repo + dataset via SSH
3. Corre `optimize_vector` remoto
4. Descarga `vectors/{persona}.gguf` + `optimization/{persona}.report.json`
5. Termina pod (o --keep-alive para debugging)

Requiere:
- `RUNPOD_API_KEY` en `.env` (obtener en runpod.io/console/user/settings)
- SSH key configurada en RunPod (misma que ~/.ssh/id_rsa)

Costo: H100 ~$2/hr × 30 trials ≈ $1-2 por persona.

## Otras cosas para deuda tecnica

- **Model quantization post-training:** actualmente cargamos fp16, podriamos usar bitsandbytes 4bit para 3B en <2GB
- **Batch inference en optimize:** el `generate_with_vector` procesa 1 prompt a la vez, podriamos batchear
- **Cache de vectores por trial:** si dos trials tienen mismos layers y aggregation, reusar (Optuna trial pruning)
- **Streaming de responses en serve.py:** actualmente responde full, deberia stream SSE
- **Web UI para explorar personas + alpha slider:** tipo el chat de Evil Ganda pero con dropdown de persona
- **Vector composition:** sumar vectores (`0.5 * fearful + 0.7 * poetic`) — repeng ya lo soporta nativamente
