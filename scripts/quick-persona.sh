#!/usr/bin/env bash
# quick-persona.sh — Pipeline completo para una persona
#
# Uso: ./scripts/quick-persona.sh fearful
#
# Corre: generate → extract → optimize → evaluate final
# Timings estimados (Mac M4 Pro, Qwen 3B):
#   generate:    ~2 min (300 llamadas Claude)
#   extract:     ~1 min
#   optimize:    ~30-60 min (100 trials × ~30s cada uno)
#   evaluate:    ~2 min

set -e

if [ -z "$1" ]; then
    echo "Uso: $0 <persona_name>"
    echo "Personas disponibles:"
    ls personas/*.yaml | xargs -n1 basename | sed 's/.yaml//' | sed 's/^/  - /'
    exit 1
fi

PERSONA="$1"
YAML="personas/${PERSONA}.yaml"

if [ ! -f "$YAML" ]; then
    echo "ERROR: $YAML no existe"
    exit 1
fi

echo "== 1/4: Generando dataset ($PERSONA) =="
python -m src.generate_dataset "$YAML"

echo ""
echo "== 2/4: Extrayendo vector base =="
python -m src.extract_vector "$PERSONA"

echo ""
echo "== 3/4: Optimizando con Optuna =="
python -m src.optimize_vector "$PERSONA"

echo ""
echo "== 4/4: Evaluando vector final =="
python -m src.evaluate "$PERSONA" --vector "vectors/${PERSONA}.safetensors" --alpha 1.0

echo ""
echo "Listo. Vector: vectors/${PERSONA}.safetensors"
