#!/usr/bin/env bash
# run_steered.sh — Corre el modelo abliterado (GGUF Q4) con un control vector aplicado.
#
# Uso:
#   ./scripts/run_steered.sh fearful 2.0 "Contame sobre tu dia"
#   ./scripts/run_steered.sh fearful -2.0 "Contame sobre tu dia"   # invertido (confiado)
#
# Args:
#   $1 = persona (nombre del vector en vectors/{persona}.gguf)
#   $2 = alpha (coeficiente; negativo invierte el efecto)
#   $3 = prompt

set -e

PERSONA="${1:-fearful}"
ALPHA="${2:-2.0}"
PROMPT="${3:-Contame algo sobre vos}"

REPO="$HOME/repos/nogal-steering"
MODEL="$HOME/repos/jail/models/qwen-7b-jailbreak-q4.gguf"
VECTOR="$REPO/vectors/${PERSONA}.gguf"

if [ ! -f "$VECTOR" ]; then
    echo "ERROR: no existe $VECTOR"
    echo "Entrena primero: python -m src.extract_vector $PERSONA"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: no existe $MODEL"
    exit 1
fi

# Formato de prompt con chat template de Qwen
FULL_PROMPT="<|im_start|>user
${PROMPT}<|im_end|>
<|im_start|>assistant
"

echo "=== Steered generation ==="
echo "Persona: $PERSONA | alpha: $ALPHA"
echo "Prompt: $PROMPT"
echo "=========================="
echo ""

# llama-completion es one-shot (no REPL). Sintaxis vector: FNAME:SCALE
llama-completion \
    --model "$MODEL" \
    --control-vector-scaled "${VECTOR}:${ALPHA}" \
    --control-vector-layer-range 10 26 \
    -p "$FULL_PROMPT" \
    -n 200 \
    --temp 0.7 \
    2>/dev/null
