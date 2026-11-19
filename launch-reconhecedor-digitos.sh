#!/usr/bin/env bash
# Inicia o reconhecimento de dígitos pela webcam.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  notify-send "Reconhecedor de Dígitos" \
    "Ambiente .venv não encontrado. Crie com: uv venv --python 3.12 .venv && .venv/bin/pip install -r requirements.txt" \
    2>/dev/null || true
  echo "Erro: .venv não encontrado em $ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT/modelos/digitos.tflite" && ! -f "$ROOT/modelos/digitos.keras" ]]; then
  notify-send "Reconhecedor de Dígitos" \
    "Modelo não encontrado. Rode: .venv/bin/python treinar_modelo.py" \
    2>/dev/null || true
  echo "Erro: modelos/digitos.tflite ou .keras ausentes." >&2
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=2
exec "$PYTHON" "$ROOT/camera_webcam.py" "$@"
