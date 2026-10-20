#!/bin/sh
# stt_gateway entrypoint: ensure every model named by the routing table
# exists, then exec the gateway (which spawns/loads the warm engines).
set -eu

MODELS_DIR="${STT_MODELS_DIR:-/models}"
WHISPER_HF_BASE="${STT_MODEL_HF_BASE:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main}"
VAD_MODEL="${STT_VAD_MODEL:-ggml-silero-v6.2.0.bin}"
VAD_HF_BASE="${STT_VAD_MODEL_HF_BASE:-https://huggingface.co/ggml-org/whisper-vad/resolve/main}"

download_curl() {
    url="$1"
    target="$2"
    if [ -s "$target" ]; then
        return 0
    fi
    echo "Downloading $url"
    mkdir -p "$MODELS_DIR"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --retry 3 --silent --show-error -H "Authorization: Bearer $HF_TOKEN" -o "$target.part" "$url"
    else
        curl -fL --retry 3 --silent --show-error -o "$target.part" "$url"
    fi
    mv "$target.part" "$target"
}

# Emit the models every warm engine needs (whisper ggml files, seamless HF
# snapshot) from the routing table - one source of truth.
ROUTING_OUTPUT=$(python - <<'PYEOF'
import sys
sys.path.insert(0, "/app")
from stt_gateway.config import load_settings
from stt_gateway.routing import load_routing

settings = load_settings()
for spec in load_routing(settings.routing_config).engine_specs():
    if spec.kind == "whisper":
        print(f"whisper {spec.model}")
    else:
        print(f"seamless {spec.model} {spec.revision}")
PYEOF
)

if [ "${STT_DOWNLOAD_MODEL:-1}" = "1" ]; then
    echo "$ROUTING_OUTPUT" | while read -r kind model revision; do
        case "$kind" in
            whisper)
                download_curl "$WHISPER_HF_BASE/$model.bin" "$MODELS_DIR/$model.bin"
                ;;
            seamless)
                HF_TOKEN="${HF_TOKEN:-}" python - "$model" "$revision" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

model, revision = sys.argv[1], sys.argv[2]
# Only what transformers consumes for speech-to-text: skip the legacy .pt
# checkpoints and the vocoder (several GB the engine never reads).
snapshot_download(
    model,
    revision=revision or None,
    allow_patterns=["*.json", "*.model", "*.safetensors"],
)
print(f"seamless snapshot ready: {model}@{revision or 'main'}")
PYEOF
                ;;
        esac
    done
    download_curl "$VAD_HF_BASE/$VAD_MODEL" "$MODELS_DIR/$VAD_MODEL"
fi

exec python -m stt_gateway.server
