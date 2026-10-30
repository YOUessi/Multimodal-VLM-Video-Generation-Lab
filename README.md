# Multimodal VLM & Video Generation Lab

Research-oriented project for **visual question answering, temporal video reasoning, VLM LoRA adaptation, video generation, and evaluation**. The work is split into two independent tracks:

- **Track A: VLM / Video Understanding:** frame sampling, image and multi-image inference, temporal QA, LoRA fine-tuning, split-safe benchmarking.
- **Track B: Video Generation:** Wan/Diffusion pipelines and human-quality assessment, kept separate from VLM understanding and training.

## Actual verified GPU progress

Tang (RTX 4090 Laptop 16GB) has completed:
- SmolVLM2-500M model download through a mirror and **real CUDA inference**.
- 2 successful synthetic color QA answers and 1 incorrect multi-frame order answer (not an accuracy benchmark).
- 2 actual PEFT LoRA optimization steps with finite losses; local adapter saved (not a model improvement claim).
- Local 110.388s video qualitative 1-frame / 5-frame inference (private original video never uploaded).

See [measured results and limitations](docs/GPU_SMOKE_2026-10-31.md).

## Run measured GPU scripts

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install 'transformers==4.57.6' 'pillow>=10.4,<12' 'accelerate>=1.8,<2' 'peft>=0.17,<0.19' 'trl>=0.24,<0.26' 'datasets>=3,<5' 'diffusers>=0.35,<0.38' num2words soxr

# Download once; required for the scripts below.
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    local_dir='models/SmolVLM2-500M-Video-Instruct'
)
PY

.venv/bin/python experiments/smolvlm2_gpu_smoke.py
.venv/bin/python experiments/smolvlm2_lora_gpu_smoke.py
.venv/bin/python experiments/local_video_gpu_probe.py --video /your/local/video.mp4
```

**Status:** The three GPU experiment scripts and logs summary are committed here. The earlier complete CPU prototype (31 passing tests; annotated data tools and Wan dry-run CLI) is currently provided as a separate source ZIP, and **has not yet been fully pushed into this GitHub repository**. Do not assume that the repository currently contains those other modules.

**Restrictions:** Never commit raw videos, model weights, LoRA adapter files, private paths or credentials. Real-world training benchmark and Wan video generation are not yet verified.
