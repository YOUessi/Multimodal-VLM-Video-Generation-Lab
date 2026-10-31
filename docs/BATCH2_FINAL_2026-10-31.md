# Project 4 — batch 2 verified completion (2026-10-31 19:26 CST)

## Snapshot
All five approved batch-2 stages **completed with process return code 0** on Tang. Local automation state:
`ALL_APPROVED_STAGES_DONE`. Frozen local source SHA: `887b7ca37518fe26f81e17ec53cfd123d5ed65a6`.

### B1–B3: VLM/LoRA controlled benchmark
- Procedurally generated MP4 splits: 48 training / 12 validation / 18 heldout.
- SmolVLM2-500M, no image splitting.
- Actual 48 LoRA optimization steps, adapter saved locally.
- Baseline results: one frame 18/18, three frames 4/18, five frames 4/18.
- After LoRA: 18/18 in all three evaluated tasks.
- **This is synthetic color-sequence video only**; not representative of real natural videos or generalized improvement.

### B4: Wan2.1 feasibility
- Conservative full-weight offload RAM criterion failed. The stage was a preflight and generated no Wan video.

### B5: actual Diffusion generation on Tang
Model: **ZeroScope v2 576w** (not Wan2.1), safetensors weight revision pinned, research-only license.

| Output | Settings | Seconds | Peak PyTorch allocated VRAM | Actual MP4 |
|---|---|---:|---:|---|
| rolling_ball.mp4 | 576×320, 12 frames, 6 fps, 14 diffusion steps, seed 42 | 18.820 | 3.327 GiB | 2 s, MPEG-4; ffmpeg decoded successfully |
| turning_robot.mp4 | 576×320, 12 frames, 6 fps, 14 diffusion steps, seed 31415 | 18.194 | 3.327 GiB | 2 s, MPEG-4; ffmpeg decoded successfully |

B5 log: `END stage=B5_zeroscope_diffusion_video returncode=0` at **2026-10-31 18:50:42 CST**. Video quality scores were explicitly `null`, and an empty human-review CSV was created. No unverified quality claims are made.

All raw generated videos, model checkpoints and private inputs remain on Tang; only the reproducibility code and this summary are pushed to GitHub.

### Runtime and scheduling
- Tang: online, RTX 4090 Laptop; 15,569 MiB free / 1% utilization at 19:26 CST.
- Two Tang local cron jobs are active, each checked every 10 minutes: frozen experiment-runner and bounded checkpoint downloader/recovery.
- The download-state file is `complete` with verified UNet size **2,822,650,224 bytes**.
- The experiment runner will continue to **check**, but there are no additional approved stages to execute. It **cannot autonomously develop and implement the next research batch** or post to this chat.

### Next batch required
1. Merge and test the earlier 31-test CPU module suite from the existing source ZIP, checking parity with GPU scripts.
2. Build genuine, human-reviewed real-video temporal QA with train/validation/test separation, source provenance and leakage checks.
3. Multiple-seed LoRA and frame-sampling ablations on heldout natural videos; report confidence intervals and failure cases.
4. Human-blind assessment of generated clips and, only after verified feasibility, low-memory Wan experimentation.
5. Publish reproducibility manifest and job-targeted results report.

These steps are **planned, not executed**. A new tested/frozen plan is needed before the 10-minute Tang cron can advance them.
