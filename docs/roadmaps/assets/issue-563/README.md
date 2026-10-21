# English diagrams for RL-Kernel issue #563

These six diagrams preserve the layout and visual style of the approved Chinese roadmap: rounded boxes, dotted groups, and blue hatching with `[+]` for proposed additions or extensions.

The PNGs are the images embedded in [issue #563](https://github.com/RL-Align/RL-Kernel/issues/563). The diagrams are discussion designs; they do not report new NPU validation results.

## Regenerate

Install `reportlab`, `pymupdf` and `Pillow`, then run `python render.py`. It writes PNG, SVG and PDF versions beside the script. PNGs use a white background and a 2400 x 1320 canvas.

| File | Figure |
| --- | --- |
| architecture.png | New and extended RL-Kernel code, including verl integration |
| dispatcher.png | Hardware-aware kernel selection |
| speculative.png | Shared speculative-decoding interfaces |
| pd-consistency.png | Prefill/Decode handoff and training comparison |
| pd-topology.png | Prefill/Decode expert-parallel placement |
| pd-combine.png | Fixed-slot MoE combine and FP32 reduction tree |

The top-k of eight and the device/expert counts are illustrative. Detailed assumptions and validation requirements remain in the issue body.

The architecture diagram follows PR #497. It uses the vime integration present in v0.1.0 as a reference: training initialization and logprob-provider hooks, plus the vLLM rollout plugin. The vime-to-verl dotted arrow denotes reuse of the integration approach, not a runtime dependency. Proposed common-code extraction is described in the issue.
