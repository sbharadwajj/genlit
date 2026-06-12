# GenLit Gradio demo

Browser UI that wraps `genlit.inference` for single-image relighting.

Supports `single` and `multi` modes.
## Run locally

From the repo root:

```bash
pip install -e ".[demo]"
python demo/app.py
```

Open the printed URL (usually http://127.0.0.1:7860). Upload an image, pick a
mode, click Generate. First run downloads ~3GB of weights from HuggingFace.

## Deploy as HF Space

1. Create a new Space at https://huggingface.co/new-space (SDK: Gradio).
2. Choose hardware: GPU (the demo is unusable on CPU — 25-frame inference would take ~30 min).
3. Push the contents of `demo/` plus the `genlit/` package to the Space.

Suggested structure for the Space:

```
genlit-demo/
├── app.py              # from demo/app.py
├── requirements.txt    # from demo/requirements.txt
├── genlit/             # copy of the importable package
└── configs/, trajectories/   # needed by the loader
```

Easier alternative: `pip install -e git+https://github.com/<your-org>/genlit.git`
in `requirements.txt` and import `from genlit.inference import ...`.
