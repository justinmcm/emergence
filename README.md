# Emergence — AI Safety Interpretability Viewer

A 3D physics-based tool for visualizing neural network internals and measuring safety processing in language models.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-CUDA-green)
![License](https://img.shields.io/badge/license-MIT-purple)

---

## What this is

Emergence renders a running language model as a **living 3D physics simulation**. Each token becomes a particle. Attention weights become gravitational forces. Tokens that attend to each other pull together. Tokens that ignore each other drift apart.

The result is an interpretability tool that lets you **watch a model think** — and measure where safety processing happens inside the network.

---

## The Safety Experiment

We ran 500 matched sentence pairs (safe vs unsafe) across 10 harm categories through Phi-3 Mini 4k Instruct and measured **hidden state divergence at every layer**.

The core question: when a safety-trained model internally distinguishes between a safe and an unsafe sentence — at which layer does that distinction first appear?

**Key findings (preliminary — see results_viewer.html for full data):**

- When detection occurs, it almost always happens at a single consistent layer in the final 6% of the network
- Detection rates vary dramatically by harm category — some harm types are nearly invisible to the model internally
- Intent framing produces minimal internal distinction — sentences with identical content but opposite intent are represented nearly identically inside the model
- The late alarm profile suggests the model fully processes harmful content before recognizing it

---

## Setup

### Requirements

- Python 3.12
- NVIDIA GPU with 6GB+ VRAM (CPU works but is slow)
- 8GB free disk space

### Install

```bash
git clone https://github.com/justinmcm/emergence
cd emergence
pip install -r requirements.txt
```

### Run

```bash
python server.py
# Then open frontend/index.html in Chrome or Firefox
```

First run downloads the model (~2.4GB). Subsequent runs load instantly.

---

## The Viewer

**Explore mode** — type any sentence and watch the model process it in 3D. Tokens are spheres. Attention weights are arcing light threads. Scrub through all 32 layers. Click any token to focus on its connections.

**Safety comparison mode** — type a safe/unsafe sentence pair. Both run simultaneously in split screen. Left universe is tinted green (safe), right is red (unsafe). A divergence panel shows when and where the model's internal representations diverge.

Controls: `←` `→` to step layers, `Space` to pause physics, `Escape` to deselect, scroll to zoom, drag to orbit.

---

## Run the Full Experiment

```bash
# With server.py running:
python run_experiment.py
```

Runs 500 sentence pairs. Saves progress to `results.json` every 10 pairs. Produces `summary.json` on completion. Open `results_viewer.html` and load `summary.json` to view findings.

Estimated runtime: 4-6 hours on an RTX 3050.

---

## Project structure

```
emergence/
├── server.py              — Flask server, model loading, attention extraction
├── pairs.py               — 500 sentence pairs across 10 harm categories
├── run_experiment.py      — Bulk experiment runner
├── results_viewer.html    — Standalone results visualizer
├── requirements.txt       — Python dependencies
└── frontend/
    └── index.html         — Three.js 3D viewer
```

---

## Methodology

For each sentence pair:

1. Run both sentences through the model with `output_hidden_states=True`
2. At each layer, compute cosine distance between mean hidden state vectors
3. Detect alarm when divergence spikes sharply (>1.8x previous layer, absolute >0.01)
4. Record alarm layer, full divergence curve, and top diverging tokens

This measures where internal representation diverges — not what the model outputs.

**Limitations:** Single model, proxy measure, hand-crafted pairs, heuristic threshold. Results are preliminary and should be replicated on larger models.

---

## Background

This started as an attempt to build a visual programming language. The core insight: neural networks already are physics systems. The interpretability viewer emerged from asking what it would look like to watch intelligence think. The safety experiment emerged from asking whether we can see where safety processing happens.

---

## License

MIT — built by Justin McMahon, independent researcher.
