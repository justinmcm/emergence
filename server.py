from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__)
CORS(app)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Loading {MODEL_NAME} in 4-bit quantization...")
print("First run downloads ~4GB. This takes a few minutes...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    output_attentions=True,
    attn_implementation="eager"
)
model.eval()

print(f"Model ready on {DEVICE}.")
print("Server running at http://localhost:5000")


def run_model(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True
        )

    return tokens, outputs


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tokens, outputs = run_model(text)
    n_tokens   = len(tokens)
    attentions = outputs.attentions
    n_layers   = len(attentions)
    n_heads    = attentions[0].shape[1]

    if n_tokens < 2:
        return jsonify({"error": "Please enter at least two words"}), 400

    layers_data = []
    for li in range(n_layers):
        avg_attn = attentions[li][0].mean(dim=0).float().cpu().numpy()

        connections = []
        for src in range(n_tokens):
            for dst in range(n_tokens):
                w = float(avg_attn[src, dst])
                if w > 0.015:
                    connections.append({"from": src, "to": dst, "weight": round(w, 4)})

        heads = []
        for hi in range(n_heads):
            head_attn = attentions[li][0, hi].float().cpu().numpy()
            hconns = []
            for src in range(n_tokens):
                for dst in range(n_tokens):
                    w = float(head_attn[src, dst])
                    if w > 0.03:
                        hconns.append({"from": src, "to": dst, "weight": round(w, 4)})
            heads.append(hconns)

        layers_data.append({"connections": connections, "heads": heads})

    temperatures = []
    for ti in range(n_tokens):
        received = sum(
            float(attentions[li][0].mean(dim=0)[:, ti].mean().float().cpu())
            for li in range(n_layers)
        ) / n_layers
        temperatures.append(round(received, 4))

    t_min   = min(temperatures)
    t_range = max(temperatures) - t_min or 1.0
    temperatures_norm = [round((t - t_min) / t_range, 4) for t in temperatures]

    return jsonify({
        "tokens":        tokens,
        "n_tokens":      n_tokens,
        "n_layers":      n_layers,
        "n_heads":       n_heads,
        "device":        str(DEVICE),
        "model":         MODEL_NAME,
        "layers":        layers_data,
        "temperatures":  temperatures_norm
    })


@app.route("/safety_compare", methods=["POST"])
def safety_compare():
    data = request.get_json()
    text_safe   = data.get("safe",   "").strip()
    text_unsafe = data.get("unsafe", "").strip()

    if not text_safe or not text_unsafe:
        return jsonify({"error": "Both safe and unsafe sentences required"}), 400

    tokens_safe,   out_safe   = run_model(text_safe)
    tokens_unsafe, out_unsafe = run_model(text_unsafe)

    hidden_safe   = out_safe.hidden_states
    hidden_unsafe = out_unsafe.hidden_states
    n_layers = len(hidden_safe)

    # Layer-level divergence — cosine distance between mean hidden states
    layer_divergence = []
    for li in range(n_layers):
        hs = hidden_safe[li][0].float().mean(dim=0)
        hu = hidden_unsafe[li][0].float().mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            hs.unsqueeze(0), hu.unsqueeze(0)
        ).item()
        layer_divergence.append(round(1.0 - cos_sim, 6))

    # Alarm layer — first significant spike in divergence
    alarm_layer = None
    for li in range(1, n_layers):
        if layer_divergence[li] > 0.01:
            prev = layer_divergence[li-1] if layer_divergence[li-1] > 0 else 0.001
            if layer_divergence[li] / prev > 1.8:
                alarm_layer = li
                break

    # Token-level divergence at each layer
    min_tokens = min(len(tokens_safe), len(tokens_unsafe))
    token_divergence_by_layer = []
    for li in range(n_layers):
        tok_divs = []
        for ti in range(min_tokens):
            hs = hidden_safe[li][0, ti].float()
            hu = hidden_unsafe[li][0, ti].float()
            cos_sim = torch.nn.functional.cosine_similarity(
                hs.unsqueeze(0), hu.unsqueeze(0)
            ).item()
            tok_divs.append(round(1.0 - cos_sim, 6))
        token_divergence_by_layer.append(tok_divs)

    # Which tokens drive divergence most at alarm layer
    driver_layer = alarm_layer if alarm_layer is not None else n_layers - 1
    token_drivers = []
    if driver_layer < len(token_divergence_by_layer):
        divs = token_divergence_by_layer[driver_layer]
        for ti, d in enumerate(divs):
            token_drivers.append({
                "index":        ti,
                "safe_token":   tokens_safe[ti]   if ti < len(tokens_safe)   else "?",
                "unsafe_token": tokens_unsafe[ti] if ti < len(tokens_unsafe) else "?",
                "divergence":   d
            })
        token_drivers.sort(key=lambda x: x["divergence"], reverse=True)

    # Normalize 0→1
    dmax = max(layer_divergence) or 1.0
    layer_divergence_norm = [round(d / dmax, 4) for d in layer_divergence]

    return jsonify({
        "tokens_safe":               tokens_safe,
        "tokens_unsafe":             tokens_unsafe,
        "n_layers":                  n_layers,
        "layer_divergence":          layer_divergence,
        "layer_divergence_norm":     layer_divergence_norm,
        "alarm_layer":               alarm_layer,
        "token_divergence_by_layer": token_divergence_by_layer,
        "token_drivers":             token_drivers[:5],
        "model":                     MODEL_NAME
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
