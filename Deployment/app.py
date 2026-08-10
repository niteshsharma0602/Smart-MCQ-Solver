"""
Fine-tuned DistilBERT (checkpoint-2820) pairwise sequence scorer.
Accepts a question and five options (A–E), returns the top 3 ranked answers.
"""

import spaces

import re
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "./model"
MAX_LENGTH = 160
OPTIONS = ["A", "B", "C", "D", "E"]



NOISE = [
    "pick the best possible answer",
    "select the most accurate option",
    "identify the correct statement",
    "choose the correct answer",
    "determine the correct option",
    "which of the following is correct",
    "among the listed options",
    "from the following choices",
    "based on the given context",
    "carefully",
]

REQUIRED_FILES = [
    "model/config.json",
    "model/model.safetensors",
    "model/tokenizer.json",
    "model/tokenizer_config.json",
]

for file_path in REQUIRED_FILES:
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Required deployment file missing: {file_path}"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model / tokenizer loading
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Preprocessing

def clean_prompt(text: str) -> str:
    """Remove known boilerplate wrappers and normalize whitespace."""
    text = str(text)
    for phrase in NOISE:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Prediction function
@spaces.GPU
def predict_mcq(
    prompt: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    option_e: str,
) -> str:
    """Validate inputs, run inference, return top-3 ranked options."""

    prompt = (prompt or "").strip()
    options_raw = [
        (option_a or "").strip(),
        (option_b or "").strip(),
        (option_c or "").strip(),
        (option_d or "").strip(),
        (option_e or "").strip(),
    ]

    if not prompt:
        return "⚠️ Please enter a question."

    missing = [OPTIONS[i] for i, opt in enumerate(options_raw) if not opt]
    if missing:
        return f"⚠️ Please fill in option(s): {', '.join(missing)}"

    cleaned_prompt = clean_prompt(prompt)
    option_texts = [str(opt) for opt in options_raw]

    encoded = tokenizer(
        [cleaned_prompt] * 5,
        option_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded)

    logits = outputs.logits.squeeze(-1)
    scores = torch.sigmoid(logits)

    ranked_indices = torch.argsort(scores, descending=True)
    top3_indices = ranked_indices[:3]

    MEDALS = ["🥇", "🥈", "🥉"]
    lines = []
    for rank, idx in enumerate(top3_indices, start=1):
        label = OPTIONS[idx]
        text = option_texts[idx]
        score_val = scores[idx].item()
        lines.append(
            f"{MEDALS[rank-1]} **Option {label}** &nbsp;·&nbsp; confidence {score_val:.2f}\n\n{text}"
        )

    return "\n\n---\n\n".join(lines)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* Reset & base */
* { box-sizing: border-box; }

body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    background: #0f1117 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* Kill default Gradio chrome */
.gradio-container > .main > .wrap {
    padding: 0 !important;
    gap: 0 !important;
}

/* ── Shell layout ── */
#shell {
    display: grid;
    grid-template-rows: auto 1fr;
    height: 100vh;
    padding: 16px 20px 12px;
    gap: 12px;
}

/* ── Header ── */
#header {
    display: flex;
    align-items: baseline;
    gap: 10px;
}
#header h1 {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
}
#header span {
    font-size: 12px;
    color: #64748b;
    font-weight: 400;
}

/* ── Two-column body ── */
#body {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 14px;
    min-height: 0;
}

/* ── Panels ── */
.panel {
    background: #1e2130;
    border: 1px solid #2d3147;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
    overflow: hidden;
}

.panel-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d3147;
}

/* ── Gradio textbox overrides ── */
.gr-textbox, .gr-input, textarea, input[type=text] {
    background: #0f1117 !important;
    border: 1px solid #2d3147 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    padding: 7px 10px !important;
    transition: border-color 0.15s !important;
    resize: none !important;
}
.gr-textbox:focus-within, textarea:focus, input[type=text]:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
label span, .gr-textbox label span {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    margin-bottom: 3px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Question takes a bit more height */
#question-box textarea { min-height: 80px !important; max-height: 80px !important; }

/* Options grid */
#options-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    flex: 1;
}
#options-grid > div:first-child {
    grid-column: span 2;   /* Option A full width */
}
#options-grid textarea {
    min-height: 48px !important;
    max-height: 48px !important;
}

/* ── Buttons ── */
#btn-row {
    display: flex;
    gap: 8px;
    margin-top: 2px;
}
#predict-btn {
    flex: 1;
    background: #6366f1 !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 0 !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
#predict-btn:hover { background: #4f46e5 !important; }

#clear-btn {
    background: transparent !important;
    border: 1px solid #2d3147 !important;
    border-radius: 6px !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: border-color 0.15s, color 0.15s !important;
}
#clear-btn:hover {
    border-color: #64748b !important;
    color: #94a3b8 !important;
}

/* ── Output panel ── */
#output-panel {
    display: flex;
    flex-direction: column;
}
#output-content {
    flex: 1;
    overflow-y: auto;
    padding-right: 4px;
}
#output-content::-webkit-scrollbar { width: 4px; }
#output-content::-webkit-scrollbar-track { background: transparent; }
#output-content::-webkit-scrollbar-thumb { background: #2d3147; border-radius: 2px; }

/* Markdown inside output */
#output-content .prose, #output-content p, #output-content .gr-markdown {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #cbd5e1 !important;
}
#output-content strong { color: #f1f5f9 !important; }
#output-content hr {
    border: none !important;
    border-top: 1px solid #2d3147 !important;
    margin: 12px 0 !important;
}

/* Empty state */
#empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 8px;
    color: #334155;
}
#empty-state .icon { font-size: 32px; }
#empty-state p { font-size: 13px; margin: 0; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Smart MCQ Solver") as demo:

    with gr.Column(elem_id="shell"):

        # Header
        with gr.Row(elem_id="header"):
            gr.HTML('<h1>🎓 Smart MCQ Solver</h1><span>Fine-tuned DistilBERT · top-3 ranking</span>')

        # Body — two columns
        with gr.Row(elem_id="body"):

            # ── Left panel: inputs ──────────────────────────────────────────
            with gr.Column(elem_classes=["panel"]):
                gr.HTML('<p class="panel-title">Question &amp; Options</p>')

                question = gr.Textbox(
                    label="Question",
                    placeholder="Type your question here…",
                    lines=2,
                    max_lines=2,
                    elem_id="question-box",
                )

                with gr.Column(elem_id="options-grid"):
                    opt_a = gr.Textbox(label="Option A", placeholder="Option A", lines=1, max_lines=1)
                    with gr.Row():
                        opt_b = gr.Textbox(label="Option B", placeholder="Option B", lines=1, max_lines=1)
                        opt_c = gr.Textbox(label="Option C", placeholder="Option C", lines=1, max_lines=1)
                    with gr.Row():
                        opt_d = gr.Textbox(label="Option D", placeholder="Option D", lines=1, max_lines=1)
                        opt_e = gr.Textbox(label="Option E", placeholder="Option E", lines=1, max_lines=1)

                with gr.Row(elem_id="btn-row"):
                    predict_btn = gr.Button("🔍 Predict Top 3", variant="primary", elem_id="predict-btn")
                    clear_btn = gr.ClearButton(
                        components=[question, opt_a, opt_b, opt_c, opt_d, opt_e, output],
                        value="✕ Clear",
                        elem_id="clear-btn",
                    )

            # ── Right panel: output ─────────────────────────────────────────
            with gr.Column(elem_classes=["panel"], elem_id="output-panel"):
                gr.HTML('<p class="panel-title">Results</p>')

                output = gr.Markdown(
                    value=(
                        "<div id='empty-state'>"
                        "<div class='icon'>📋</div>"
                        "<p>Results will appear here after prediction.</p>"
                        "</div>"
                    ),
                    elem_id="output-content",
                )

    EMPTY_STATE = (
        "<div id='empty-state'>"
        "<div class='icon'>📋</div>"
        "<p>Results will appear here after prediction.</p>"
        "</div>"
    )

    predict_btn.click(
        fn=predict_mcq,
        inputs=[question, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=output,
    )

    clear_btn.click(
        fn=lambda: EMPTY_STATE,
        inputs=[],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()