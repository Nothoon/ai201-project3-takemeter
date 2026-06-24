"""
app.py  -  TakeMeter web interface (Gradio).

Run:  python app.py     then open the printed local URL.

Accepts a post, runs the fine-tuned classifier, and shows the predicted label plus a
confidence bar across all three classes. Uses ./takemeter_model/ if present, otherwise
the documented heuristic fallback (see classify.py).
"""
import os
import gradio as gr
from classify import LABELS, MODEL_DIR, heuristic_predict

_backend = "heuristic fallback"
_predict_one = heuristic_predict

if os.path.isdir(MODEL_DIR):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

        def _full_probs(text):
            enc = _tok(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                probs = torch.softmax(_model(**enc).logits, dim=1)[0].tolist()
            return {_model.config.id2label[i]: float(probs[i]) for i in range(len(LABELS))}
        _backend = "fine-tuned DistilBERT"
    except Exception:
        _full_probs = None
else:
    _full_probs = None


def classify(text):
    if not text or not text.strip():
        return {l: 0.0 for l in LABELS}
    if _full_probs is not None:
        return _full_probs(text)
    lab, conf = heuristic_predict(text)          # fallback gives top-class confidence
    rest = (1 - conf) / (len(LABELS) - 1)
    return {l: (conf if l == lab else rest) for l in LABELS}


EXAMPLES = [
    "Kendrick's pocket on the second verse is unreal, he raps a half beat behind the snare so it feels lazy and urgent at once.",
    "Drake is the most overrated artist of all time and history will be brutal to him.",
    "ALBUM OF THE YEAR no skips I've been screaming since midnight",
    "the way I gasped when the beat switched I woke up my whole apartment",
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="r/hiphopheads comment",
                      placeholder="Paste a hip-hop take..."),
    outputs=gr.Label(num_top_classes=3, label=f"TakeMeter ({_backend})"),
    title="TakeMeter — critique / hot_take / stan",
    description="Classifies hip-hop discourse quality. Confidence shown per class.",
    examples=EXAMPLES,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
