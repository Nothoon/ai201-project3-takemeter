"""
classify.py  -  TakeMeter command-line inference.

Usage:
  python classify.py "Kendrick's pocket on the second verse is unreal..."
  python classify.py            # interactive REPL, one post per line

It loads the fine-tuned DistilBERT from ./takemeter_model/ (download it from the Colab
notebook's Section 6). If those weights aren't present, it falls back to the documented
feature heuristic that reproduces the model's reported decision boundary, so the
interface is demoable without a GPU. The fallback prints a clear warning.
"""
import os, sys, re

LABELS = ["critique", "hot_take", "stan"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "takemeter_model")


# ---- real fine-tuned model path ------------------------------------------------
def load_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    def predict(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[0]
        i = int(probs.argmax())
        return model.config.id2label[i], float(probs[i])
    return predict


# ---- heuristic fallback (mirrors run_eval.py reconstruction) -------------------
ANCHOR = re.compile(r"\b(chord|progression|flow|triplet|cadence|mix(?:ing|ed)?|808s?|"
                    r"snare|kick|hi[- ]?hat|bpm|tempo|reverb|sampl|loop|bridge|hook|"
                    r"verse|bars?|enjambment|rhyme scheme|octave|sidechain|register|"
                    r"key change|modulat|sequenc|breath|drums?|production|produc(?:er|ed)|"
                    r"beat switch|beats?|vocal|melody|melodic|bassline|bass|ad libs?|"
                    r"punchline|multi|internal rhyme|pocket|panned|master(?:ing|ed)|"
                    r"interval|runtime|arrangement|diction|enunciat)\b", re.I)
VERDICT = re.compile(r"\b(overrated|underrated|goat|best|worst|top \d|mid|greatest|never|"
                     r"only|carried|washed|not close|debate|facts|period|full stop|"
                     r"there i said it|magnum opus|history will|cope|coping|denial)\b", re.I)
CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")


def heuristic_predict(text):
    caps = [w for w in CAPS_WORD.findall(text)
            if w not in {"DOOM", "JID", "DAMN", "BPM", "AABB", "AOTY", "UK", "XXL"}]
    shouting = len(caps) >= 2 or any(len(w) >= 5 for w in caps)
    if shouting:
        return "stan", 0.93
    if ANCHOR.search(text):
        return "critique", 0.85
    if VERDICT.search(text):
        return "hot_take", 0.83
    return "stan", 0.66


def get_predictor():
    if os.path.isdir(MODEL_DIR):
        try:
            return load_model(), "fine-tuned DistilBERT (takemeter_model/)"
        except Exception as e:  # transformers/torch missing or bad weights
            print(f"[warn] could not load model: {e}\n[warn] using heuristic fallback.\n")
    else:
        print("[warn] ./takemeter_model not found — using heuristic fallback "
              "(download the trained model from Colab for real inference).\n")
    return heuristic_predict, "heuristic fallback"


def main():
    predict, backend = get_predictor()
    print(f"TakeMeter  |  backend: {backend}  |  labels: {', '.join(LABELS)}\n")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        lab, conf = predict(text)
        print(f"  {lab:9s}  {conf:.0%}   {text}")
        return
    print("Type a post and press enter (blank line to quit):")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        lab, conf = predict(text)
        print(f"  -> {lab:9s}  confidence {conf:.0%}")


if __name__ == "__main__":
    main()
