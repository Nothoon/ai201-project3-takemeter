"""Builds takemeter_finetune.ipynb (the filled-in Colab notebook)."""
import json

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# TakeMeter — Fine-tuning DistilBERT on r/hiphopheads discourse

Classifies a hip-hop comment as **critique** / **hot_take** / **stan**.

**Runtime → Change runtime type → T4 GPU** before running anything.

Sections:
0. Install + setup
1. Label map + upload CSV
2. Split (70/15/15) + tokenize
3. Fine-tune `distilbert-base-uncased`
4. Evaluate fine-tuned model + confusion matrix
5. Zero-shot baseline (Groq `llama-3.3-70b-versatile`)
6. Compare + export (`evaluation_results.json`, `confusion_matrix.png`)
7. (Stretch) interactive inference
"""))

cells.append(md("## 0. Install + setup"))
cells.append(code("""!pip -q install -U transformers datasets scikit-learn groq matplotlib accelerate

import os, json, numpy as np, pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
print("GPU available:", torch.cuda.is_available())"""))

cells.append(md("""## 1. Label map + upload CSV

The label map is the one design decision the starter notebook asks you to fill in.
When prompted, upload **`takemeter_dataset.csv`** (columns: `text`, `label`, `notes`)."""))
cells.append(code("""LABELS = ["critique", "hot_take", "stan"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

from google.colab import files
uploaded = files.upload()                      # choose takemeter_dataset.csv
csv_name = next(iter(uploaded))
df = pd.read_csv(csv_name)
df = df[df["label"].isin(LABELS)].reset_index(drop=True)
print(f"Loaded {len(df)} rows")
print(df["label"].value_counts())"""))

cells.append(md("## 2. Split (70/15/15, stratified, seed 42) + tokenize"))
cells.append(code("""X = df["text"].tolist(); y = df["label"].tolist()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                            random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                            random_state=SEED, stratify=y_tmp)
print(f"train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

from transformers import AutoTokenizer
from datasets import Dataset
MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)

def to_ds(texts, labels):
    d = Dataset.from_dict({"text": texts, "label": [label2id[l] for l in labels]})
    return d.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                               max_length=128), batched=True)

ds_tr, ds_val, ds_te = to_ds(X_tr, y_tr), to_ds(X_val, y_val), to_ds(X_te, y_te)"""))

cells.append(md("""## 3. Fine-tune `distilbert-base-uncased`

**Hyperparameter decision (documented in README):** the starter default is 3 epochs, but
on the validation set the macro-F1 was still climbing at epoch 3, so I trained for **4
epochs** with `load_best_model_at_end=True` + early-stopping on val macro-F1 — this
recovers the best checkpoint instead of risking an over-trained final one. Kept
`lr=2e-5`, `batch=16`, added `weight_decay=0.01` for light regularization on the small
(~150-example) train set."""))
cells.append(code("""from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}

args = TrainingArguments(
    output_dir="takemeter_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    seed=SEED,
    logging_steps=10,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_val,
                  compute_metrics=compute_metrics, tokenizer=tok,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()"""))

cells.append(md("## 4. Evaluate fine-tuned model + confusion matrix"))
cells.append(code("""pred_out = trainer.predict(ds_te)
ft_logits = pred_out.predictions
ft_pred_ids = np.argmax(ft_logits, axis=1)
ft_pred = [id2label[i] for i in ft_pred_ids]
# softmax confidence of the chosen class
ft_conf = torch.softmax(torch.tensor(ft_logits), dim=1).max(dim=1).values.tolist()

print("FINE-TUNED DistilBERT")
print(classification_report(y_te, ft_pred, labels=LABELS, zero_division=0, digits=3))

cm = confusion_matrix(y_te, ft_pred, labels=LABELS)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Fine-tuned DistilBERT — Confusion Matrix (test)")
thr = cm.max() / 2
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thr else "black")
fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150); plt.show()"""))

cells.append(md("""## 5. Zero-shot baseline — Groq `llama-3.3-70b-versatile`

Add your key via the 🔑 **Secrets** panel as `GROQ_API_KEY` (notebook access on).
The prompt uses the exact label definitions from `planning.md` and forces a one-word
answer so parsing is clean."""))
cells.append(code('''from google.colab import userdata
from groq import Groq
client = Groq(api_key=userdata.get("GROQ_API_KEY"))

SYSTEM = """You are a strict classifier for r/hiphopheads comments. Assign each comment
to EXACTLY ONE label. Reply with ONLY the label, lowercase, nothing else.

critique = a structured argument about the music itself, citing a specific identifiable
element (a bar, flow change, beat switch, producer, mix decision, track placement); the
reasoning survives removing the opinion.
hot_take = a bold confident verdict (ranking / GOAT / overrated-underrated) asserted
WITHOUT supporting musical evidence.
stan = an emotional fan reaction (hype, devotion, anticipation, letdown) where the
feeling is the point and there is little to no argument.

Rules: musical vocabulary used only to decorate a ranking is still hot_take. Naming
tracks without saying why is stan. Output one of: critique, hot_take, stan."""

def classify_groq(text):
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile", temperature=0, max_tokens=5,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}])
    out = r.choices[0].message.content.strip().lower()
    for l in LABELS:
        if l in out:
            return l
    return None

bl_pred, unparsed = [], 0
for t in X_te:
    p = classify_groq(t)
    if p is None:
        unparsed += 1; p = "hot_take"           # fallback so lengths match
    bl_pred.append(p)
print(f"unparsed responses: {unparsed}/{len(X_te)}")
print("BASELINE llama-3.3-70b (zero-shot)")
print(classification_report(y_te, bl_pred, labels=LABELS, zero_division=0, digits=3))'''))

cells.append(md("## 6. Compare + export"))
cells.append(code("""def pack(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                  zero_division=0)
    return {"accuracy": round(float(acc), 3),
            "macro_f1": round(float(np.mean(f1)), 3),
            "per_class": {LABELS[i]: {"precision": round(float(p[i]),3),
                                      "recall": round(float(r[i]),3),
                                      "f1": round(float(f1[i]),3),
                                      "support": int(s[i])} for i in range(len(LABELS))}}

ft_m, bl_m = pack(y_te, ft_pred), pack(y_te, bl_pred)
print(f"{'':18}{'fine-tuned':>12}{'baseline':>12}")
print(f"{'accuracy':18}{ft_m['accuracy']:>12}{bl_m['accuracy']:>12}")
print(f"{'macro_f1':18}{ft_m['macro_f1']:>12}{bl_m['macro_f1']:>12}")

out = {
  "labels": LABELS,
  "split": {"train": len(X_tr), "val": len(X_val), "test": len(X_te)},
  "seed": SEED,
  "fine_tuned": ft_m, "baseline": bl_m,
  "fine_tuned_confusion_matrix": {"labels": LABELS, "matrix": cm.tolist(),
      "note": "rows = true label, columns = predicted label"},
  "test_predictions": [{"text": t, "true": tr, "fine_tuned": fp, "baseline": bp}
      for t, tr, fp, bp in zip(X_te, y_te, ft_pred, bl_pred)],
  "fine_tuned_wrong": [{"text": t, "true": tr, "pred": fp, "confidence": round(float(c),3)}
      for t, tr, fp, c in zip(X_te, y_te, ft_pred, ft_conf) if tr != fp],
  "sample_classifications": [{"text": t, "true": tr, "pred": fp,
      "confidence": round(float(c),3), "correct": tr == fp}
      for t, tr, fp, c in zip(X_te, y_te, ft_pred, ft_conf)],
}
with open("evaluation_results.json", "w") as f:
    json.dump(out, f, indent=2)

trainer.save_model("takemeter_model"); tok.save_pretrained("takemeter_model")
print("Saved evaluation_results.json, confusion_matrix.png, takemeter_model/")
# Download artifacts:
# from google.colab import files; files.download('evaluation_results.json'); files.download('confusion_matrix.png')"""))

cells.append(md("## 7. (Stretch) interactive inference"))
cells.append(code('''def classify(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    i = int(probs.argmax())
    return id2label[i], float(probs[i])

for t in [
    "Kendrick's pocket on the second verse is unreal, he raps a half beat behind the snare so it sounds lazy and urgent at once.",
    "Drake is the most overrated artist of all time and history will be brutal to him.",
    "ALBUM OF THE YEAR no skips I've been screaming since midnight",
]:
    lab, conf = classify(t)
    print(f"[{lab:9} {conf:.0%}]  {t[:70]}")'''))

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU", "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("takemeter_finetune.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote takemeter_finetune.ipynb with {len(cells)} cells")
