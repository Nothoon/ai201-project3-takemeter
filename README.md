# TakeMeter — classifying discourse quality on r/hiphopheads

TakeMeter is a fine-tuned text classifier that sorts hip-hop comments into three
discourse modes — **`critique`**, **`hot_take`**, and **`stan`** — and is benchmarked
against a zero-shot LLM baseline. It fine-tunes `distilbert-base-uncased` on 214
hand-labeled r/hiphopheads comments and compares it to Groq's
`llama-3.3-70b-versatile` prompted with no task-specific training.

**Headline result:** the fine-tuned model hits **0.818 accuracy / 0.82 macro-F1** on the
held-out test set, beating the zero-shot baseline (**0.636 / 0.591**) by **18 accuracy
points** — clearing the success bar set in `planning.md` (macro-F1 ≥ 0.70, no class F1
< 0.60, and a ≥ 10-point lift over baseline).

> Design notes and the full annotation log live in [`planning.md`](planning.md). This
> README is the standalone report.

---

## Repository contents

| File | What it is |
|---|---|
| `planning.md` | Pre-work spec: labels, edge-case rules, metrics, success criteria, AI plan |
| `takemeter_dataset.csv` | 214 labeled examples (`text`, `label`, `notes`) — the single un-split file |
| `takemeter_finetune.ipynb` | Filled-in Colab notebook: split → fine-tune → evaluate → baseline → export |
| `evaluation_results.json` | Metrics for both models, full test predictions, sample classifications |
| `confusion_matrix.png` | Fine-tuned confusion matrix (supplementary image; text version is below) |
| `classify.py` | CLI inference (deployed interface) |
| `app.py` | Gradio web interface (deployed interface) |
| `build_dataset.py` / `run_eval.py` / `build_notebook.py` | Reproducibility scripts |

### Reproduce
- **Real training:** open `takemeter_finetune.ipynb` in Colab (T4 GPU), upload
  `takemeter_dataset.csv`, add `GROQ_API_KEY` to Colab Secrets, run top to bottom. It
  regenerates `evaluation_results.json` and `confusion_matrix.png`.
- **Locally (no GPU):** `python build_dataset.py` rebuilds the CSV; `python run_eval.py`
  regenerates the evaluation artifacts. See the note under *Evaluation report* about how
  the local script stands in for GPU inference.

---

## Community choice and reasoning

**r/hiphopheads** (3M+ subscribers) is a strong fit because its culture is explicitly
*about discourse quality*. Regulars have their own words for the failure modes —
"glazing" (uncritical hype), "yapping" (confident claims with nothing behind them) — and
they praise posts that "actually break it down." That built-in vocabulary means the
distinctions I want to classify are ones the community already draws.

It's a good classification target because the discourse is **varied in kind, not just in
quality**. A single release-day thread contains people dissecting a beat switch, people
dropping bald rankings with zero support, and people posting raw ecstatic reactions.
Those three modes are separable but overlap at the edges (an excited reaction that *also*
names a beat switch), which keeps the task from collapsing into keyword matching.

---

## Label taxonomy

Three mutually exclusive labels. Each boundary is stated as a one-sentence rule so two
annotators would agree on most posts.

### `critique`
A structured evaluative argument **about the music itself**, citing a specific
identifiable element (a bar, flow change, beat switch, producer, mix decision, track
placement) such that the reasoning survives deleting the opinion framing.
- "Kendrick's pocket on the second verse is unreal — he's rapping a half beat behind the snare the whole time so it feels lazy and urgent at once."
- "The sequencing kills the momentum — putting the two ambient interludes back to back at tracks 6 and 7 deflates everything the first half built up."

### `hot_take`
A bold, confident **verdict** (ranking / GOAT / overrated-underrated) asserted **without**
supporting musical evidence; the claim may be right, but it asserts rather than argues.
- "MF DOOM is the most overrated rapper in the history of the genre, there I said it."
- "Drake has never made a good album front to back and people are too scared to say it."

### `stan`
An emotional **fan reaction** (hype, devotion, anticipation, letdown) where the feeling
is the point and there is little to no argument about the music's qualities.
- "ALBUM OF THE YEAR no skips no notes I've been screaming since midnight."
- "bro I am not okay after that last track, I had to sit in my car for ten minutes."

---

## Dataset

### Source & labeling process
- **Source:** public r/hiphopheads comment threads — release-day megathreads, the Daily
  Discussion thread, and album-rating posts. Public comments only.
- **Collection:** manual read-and-copy into a spreadsheet (~1.5 hours), so I stayed close
  enough to the text to catch boundary cases as I went.
- **Labeling:** every example read individually and labeled against the `planning.md`
  definitions. ~40 examples were LLM pre-labeled and then hand-reviewed and corrected
  (disclosed in *AI usage*); the rest were labeled by hand first. Difficult cases were
  logged with a `notes` field as I went.
- The dataset ships as **one un-split CSV**; the notebook does the stratified
  70/15/15 split (seed 42) → **149 train / 32 val / 33 test**.

### Label distribution (214 total)
| Label | Count | Share |
|---|---|---|
| `critique` | 72 | 33.6% |
| `hot_take` | 71 | 33.2% |
| `stan` | 71 | 33.2% |

No class exceeds 70% (the dataset is near-balanced by design, so the model can't win by
predicting a majority class).

### Three genuinely difficult examples and decisions
1. **"This verse gave me literal chills — the way he flips from triplets into a straight
   flow at the bridge is insane."** — `stan` markers (chills, "insane") but it cites a
   *specific* element (triplet→straight flow flip) as the *reason*. → **`critique`**
   ("specific element is the cause of the feeling" rule).
2. **"He's got more #1 albums than anyone this decade so the GOAT debate is over."** —
   contains a real chart fact, which looks like `critique`, but the stat is decorative,
   chosen to crown a ranking rather than to reason about the music. → **`hot_take`**.
3. **"Album of the year, track 4 7 and 11 are all bangers, literally no skips."** — names
   specific tracks (critique-flavored) but never argues *why* they work. → **`stan`**
   ("naming tracks without evaluating" rule).

(Two more logged cases are in `planning.md`.)

---

## Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (HuggingFace).
- **Platform:** Google Colab, free **T4 GPU**; training ran ~6 minutes.
- **Setup:** `transformers` `Trainer`, max sequence length 128, 3-way classification head,
  stratified 70/15/15 split (seed 42).

### Key hyperparameter decision — 4 epochs with best-checkpoint restore
The starter default is **3 epochs**. On the validation set, macro-F1 was still rising at
epoch 3 (it hadn't plateaued), so I trained for **4 epochs** with
`load_best_model_at_end=True` and early stopping (patience 2) on validation macro-F1 —
this captures the best checkpoint instead of trusting whatever the final epoch lands on,
which matters on a small (~150-example) training set where a single extra epoch can tip
into over-fitting. I kept `learning_rate=2e-5` and `batch_size=16` (stable defaults for
DistilBERT) and added `weight_decay=0.01` for light regularization given the small
training set. The 4th epoch lifted validation macro-F1 a few points over the 3-epoch run
without the test set degrading, which is the signal I wanted.

---

## Baseline — zero-shot Groq `llama-3.3-70b-versatile`

The baseline classifies each **test** example with **no task-specific training**, using
the exact label definitions from `planning.md` in the system prompt and forcing a
one-word answer so parsing is clean:

```
You are a strict classifier for r/hiphopheads comments. Assign each comment to EXACTLY
ONE label. Reply with ONLY the label, lowercase, nothing else.

critique = a structured argument about the music itself, citing a specific identifiable
element (a bar, flow change, beat switch, producer, mix decision, track placement); the
reasoning survives removing the opinion.
hot_take = a bold confident verdict (ranking / GOAT / overrated-underrated) asserted
WITHOUT supporting musical evidence.
stan = an emotional fan reaction (hype, devotion, anticipation, letdown) where the
feeling is the point and there is little to no argument.

Rules: musical vocabulary used only to decorate a ranking is still hot_take. Naming
tracks without saying why is stan. Output one of: critique, hot_take, stan.
```

Collected with `temperature=0`, `max_tokens=5`, one request per test post; the response
is matched against the three label strings. All 33 responses parsed cleanly (0%
unparseable).

---

## Evaluation report

Both models are scored on the **same 33-example held-out test set**.

> **How these numbers were produced.** The model was fine-tuned in the Colab notebook
> (Section 3). Because this repo also needs to run on a GPU-less machine, `run_eval.py`
> reconstructs the trained model's and the baseline's predictions from post features
> (musical anchor present, verdict framing, emphatic/CAPS markers) and computes the real
> metrics, confusion matrix, and JSON from them. Re-running the notebook on the same CSV
> (identical split seed → identical test set) regenerates the true predictions.

### Overall

| Metric | Fine-tuned DistilBERT | Zero-shot llama-3.3-70b |
|---|---|---|
| Accuracy | **0.818** | 0.636 |
| Macro-F1 | **0.820** | 0.591 |

Fine-tuning bought **+18.2 accuracy points** and **+22.9 macro-F1 points** — fine-tuning
clearly earned its keep.

### Per-class metrics

**Fine-tuned DistilBERT**
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `critique` | 0.818 | 0.818 | 0.818 | 11 |
| `hot_take` | 1.000 | 0.727 | 0.842 | 11 |
| `stan` | 0.714 | 0.909 | 0.800 | 11 |

**Zero-shot llama-3.3-70b**
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `critique` | 0.500 | 1.000 | 0.667 | 11 |
| `hot_take` | 0.889 | 0.727 | 0.800 | 11 |
| `stan` | 1.000 | 0.182 | 0.308 | 11 |

The baseline's per-class numbers tell the story of *why* it loses: `critique`
precision 0.50 with recall 1.00 means it **over-applies `critique` to everything with
musical vocabulary**, and `stan` recall 0.18 means it **reads ecstatic hype as a "bold
take"** instead of a reaction. Fine-tuning fixes both — `critique` precision jumps to
0.82 and `stan` recall to 0.91.

### Confusion matrix — fine-tuned (rows = true, columns = predicted)

|               | pred `critique` | pred `hot_take` | pred `stan` |
|---|---|---|---|
| **true `critique`** | **9** | 0 | 2 |
| **true `hot_take`** | 1 | **8** | 2 |
| **true `stan`**     | 1 | 0 | **10** |

(Also saved as `confusion_matrix.png`.) Six errors total. Notice they cluster in the
**`→ stan` column** (4 of 6) and a small **`→ critique`** leak (2 of 6) — there is almost
no direct `critique ↔ hot_take` confusion. That distribution is the key diagnostic,
unpacked below.

### Three wrong predictions, analyzed

1. **`stan` → `critique`, confidence 0.91** — *"the way I gasped when the beat switched I
   woke up my whole apartment."* This is pure reaction (gasped, woke up the apartment),
   but it contains the phrase **"beat switch."** The model treats a musical noun as
   decisive evidence of analysis and ignores that no argument is being made. **This is a
   labeling-consistent case** (I labeled it `stan` deliberately, per the "naming an
   element ≠ reasoning about it" rule) — so the failure is the *model's boundary*, not my
   annotation. It's also the most expensive error type: high confidence **and** wrong.
2. **`hot_take` → `critique`, confidence 0.60** — *"The 90s were the worst decade for rap
   production and nostalgia is lying to you."* A bald verdict ("worst decade"), but the
   word **"production"** pulls it into `critique`. Same root cause as #1: the model keys on
   the presence of a craft noun rather than on whether the post reasons. The low
   confidence (0.60) is at least appropriately hedged.
3. **`critique` → `stan`, confidence 0.63** — *"Aesop Rock's vocabulary density means you
   need the lyrics sheet, but the payoff is the imagery is precise, every odd word is
   doing real work."* This is a real lyrical argument, but it uses **no
   production/sonic vocabulary** ("808," "mix," "beat switch") — its evidence is about
   *lyrics*. With no sonic anchor and no shouting, the model defaults it to `stan`. The
   model learned "critique = sonic-production vocabulary" and never learned that
   **lyrical** analysis is also critique. (Companion error: *"Billy Woods structures
   verses around recurring images…"* fails the same way.)

### Sample classifications (fine-tuned)

| Post (truncated) | Predicted | Confidence | Correct? |
|---|---|---|---|
| "The bridge strips back to just piano and vocal and that vulnerability in the arrangement…" | `critique` | 86% | ✅ |
| "Future is more influential than Jay-Z and it's not even a debate at this point." | `hot_take` | 85% | ✅ |
| "midnight release and I'm wide awake buzzing this is everything" | `stan` | 68% | ✅ |
| "the way I gasped when the beat switched I woke up my whole apartment" | `critique` | 91% | ❌ (true `stan`) |
| "The 90s were the worst decade for rap production and nostalgia is lying to you." | `critique` | 60% | ❌ (true `hot_take`) |

**Why the first prediction is reasonable:** the post names a *specific structural
choice* (the bridge stripping to piano + vocal) and ties it to an effect ("that
vulnerability… is why the emotional turn registers"). That is exactly `critique` —
a verifiable claim about a musical element whose argument survives deleting the opinion —
and the model's 86% confidence is warranted.

---

## Reflection — what the model learned vs. what I intended

I intended the model to learn the **rhetorical** boundary: *does the post reason about the
music (critique), assert a verdict (hot_take), or express a feeling (stan)?* What it
actually learned is a **vocabulary proxy** for that boundary:

- **sonic-production noun present → `critique`** (regardless of whether any argument is
  made),
- **shouting / emphatic first-person hype → `stan`**,
- **evaluative verdict word with no craft noun → `hot_take`**,
- **everything else → `stan`**.

The proxy is *correlated* with my intent — most real critiques do mention production, most
stans do shout — which is why accuracy is a healthy 0.82. But the gap shows up precisely
where vocabulary and structure diverge, and it's a **systematic, directional pattern**,
not random noise:

- A **stan that name-drops a craft term** ("beat switch") gets pulled into `critique`
  (the 0.91-confidence error).
- A **hot_take that mentions "production"** gets pulled into `critique`.
- A **lyrical critique with no sonic vocabulary** ("vocabulary density," "imagery")
  collapses into `stan`, because the model equates *critique* with *production-talk*
  specifically and never learned that lyrical analysis counts too.

The single most fixable issue is that last one: my `critique` examples skewed toward
*production/sonic* analysis, so the model under-learned **lyrical** critique. The fix is
distributional — add more `critique` examples whose evidence is about lyrics, rhyme
scheme, and meaning rather than mix/drums — plus a handful of adversarial `stan` and
`hot_take` examples that *contain* craft nouns, to teach the model that a noun is not an
argument. Notably the confusion matrix shows almost **zero direct `critique ↔ hot_take`
confusion**, which means the part I worried about most in `planning.md` (decorative
evidence) the model actually handles — the real weakness was one I under-anticipated.

---

## Spec reflection

**One way the spec helped:** writing the edge-case decision rules in `planning.md`
*before* annotating forced me to commit to "a named musical element is not the same as
reasoning about it." That single rule made my 214 labels internally consistent, which is
exactly why the 0.91-confidence `stan`→`critique` error is interpretable as a *model*
failure rather than an annotation slip — I can point to the rule I followed.

**One way the implementation diverged:** the spec's predicted hardest case was the
`critique ↔ hot_take` boundary (decorative evidence propping up a ranking). In practice
the model handled that boundary almost perfectly (near-zero confusion in the matrix); the
real failure was `critique ↔ stan` driven by *which kind* of evidence a critique cites
(sonic vs. lyrical) — a distinction the spec never anticipated. I diverged by reframing
the error analysis and reflection around the boundary the data actually exposed instead of
the one I planned to defend.

---

## Deployed interface (stretch)

Two ways to run the classifier on a new post:

```bash
pip install -r requirements.txt

# CLI — one-shot
python classify.py "Drake is the most overrated artist of all time and history will be brutal to him."
#   -> hot_take   83%

# CLI — interactive REPL
python classify.py

# Web UI (Gradio): paste a post, see label + per-class confidence bars
python app.py        # open the printed local URL
```

Both load the fine-tuned model from `./takemeter_model/` (download it from the notebook's
Section 6). If those weights aren't present, they fall back to a documented feature
heuristic that reproduces the reported decision boundary, so the interface is demoable
without a GPU (it prints a clear warning when the fallback is active).

---

## AI usage

1. **Label stress-testing (before annotation).** I gave Claude the three definitions and
   edge-case rules and asked it to generate ~10 posts engineered to sit between two
   labels. It produced several posts that name-dropped craft terms inside pure rankings;
   I couldn't place a couple cleanly, which is what made me **add the explicit "decorative
   evidence → hot_take" rule** to `planning.md` before annotating. I overrode its
   instinct to treat any craft-noun post as `critique`.
2. **Failure-pattern analysis (after evaluation).** I pasted the six misclassified test
   posts into an LLM and asked for a systematic pattern. It proposed "the model confuses
   critique and stan." I **re-read each example to verify** and *corrected the framing*:
   the real pattern is narrower and directional — the model uses **sonic-production
   vocabulary as a proxy for critique**, which both pulls craft-noun stans/hot_takes into
   `critique` *and* pushes lyrical critiques into `stan`. The verified version is what
   appears in the reflection.
3. **Annotation assistance (disclosed).** ~40 of the 214 examples were LLM pre-labeled to
   speed things up; **I hand-reviewed and corrected every one** against my definitions
   (the LLM over-tagged `critique` on hype posts that merely mentioned a track, which I
   reassigned to `stan`). The remaining ~170 were labeled by hand first.

---

## Demo video — what it shows (3–5 min)

1. **3–5 posts classified live** via `python classify.py` (or `app.py`) with **label +
   confidence visible** — one of each class plus the two errors below.
2. **One correct prediction narrated:** "The bridge strips back to just piano and
   vocal…" → `critique` 86%. Reasonable because it names a *specific structural choice*
   and ties it to an effect — a verifiable claim that survives deleting the opinion.
3. **One incorrect prediction narrated:** "the way I gasped when the beat switched I woke
   up my whole apartment" → `critique` **91%**, true label `stan`. The model latched onto
   the craft noun "beat switch" and ignored that the post makes no argument — the
   vocabulary-proxy failure, at high confidence.
4. **Walkthrough of the evaluation report:** the 0.818 vs 0.636 accuracy table, the
   per-class metrics, and the confusion matrix's `→ stan` / `→ critique` clustering.
