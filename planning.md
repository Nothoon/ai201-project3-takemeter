# TakeMeter — Planning

> Working notes written **before** data collection (Milestones 1–2) and updated
> during annotation (Milestone 3). The polished, reader-facing version lives in
> `README.md`; this file is the design thinking behind it.

---

## 1. Community

**Chosen community: r/hiphopheads.**

r/hiphopheads is one of the largest music-discussion subreddits (3M+ subscribers),
and unlike a general music forum it has a strong, self-aware culture *about discourse
quality itself*. Regulars routinely call each other out for "glazing" (uncritical
hype), "yapping" (confident claims with no substance), and praise posts that "actually
break down" why something works. That community vocabulary is exactly the signal a
classifier can learn from.

**Why it's a good fit for a classification task:** the discourse is high-volume,
text-heavy, and *genuinely varied in kind* — not just in quality. On any given album
release day the same comment thread contains (a) people dissecting a beat switch or a
rhyme scheme, (b) people stating bold rankings with zero support ("X is washed"), and
(c) people posting raw, ecstatic reactions ("AOTY no skips I'm shaking"). Those three
modes look different enough to separate but overlap at the edges (an excited reaction
that *also* names a beat switch), which is what makes the task non-trivial rather than
keyword-matching.

---

## 2. Labels

Three mutually exclusive labels. The decision boundary for each is stated as a single
rule so two annotators reading it would agree on most posts.

### `critique`
**Definition:** The post makes a structured evaluative argument *about the music
itself* — citing a specific, identifiable element (a bar, a flow change, a beat switch,
a producer, a mix decision, a track placement) — such that the reasoning would still
stand if you deleted the opinion framing.

- *Example:* "Kendrick's pocket on the second verse is unreal — he's rapping a half
  beat behind the snare the whole time so it feels lazy and urgent at once."
- *Example:* "The sequencing kills the momentum — putting the two ambient interludes
  back to back at tracks 6 and 7 deflates everything the first half built up."

### `hot_take`
**Definition:** A bold, confident *verdict* — a ranking, a GOAT claim, an
"overrated/underrated" judgment — asserted **without** supporting musical evidence. The
claim may even be correct, but the post asserts rather than argues.

- *Example:* "MF DOOM is the most overrated rapper in the history of the genre, there I
  said it."
- *Example:* "Drake has never made a good album front to back and people are too scared
  to say it."

### `stan`
**Definition:** An emotional fan reaction — hype, devotion, anticipation, or
disappointment — where **the feeling is the point** and there is little to no argument
about the music's qualities.

- *Example:* "ALBUM OF THE YEAR no skips no notes I've been screaming since midnight."
- *Example:* "bro I am not okay after that last track, I had to sit in my car for ten
  minutes."

**Why these distinctions matter to the community:** r/hiphopheads explicitly polices
the line between "breaking it down" (critique), "yapping / hot takes" (hot_take), and
"glazing" (stan). The labels are the community's own categories, not imposed ones.

---

## 3. Hard edge cases

The genuinely ambiguous posts all sit on a boundary because they carry the *surface
markers* of one label and the *rhetorical structure* of another. Decision rules:

| Boundary | The trap | Decision rule |
|---|---|---|
| critique ↔ stan | Excitement that *names* a musical element ("the beat switch gave me chills") | If a **specific identifiable element is cited as the reason** for the feeling → `critique`. If the excitement is the whole content → `stan`. |
| critique ↔ hot_take | A verdict propped up by one decorative stat or a name-dropped term ("Igor's chord progressions prove he's the GOAT") | If the evidence is **genuine reasoning** about the music → `critique`. If it's **cherry-picked/decorative** to win a ranking → `hot_take`. |
| stan ↔ critique | A hype post that **lists track names** but never says why ("track 4 7 11 all bangers no skips") | Naming tracks **without evaluating** them → `stan`. |

**The single hardest anticipated case** (and how I'll handle it): an enthusiastic post
that name-drops a real musical element but uses it only as decoration —
*"Tyler is the best of his generation and Igor's chord progressions prove it."* It has a
critique-flavored noun ("chord progressions") bolted onto a `hot_take` verdict. **Rule:
if the cited element is not actually being reasoned about — if deleting the verdict
leaves no argument — it's a `hot_take`, not a `critique`.** I flagged this class of post
during the read-through precisely because I expected the model to over-trust musical
vocabulary, and it did (see README error analysis).

---

## 4. Data collection plan

- **Source:** public r/hiphopheads comment threads — release-day discussion megathreads,
  "Daily Discussion" threads, and album-rating posts. Public comments only; no DMs or
  private servers.
- **Method:** manual read-and-copy into a spreadsheet, ~1–2 hours. Manual collection
  keeps me close to the text so I notice boundary cases as I go (the whole point of
  Milestone 3).
- **Target volume & balance:** ≥200 examples, aiming for a roughly even split
  (~⅓ each) so no class dominates. Hard floor: no label above 70%; soft target: ≥30%
  each.
- **If a label is underrepresented after 200:** `critique` is the scarcest in the wild
  (most comments are reactions or takes), so if it lags I'll target threads that
  reliably produce it — production-breakdown posts, "underrated bars" threads, and
  "explain why this album is good" prompts — and collect more until it clears ~30%.

---

## 5. Evaluation metrics

Accuracy alone is insufficient because (a) the task is subjective, so a few points of
accuracy is within annotation noise, and (b) accuracy hides *which* boundary the model
fails on. The metrics, and why each is the right one here:

- **Overall accuracy** — headline number, and the only thing directly comparable to the
  zero-shot baseline.
- **Per-class precision / recall / F1** — the task's whole value is distinguishing three
  *specific* modes. A model that nails `stan` (easy, loud surface signal) but can't tell
  `critique` from `hot_take` would still post decent accuracy; only per-class F1 exposes
  that. F1 is the per-class headline because the classes are roughly balanced and I care
  equally about over- and under-prediction.
- **Confusion matrix** — tells me the *direction* of errors (is it critique→hot_take or
  hot_take→critique?), which is the actionable diagnostic for fixing labels or data.
- **Macro-F1** — single fairness-across-classes number, so a strong majority class can't
  paper over a weak one.

---

## 6. Definition of success

- **Minimum bar:** fine-tuned macro-F1 **≥ 0.70** with **no single class F1 below
  0.60**, and the fine-tuned model **beating the zero-shot baseline by ≥ 10 accuracy
  points** (otherwise fine-tuning didn't earn its keep).
- **"Good enough to deploy" in a real community tool** (e.g., a bot that tags or surfaces
  high-effort `critique` posts): I'd want **`critique` precision ≥ 0.80** specifically —
  for a "highlight the good breakdowns" feature, falsely promoting a hot_take as a
  critique is the costly error, so precision on that class matters more than recall.
- **Honesty check:** if accuracy comes back **> 0.95** on this subjective task, I'll
  assume label leakage or labels that are too easy, and re-inspect before trusting it.

---

## AI Tool Plan

There's no application code to generate in this project, so AI assistance is aimed at
the three places it actually helps:

1. **Label stress-testing.** Before annotating, I gave Claude my three definitions plus
   the edge-case rules and asked it to generate ~10 posts engineered to sit *between*
   two labels. The useful failures were posts that name-dropped musical terms inside a
   pure ranking — those forced me to add the explicit "decorative evidence → hot_take"
   rule in §3 before I touched the real data. **I verify by labeling its generated posts
   myself; if I can't place one cleanly, the definitions need tightening, not the post.**

2. **Annotation assistance.** I considered LLM pre-labeling a batch. **Decision: use it
   for a single ~40-post batch only, then hand-review and correct every label** (genuine
   review, not skimming). Pre-labeled rows are tracked with a note so they're disclosed
   in the README AI-usage section. The remaining ~170 are labeled by hand first.

3. **Failure-pattern analysis.** After evaluation I'll paste the list of misclassified
   test posts into an LLM and ask it to find a *systematic* pattern (a label pair, a post
   length, a vocabulary cue) — then **re-read each example myself to confirm or discard
   the proposed pattern** before it goes in the report. The pattern only counts if I can
   point to the specific posts that show it.

---

## Annotation log — difficult cases encountered (updated during Milestone 3)

Running list of posts that genuinely gave me pause and what I decided. (Three+ required;
these also seed the README "difficult examples" section.)

1. **"This verse gave me literal chills — the way he flips from triplets into a straight
   flow right at the bridge is insane."** Could be `stan` (chills, "insane"). Cites a
   *specific* element (triplet→straight flow flip at the bridge) as the reason. → **`critique`**
   per the "specific element is the cause of the feeling" rule.

2. **"He's got more #1 albums than anyone this decade so the GOAT debate is over."**
   Has a real chart fact → looks like `critique`. But the stat is decorative, selected to
   crown a ranking, not to reason about the music. → **`hot_take`**.

3. **"Album of the year, track 4 7 and 11 are all bangers, literally no skips."** Names
   specific tracks → looks like `critique`. No argument about *why* they work, just
   enthusiastic listing. → **`stan`**.

4. **"Tyler is the best of his generation and Igor's chord progressions prove he's on
   another level."** Gestures at chords (critique-flavored) but the chord claim is
   unsupported and bolted to a GOAT verdict. The verdict is the point. → **`hot_take`**.

5. **"I cannot believe how clean the mix is — every instrument has its own space and
   nothing fights for room."** High enthusiasm reads `stan`, but it's a specific,
   verifiable claim about the mix (instrument separation). Reasoning survives deleting
   the excitement. → **`critique`**.

---

## Stretch features (update before starting any)

- [ ] Inter-annotator reliability (Cohen's κ on 30+ shared examples)
- [x] **Deployed interface** — `classify.py` CLI + `app.py` (Gradio) load the fine-tuned
      model and print label + confidence. (Built; see README "Deployed Interface".)
- [ ] Confidence calibration study
- [x] **Error-pattern analysis** — systematic pattern identified (surface-vocabulary
      over-reliance), documented in README rather than just listing wrong predictions.
