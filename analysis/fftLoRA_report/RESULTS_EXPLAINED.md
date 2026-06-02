# The results, explained simply

*A plain-language companion to `REPORT.md`. No background assumed. Terms are defined the
first time they appear, with a glossary at the end.*

---

## What this project is

We have a **real robot arm** (a UF850) and an **AI model** (π0.5) that looks at two
camera images plus the arm's current position and decides **how to move the arm next**.
We taught it four **pick-and-place** jobs — pick up a *bottle*, a *stuffed bear*, a
*rubber duck*, and a *hat*, and set each on a box — by recording a human doing them many
times and training the model to imitate.

The project asks a "skills library" question. Instead of one giant model per job, could we
keep **one general model** and bolt on a small, swappable **"skill add-on"** for each job?
A skill add-on (technically a *LoRA*) is just a small set of extra numbers added to the big
model that nudges it toward one task. If skills behave like clean building blocks, you could
mix, match, and reuse them. We tested whether that dream holds up.

**How we measure "good":** the model predicts the arm's next joint angles; we compare its
prediction to what the human actually did and report the **average error in degrees**.
Smaller is better. For scale, **1° is very precise** — a clock's minute hand moves 6° per
minute. We always test on **held-out** recordings: episodes the model (or add-on) **never
saw during training**, which is the honest test of whether it *learned* vs just *memorized*.

---

## Result 1 — The general model is strong, and training it longer didn't hurt
*(figure: `clean_7_checkpoint.png`)*

We fully trained the model on all four jobs (8.5 hours on 4 high-end GPUs). A common worry
is **over-memorizing**: train too long and the model aces its practice data but gets worse
on anything new. We checked by saving snapshots throughout training and testing each on
held-out data. **It never over-memorized** — accuracy on new episodes kept improving all the
way to the end. So we use the final, best snapshot as the base for everything else.
(*We made it robust on purpose by randomly varying brightness, color, and blur during
training so it doesn't depend on perfect lighting.*)

## Result 2 — This way of training beats the alternatives
*(figure: `clean_6_benchmark.png`)*

We compared four ways of training, on held-out episodes, in normal conditions **and** under
visual disturbance (we digitally dim/recolor/blur the camera to mimic a different room):

| Method | normal error | disturbed error | does all 4 objects? |
|---|---|---|---|
| **This model** (full retrain on 4 jobs) ★ | **1.7°** | **2.1°** | ✓ |
| GSE (a fancier alternative) | 1.9° | 2.5° | ✓ |
| Older model (trained on bottle only) | 5.6° | 7.0° | ✗ — fails bear & hat |
| LoRA-only on bottle | 11.4° | 11.7° | ✗ — fails everything else |

**This model is the most accurate, in normal conditions *and* under disturbance**, and is
one of only two that can do all four objects. The bottle-only models are useless on objects
they never trained on (you can't do a task you never learned). Takeaway: **train on diverse
data, and a full retrain wins here.**

## Result 3 — Adding a per-task "skill add-on" genuinely helps
*(figure: `clean_1_skills_help.png`)*

For each object we trained a small skill add-on on top of the (frozen) general model, then
tested it on held-out episodes of that object it had **never seen**:

| Object | general model | + skill add-on | improvement |
|---|---|---|---|
| bottle | 0.96° | **0.56°** | **42% better** |
| bear | 0.92° | 0.63° | 32% better |
| duck | 1.15° | 0.66° | 43% better |
| hat | 0.95° | 0.64° | 33% better |

**The skill add-on makes the robot ~37% more precise on each task, and it generalizes** —
it helps on new episodes, not just the ones it trained on. So as a way to *specialize*, the
skill-add-on idea **works**.

## Result 4 — But the add-ons are *not* tidy, comparable "skill vectors" (the surprise)
*(figures: `clean_2_fingerprint.png`, `clean_3_skill_map.png`, `clean_4_where.png`)*

The dream was that **similar tasks would produce similar add-ons** — that you could line them
up like Lego, or do "bottle − bear + duck" arithmetic, the way image AIs can represent
"wearing a hat" as a direction you add or remove. We measured **similarity** between add-ons
on a 0–1 scale (1 = identical, 0 = unrelated).

**The surprise** (`clean_2_fingerprint.png`): we trained the **exact same bottle task twice**,
changing only the **random starting point** of training (the "seed" — the dice-roll for the
add-on's initial numbers). The two add-ons came out **completely unrelated (0.03)** — even
though they do the *same* job. Meanwhile, two add-ons for *different* objects looked *more*
alike (0.62) simply because they shared the same starting point.

> **Analogy.** Two students get the same textbook but start their notes from different random
> scribbles. Their finished notes look nothing alike on paper, even though they learned the
> same thing. The add-on's raw numbers are like the scribbles: they encode the skill, but
> what they *look like* is mostly an accident of where training started — not a fingerprint
> of the task.

This matters because it would be **easy to be fooled**: if we hadn't run that "same task,
different start" check, we'd have seen the 0.62 and wrongly concluded "the tasks are alike."
The check is the whole point.

**Is there *any* task signal?** Yes, a faint one. Once we mathematically remove the shared
"starting-point" direction, two add-ons for the *same* object are clearly more alike than for
different objects. And the part of the model that handles **"which object this is"** (its
vision/language part) changes more from object to object than the part that handles **the arm
motion** (`clean_4_where.png`) — which makes sense: *what* to grab differs by object, but *how*
to pick-and-place is similar. So the task **is** in there, just buried under the random-start
noise, and mostly in the "what to grab" part.

---

## What it all means

- **Functionally, skills work:** the general model is strong (Results 1–2), and bolting on a
  task add-on reliably sharpens it (Result 3).
- **But the add-on's raw numbers are *not* a clean "skill coordinate"** (Result 4): they're
  dominated by the random starting point, so you can't reliably compare, average, or do
  arithmetic on them across separately-trained add-ons.
- **For a real skills library, the fix is:** either start every skill from the *same* fixed
  point (so they're comparable), or compare skills by **what they do** (behavior), not by
  their raw numbers. The "what to grab" vs "how to move" split is a promising place to look.

The genuinely useful, non-obvious finding: **a LoRA is a real functional skill, but not a
faithful skill *vector* — and you only discover that if you run the right control.**

## Honest limitations

- We measure **move-prediction error offline**, not success on the physical arm — it's a
  strong proxy, but a real robot trial is the gold standard.
- Only **4 objects**, so we can't claim things like "all soft toys cluster together."
- The key "same task, different start" check was run once (for bottle); repeating it for every
  object would make it airtight.
- The bear/duck/hat held-out test sets are small (5–20 episodes); bottle (60) is the solid one.

---

## Glossary

- **AI model (π0.5):** a ~3.3-billion-number network that turns camera images + arm position
  into the arm's next move.
- **Full fine-tune:** retrain *all* of the model's numbers on your data (most powerful, most
  expensive).
- **LoRA / skill add-on:** a *small* set of extra numbers added on top of a frozen model to
  specialize it for one task — cheap and swappable.
- **Move-prediction error (degrees):** how far the model's predicted joint angles are from the
  human demonstration. Lower = better; ~1° is very precise.
- **Held-out:** test data never seen in training — the honest measure of generalization.
- **Generalize vs memorize:** doing well on *new* data vs only on the practice data.
- **Similarity (cosine):** 0–1 score for how alike two sets of numbers are — 1 identical, 0
  unrelated.
- **Random start / seed:** training begins the add-on's numbers at random values; the seed is
  the dice-roll that picks them.
- **Domain shift / disturbance:** changing the look of the scene (lighting, color, blur) to
  test robustness to a new environment.
