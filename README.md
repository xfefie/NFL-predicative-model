# NFL-predicative-model
Project to make an NFL and CFB predictive model based on manual input of an avg. of a 6 second latency input.


# SituationIQ — Manual Model (README)

SituationIQ’s **Manual Model** is the “coach-first” rules + analytics engine that generates fast, explainable recommendations (sub-6 seconds) without needing a fully-trained ML pipeline. It’s designed to work immediately with limited data, then improve as you collect more play-by-play and tagging.

---

## What this is

The Manual Model is a **highly tested, deterministic** decision layer that:
- turns football game context into probabilities + recommendations
- produces **clear “why” explanations** (the part coaches trust)
- can run offline / locally
- acts as the baseline that future ML models must beat

Think: **Decision table + priors + situational adjustments + sanity checks**.

---

## Core outputs

Depending on your configuration, the Manual Model can output:

### In-game decision recommendations
- **Go / No-Go (4th down)**
- **2-Point attempt vs PAT** (only after a score)
- **Punt vs Field Goal vs Go** (4th down)
- **Timeout / clock management prompts** (optional)

### Situation summaries (coach-facing)
- Offense tendency snapshot (last N plays + game-level)
- Defense tendency snapshot (shells/pressure tendencies if tagged)
- “Tendency breaker” suggestions (simple rules; optional)

---

## How it works (high level)

The Manual Model combines:

1. **Base tables / priors**
   - e.g., historical conversion rates by down-distance
   - FG make probability by distance + hash + weather (optional)
   - punt net expectations by field position (optional)

2. **Context modifiers**
   - score differential, time remaining, quarter
   - field position, timeouts
   - opponent strength adjustments (optional)
   - momentum is *not* used unless you explicitly add it (keeps it grounded)

3. **Value framework**
   - expected points / win probability delta
   - risk preference sliders (coach style)
   - hard constraints (don’t recommend impossible / nonsensical actions)

4. **Explainability layer**
   - the “top 3 reasons” behind the recommendation
   - comparable scenarios (“similar situations: 4Q, +3, own 42…”)



