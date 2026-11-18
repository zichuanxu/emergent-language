# Why a first reproduction failed, and what it takes to reproduce the paper

Paper: Mordatch & Abbeel, *Emergence of Grounded Compositional Language in Multi-Agent Populations* (arXiv:1703.04908v2).
Scope: an earlier, careful attempt at this reproduction, referred to as **v1**, and the implementation in this repository, referred to as **v2**. v1 is not included in the repository; file and line references of the form `src/...`, `scripts/...` and `results/...` in §1–2 point into it and are kept as a record of where each problem was.

This file has three parts: why v1 did not reproduce the paper (§1–2), what a faithful and working implementation needs (§3), and what remains open (§4). Measured results of this repository's implementation are in [`RESULTS.md`](RESULTS.md).

## 1. Verdict on v1

v1 is careful engineering around a training setup that could not have produced the paper's language. Its simulator, policy wiring, estimator and evaluation harness are correct: an independent line-by-line pass found no sign, index, masking, leak or detach error in `src/world.py`, `src/policy.py` or `src/train.py`, and its paired evaluation design is sound. Its headline numbers are accurately reported.

What failed is optimization, for three reasons that compound. Each is demonstrated below with a measurement rather than argued.

| # | Cause | Where | Evidence |
|---|---|---|---|
| 1 | The networks start in a near-linear regime, so the one computation the task hinges on (bind a location to a landmark's colour) sits on a long plateau | PyTorch's default `nn.Linear` init in `src/policy.py:61-69` | Under **full supervision**, v1's own policy is still at chance (34%) after 500 updates at naming the colour of the landmark at its goal location. Re-initialising only its hidden layers (He-normal) gives 98% by update 400. `analysis/prior_code_probe_*.log` |
| 2 | The vocabulary and utterance penalties act at full weight from update 1 | `src/train.py:386-388` | Active words fall 19 → 1 by update 133 (seed 1) and 19 → 0 by update 112 (seed 2), before any use for words exists. `results/primary/training/aligned_1x1x3*/metrics.csv` |
| 3 | The auxiliary task was implemented in a way that cannot, by itself, reward landmark reference, and v1 had nothing else that could bootstrap it | `src/world.py:295-301`, `src/train.py:162-168` | In `aligned_long_delay` (penalties postponed) the auxiliary reward reaches −0.02 while physical reward stays exactly at the centroid value: the agents learned to transmit the speaker's private-frame coordinates through the relaxed channel, which a listener cannot use. |

Cause 2 is sufficient on its own: in the rebuilt implementation, with everything else fixed, penalties from the first update leave zero active words in both seeds. Cause 1 alone is not fatal but costs time: with PyTorch's default initialisation and delayed penalties, communication emerges at update 2,000–2,500 instead of about 1,100, which is past the point where v1's penalties had already emptied the channel (ablations in `RESULTS.md` §7). Cause 3 is subtler: the same literal auxiliary target *does* bootstrap grounded language once cause 1 is removed (§3.2), so the lesson is not "the paper's auxiliary task is wrong" but "it only works when the networks can learn the binding in the time it buys".

## 2. Detailed findings on v1

### 2.1 Optimization (why no landmark words formed)

**Initialisation.** The paper does not state an initialiser. TensorFlow's fully-connected layer of that period defaulted to Glorot-uniform weights and zero biases, which puts pre-activations at order one. PyTorch's `nn.Linear` default draws weights with a third of that variance. With 552 inputs to the action module, v1's hidden pre-activations are about 0.15 in magnitude, where ELU is almost linear. Colour-of-the-landmark-at-my-goal is a pure interaction between two inputs (neither the goal location nor the pooled landmark features predicts it alone), so a near-linear network has almost no gradient towards it. Measured on v2's architecture, which has the same module structure:

| Probe (supervised, 1,024 worlds per update) | PyTorch default | He-normal hidden layers | Glorot-uniform (TF default) |
|---|---:|---:|---:|
| Speaker names the colour at its goal location: accuracy at update 200 | 0.33 (chance) | 0.96 | 0.98 by update 500 |
| Same, update 700 | 0.33 | 0.99 | — |
| Same, update 3,000 | 0.96 (plateau ends near 2,500) | — | — |
| Listener given oracle colour words: terminal error at update 200 | −0.46 (centroid) | −0.04 | −0.02 by update 300 |

The same probe run on v1's own `Policy` class gives chance at update 500 with default init and 98% at update 400 with He-normal hidden layers. Scaling the goal vector's input by 16 has the same effect as re-initialising, which confirms the mechanism: the two goal-location inputs are swamped by 512 pooled features.

**Penalty timing.** With Adam, a small loss term that is the only consistent force on a parameter moves it at the full learning rate. An independent measurement on v1's initial checkpoint found the gradient norm on the symbol-logit head to be 1.4e-3 from the auxiliary task, 6.5e-5 from the vocabulary reward, 4.6e-5 from the utterance cost and 1.9e-5 from physical reward. The physical reward, the only term that cares what words mean, is the weakest force on what gets said, and the vocabulary reward collapses the channel in about 130 updates.

**Silence counted as a word.** v1 includes the silence symbol in the Dirichlet-process counts (`src/train.py:151`, `:187-190`, `:383-385`). The paper counts symbols that were *uttered*. With silence inside the rich-get-richer dynamics it either dies or takes everything: three of the four collapsed primary policies emit one constant non-silence token every step, so "agents omit redundant utterances" could not appear.

**Auxiliary read-out.** v1 computes the auxiliary prediction in a separate pass after the episode (`src/train.py:162-168`), from the token emitted at the last step, which no action ever consumes. The three-agent policy moved action and addressee information into that dead slot: decoding from the first three tokens gives 0.68, from all six 0.99.

**DO NOTHING reward scale.** `src/world.py:342-343` scores DO NOTHING as squared speed. Forces are unbounded and agents cross the arena in two steps at speeds near 7, so a mistaken dash costs about 50 per step against at most 2 for a missed GO TO. The three-agent "communication gain" in v1 is mostly a go/stay switch.

**Not the cause.** bfloat16 autocast (`src/train.py:142`), the 6-step horizon, and pooling an agent's own utterance with its peers' all deviate from or are unspecified by the paper, but v1's float32 pilots, its 12-step run and its peer-only runs failed in the same way.

### 2.2 Reporting

The quantitative claims in v1's results file were checked against the stored summaries and raw arrays; the numbers are right. Four are framed more strongly than the data allow:

| Claim in v1 | Issue |
|---|---|
| Table 1 "test gain" of +0.00077 and +0.00011 with trial-level intervals | The two *no-communication baselines* differ from each other by +0.00088 [+0.00045, +0.00131]. The reported gain is seed-to-seed noise; pairing communication seed k with baseline seed k is arbitrary. |
| Table 1 train rewards to five decimals | Each is a single 1,024-episode training batch (sd about 0.010). |
| "Final active words 1, 1, 6" for the three-agent run | The six tokens form 430 distinct sequences in a positional code. They are not six concept words. |
| Addressee-colour scores 0.56–0.58 "weakly communicated" | The per-token metric misses it: the full stream decodes addressee colour at 0.99. |
| "The three-agent policy used its messages to coordinate actions" | Only removing the GO token matters. A lone GO command leaves the recipient 0.74 from its target, worse than the centroid's 0.61. There is no three-agent no-communication baseline to compare with. |
| Pushing sensitivity: disabling contact adds 0.914 | With contact off the recipient moves 0.26 away on its own, so the figure is not pure push. |

One latent bug: `scripts/evaluate.py:852-873` takes the last metrics row as "train reward" regardless of which training context it came from. No reported number is affected because every final row happens to be the main context.

### 2.3 Tests

v1's 13 tests check wiring (masking, stream order, gradient reaching the logits). They do not test the Euler step against the Appendix, that peers are invisible in verbal worlds, the GO TO and LOOK AT rewards, the sampler's distribution, the vocabulary reward's value, or the loss signs. Most importantly there is no learnability check, so the suite cannot tell "cannot learn" from "did not learn". The supervised probes above would have exposed cause 1 in two minutes of CPU time.

### 2.4 Process

v1 spent 23 full-size GPU runs before isolating why the simplest case failed. Its controls varied penalties, horizon and stream wiring, all downstream of the binding plateau. Two seeds for the central comparison were too few to see that the baseline itself varies by as much as the claimed effect.

## 3. What a faithful, working reproduction needs

Ordered roughly by how much each mattered in the rebuild; the ablations in `RESULTS.md` §7 quantify 3.1–3.3. "Paper" marks what the paper fixes; everything else is a reconstruction choice and is listed as such in `IMPLEMENTATION.md`.

### 3.1 Start in the nonlinear regime

He-normal hidden layers (or Glorot-uniform throughout). Under supervision this turns the binding step from a 2,500-update plateau into a 200-update problem; in full training it halves the time to emergence (about 1,100 updates against 2,000–2,500). It is consistent with the paper, which gives no initialiser and was almost certainly run with a Glorot default.

### 3.2 Use the literal auxiliary task as the bootstrap, and let it reach the listener's view

The paper's auxiliary task asks each listener to predict each speaker's goal vector. Read literally, the location part is in the speaker's private frame and cannot be grounded by a listener, which is why v1 learned a useless code. But it has a function: it is the only term that gives the *speaker* a first-order reason to make its utterances depend on its goal. Every target that can only be decoded through landmark identity (the location in the listener's frame, or even the landmark's colour as a privileged label) starts at a saddle: a channel that carries nothing is best ignored, and an ignored channel sends no gradient back. In v2 those targets did not start in 3,000 updates, and in the final ablations neither the listener-frame target, nor a target without the location, nor no auxiliary task produced any communication in 7,000.

With initialisation fixed, the literal target makes the channel informative within a few hundred updates, and physical reward then reshapes it into landmark reference:

| 1x1x3, no penalties, 3,000 updates | Terminal error | Auxiliary error |
|---|---:|---:|
| Literal target, default init (v1's situation without penalties) | −0.446 (centroid) | 0.015 |
| Literal target, He init, prediction from the stream only | −0.40, falling from update ~2,400 | 0.05 |
| Literal target, He init, prediction head also reads the listener's pooled view | −0.07, falling from update ~1,000 | 0.03 |
| Location in the listener's frame only | −0.445 | 0.446 (uninformed) |
| Landmark colour as a privileged label | −0.445 | 0.668 (uninformed) |
| No auxiliary task | −0.445 | — |

### 3.3 Let words exist before taxing them, and keep the auxiliary weight small

Start the vocabulary reward and utterance cost after communication carries reward, and exclude silence from the Dirichlet-process counts. The paper's Figure 6 shows its penalty acting from the first updates; that timing only works if words become useful faster than the penalty removes them, which did not hold in either implementation here. This is the largest remaining difference from the paper's training dynamics.

The auxiliary task is needed to start the channel and harmful once it has: at weight 1 the two-agent world reached terminal error −0.11 by update 2,400, at weight 0.1 it reached −0.016. The reported runs use 0.1.

The vocabulary penalty has no natural stopping point while training uses relaxed symbols. A single symbol's continuous activation can carry a graded message, so the penalty can shrink the vocabulary to one word with almost no loss of *training* reward (one development run: 1 active word, training terminal error −0.017). Exact symbols at test time cannot do that. The penalty weight therefore sets the vocabulary size, and the sizes reported are a function of that weight, not a property the agents converge to. The final ablation makes the cost explicit: training reward is the same at every weight from 0 to 0.2, while test reward with exact symbols falls from −0.013 at weight 0 to −0.19 at 0.1 and −0.14 at 0.2, and the share of addressees at the right landmark from 94% to 70% and 83% (`RESULTS.md` §7).

### 3.4 Keep reward terms commensurate

Score DO NOTHING as displacement from the starting position, in the same squared-distance units as GO TO and LOOK AT.

### 3.5 Measure language at the level of the stream, by what listeners do, and with baselines per world

Decode goal factors from the whole utterance stream on held-out episodes, not token by token. Train a no-communication baseline for every world and several seeds of each, and report the spread across seeds next to any gain.

Two measurements turned out to be necessary to read the result correctly:

- **Which landmark the addressee ends at.** This is the functional test of reference. In the reported runs it is 82% and 75% in the two-agent worlds, where the words alone identify the landmark's colour only 60% and 51% of the time.
- **Decoding given the landmark layout.** Private frames are rotations, so the handedness of the red-green-blue triangle and the order of its side lengths are common knowledge. Agents exploit this: they name one colour and pick out the other landmarks by where they sit relative to it, often using silence for one of them. Adding the layout to the decoder raises colour decoding from 60% to 84% (1x1x3), which matches listener behaviour. A per-word precision table alone would have reported "colour words" for a code that is mostly positional.

### 3.6 Test learnability, not only wiring

Keep the two supervised probes (`python -m emlang.probe speaker|listener`) as the first thing to run after any architectural change.

### 3.7 Give non-verbal behaviours something to be built from

With a lone "bring the other agent here" goal, agents solve pointing and guiding with arbitrary conventions (gaze points 3 units from the target; the guide stops 0.4 from it). Any scaled rotation of the true direction is an equally good code, and noise favours large scales. Literal behaviour appeared once the goal holder also had ordinary goals of its own in half of the episodes: go-to goals for guiding, look-at goals for pointing. Pushing needs a gradient before first contact; a softer contact force supplies it.

## 4. What remains open, and how to go further

Measured status per paper claim is in `RESULTS.md`, including a section on what was not reproduced. The directions most likely to close the remaining gaps:

1. **Colour words instead of positional reference.** The positional code is a consequence of rotation-only private frames with exactly three distinct landmarks. It appears without any vocabulary penalty, so the penalty is not its cause. Candidate remedies, in order of fidelity to the paper: train on worlds whose landmark sets vary (a two-landmark share is implemented as `--two-landmarks`; at 60% the code stays positional but becomes more accurate, 89% of addressees at the target against 82%, and transfers to two-landmark worlds); add private reflections so that handedness is no longer shared; draw landmark colours with repetition, or vary the number of landmarks from one to four, so that layout cannot stand in for colour.
2. **Vocabulary size without a test-time loss.** The vocabulary reward shrinks the vocabulary at no cost in training reward and a large cost with exact symbols. Train on exact symbols (straight-through, implemented as `--straight-through`) so that the penalty meets resistance from physical reward. In the final ablation this removed the gap but one of two seeds still collapsed to one word at weight 0.05, so it needs a smaller weight (0.02 kept 7–10 words and 93% at target even with relaxed training) and its own tuning. The best-performing recipe measured here is weight 0.02 or less; combining it with the two-landmark share is the obvious next run.
3. **Penalties from the start, as in the paper's Figure 6.** Needs faster emergence than the 1,100 updates measured here. The supervised probes show the binding can be learnt in about 200 updates, so the bottleneck is the joint bootstrap, not capacity. A curriculum from one landmark upward or a population of more than two speakers per listener would be the first things to try.
4. **Seed variance.** Five seeds per world are reported; the gain's standard deviation over seeds is 0.02–0.03, and which colour or position convention emerges differs by seed.
5. **Averaging under conflicting commands.** The policies here end at the midpoint in 31% of episodes and at one of the two targets in 43%. The paper's averaging would follow from a listener that pools the two streams before deciding; softmax pooling over streams should allow it, so this may be a matter of training on more than one instruction per addressee.
