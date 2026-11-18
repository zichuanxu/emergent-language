# Implementation notes

How this repository implements Mordatch & Abbeel, *Emergence of Grounded Compositional Language in Multi-Agent Populations* (arXiv:1703.04908v2): what the paper fixes, and every choice it leaves open. The implementation was written after reviewing an earlier attempt that failed; why it failed and what changed is in [`REVIEW.md`](REVIEW.md). Paths are relative to the repository root. Measured results are in [`RESULTS.md`](RESULTS.md).

## Layout

| Path | Contents |
|---|---|
| `emlang/world.py` | Batched differentiable particle world: Appendix dynamics, private frames, goals, reward, contact |
| `emlang/policy.py` | Figure 3 policy: shared per-entity and per-stream modules, softmax pooling, additive tanh memory, Gumbel-Softmax |
| `emlang/train.py` | Backpropagation through whole episodes; auxiliary goal prediction; Dirichlet-process vocabulary reward; several replicas per process |
| `emlang/evaluate.py` | Held-out evaluation: Table 1, word meanings, generalization probes, non-verbal probes |
| `emlang/report.py` | Table 1, Figures 4–7 and per-claim tables from evaluated runs |
| `emlang/probe.py` | Learnability probes: the speaker half (supervised) and the listener half (trained on reward) |
| `tests/` | 51 tests: dynamics, frames, information flow in whole rollouts, estimator, objective, ensemble independence, resume (CPU and accelerator), evaluation statistics on synthetic languages and scripted episodes |
| `analysis/` | Probe of the first implementation's policy; `named_colour_code.py`, `relaxed_vs_exact_words.py` and `timing_code.py`: the tables quoted in `docs/RESULTS.md` §2, §3 and §7 |
| `experiments/` | Job lists of the reported runs, one line per training process |
| `scripts/` | `evaluate_all.sh`; `cluster/`: launch, queue, archive and sync scripts for a GPU server with Docker |
| `runs/` | Per run: configuration, per-update metrics, evaluation, sampled test episodes (checkpoints are attached to the v1.0.0 release) |
| `report/` | Generated tables and Figures 4–7 |
| `reference/` | The paper's Figure 6 curves, digitized from the PDF |

## What the paper fixes, and how it is implemented

| Paper | Here |
|---|---|
| N agents and M landmarks in 2-D, discrete time; `p' = p + v·Δt`, `v' = γv + (u + f)·Δt`, gaze `= u_v`; Δt = 0.1, 1 − γ = 0.5 | `world.step`, tested against these equations |
| Agent i sees entity j at `R_i (p_j − p_i)` with a private random rotation | `world.observe`; observations are tested to be invariant to a global rigid motion and to other agents' frames |
| Observation = physical states of entities, all utterance streams, own memory, own goal | Entities and utterances enter through separate shared modules; the goal enters only the action module |
| "In the results that follow, agents do not observe other agents" | Verbal worlds expose only the agent itself and the landmarks |
| Goal = action (go to, look at, do nothing), a location, and the agent that should act; the location lies on a landmark, but the goal does not say which | 8 numbers: action one-hot, location in the holder's frame, addressee's colour. Landmark identity is never an input |
| No conflicting goals | Two agents instruct each other; three agents' addressees form a random permutation, so an agent may be the subject of its own goal |
| Vocabulary K = 20, silence allowed, utterances cost effort | Symbol 0 is silence; the cost is the probability of a non-silent symbol |
| Gumbel-Softmax with τ = 1 in training, categorical samples at test | `policy.gumbel_sample`; all test numbers use exact samples |
| Modules: 2 hidden layers of 256 ELU units, dropout 0.1, feature vectors 256, memory 32, softmax pooling, `m' = tanh(m + Δm + ε)` | `policy.Policy`; sizes asserted in a test |
| A fresh batch of 1,024 worlds per update, gradient through the whole episode | `train.rollout`; each replica gets its own 1,024 worlds |
| Auxiliary reward at the end of the episode for predicting other agents' goals from their utterance streams | See "auxiliary task" below |
| Vocabulary reward `r_c = Σ 1[c = c_k] log p(c_k)`, `p(c_k) = n_k / (α + n − 1)`, counts over agents, time and batch | `train.vocabulary_reward`, tested against the formula |

## Reconstruction choices

The paper leaves these open. Each one changes numbers; the first four decide whether language emerges at all. The values are those of the reported runs (`experiments/suite_main.txt`); the code defaults differ where noted.

| Choice | Value in the reported runs | Why |
|---|---|---|
| **Weight initialisation** | He-normal weights and zero biases in hidden layers; output layers keep PyTorch's default (`--init he`; the code default is PyTorch's everywhere) | PyTorch's default leaves the networks near-linear and the location-to-colour binding on a plateau of about 2,500 updates. See `docs/REVIEW.md` §2.1 |
| **Auxiliary task** | Each listener predicts each other speaker's goal vector as the speaker holds it (action, location in the speaker's frame, addressee colour) at the last step (`--aux-frame speaker`), weight 0.1. The prediction head is a separate network with one hidden layer that reads that stream's features *and the listener's pooled physical features*. It is never an input to the action module | The literal target is what makes utterances depend on goals early. Feeding the listener's view to the head (not in Figure 3) makes grounded use appear about 2.5× sooner; `--aux-ungrounded` removes it. Weight 0.1 instead of 1 lets physical reward shape the code sooner |
| **Penalty schedule** | Vocabulary reward (weight 0.05) and utterance cost (weight 0.02) are off until update 3,000 (1x1x3) or 4,000 (1x2x3, 3x3x3) and then ramp linearly over 1,000 updates | Applied from the first update they erase the channel before it is useful (in both implementations) |
| **Silence in the vocabulary reward** | Excluded from the counts | The paper counts uttered symbols. Including silence makes it either vanish or absorb everything |
| Vocabulary reward estimator | Counts are expected counts from the utterance probabilities over the batch, agents and time; the reward is averaged per utterance slot; gradient flows through both the indicator and `log p(c_k)` | With 1,024 worlds per update the reward from expected counts and from sampled counts differ by 0.001 to 0.01. With counts this large the concentration α has no measurable effect (below 0.001 for α = 10) |
| Reward | Negative squared error per goal, averaged over goals and over the steps of the episode: addressee's distance to the location (go to), gaze point's distance to it (look at), addressee's displacement from its starting position (do nothing) | Squared distance as in the authors' later particle environments. "Do nothing" is scored in the same units as the others; scoring it as squared speed would dwarf them because forces are unbounded |
| Horizon | 6 steps in verbal worlds (`--horizon 6`), 10 for pointing and guiding (the code default), 20 for pushing | Figure 5 shows time indices 0–5. With 6 steps the first scored position does not depend on the policy and the second cannot depend on any message, so the step-averaged reward has a floor that the end-of-episode reward does not; both are reported |
| Spawn area | Positions uniform in [−1, 1]², zero initial velocity, gaze at self | As in the authors' later particle environments |
| Entity features | Relative position, velocity, gaze point, RGB, is-agent flag, is-self flag (11 numbers) | The is-self flag lets an agent find its own colour among the entities |
| Colours | Agents and landmarks each get distinct colours from red, green, blue | |
| Movement scale and noise | Force = 10 × network output; Gaussian noise of 0.1 is added to force and gaze after that scaling, 0.05 to memory. The noise stays on at test time | The noise is small next to typical forces; removing it at test time changed the step-averaged reward by 0.006 to 0.010 in three development runs |
| Softmax pooling | Per feature, a softmax over entities (or streams) weights that feature's values | One reading of "softmax pooling" |
| Dropout | One dropout layer per module, between the two hidden layers | |
| Optimiser | Adam, learning rate 1e-3, gradient norm clipped at 5 per replica | |
| Mixed training worlds | For 1x2x3 and 3x3x3 (`--mixed`), half of the updates use the full world and half a simpler one (one landmark, fewer action types, or two agents) | The paper trains "on a variety of tasks and environment configurations simultaneously" |
| Seeds | Several seeds train in one process as independent replicas. A replica's seed fixes its initial weights; its training worlds and noise come from the process's random stream, and replicas of one process share the order of mixed training worlds | Replicas never share worlds, parameters or gradients (tested). A run is reproduced by rerunning its whole line of the job list |
| Active-word count | Non-silence symbols that fill at least 0.5% of utterance slots in a batch of exact samples | The paper gives no threshold |
| Word meanings | A word is given a meaning if, among streams about a goal for another agent, at least 80% of the streams containing it share one value of one goal factor and that is at least 0.2 above the value's base rate. Candidate factors: action, landmark colour, addressee colour, and the landmark's position among the landmarks (rank by distance from their centroid; nearer or farther of the other two from a coloured landmark). A word of which at least 80% of the streams are about the speaker's own goal is labelled OWN-GOAL | The paper names words by inspection. Position candidates are included because agents turned out to refer to landmarks that way |
| Decoding goals from words | Fitted on even test worlds, scored on odd ones. A set of words seen at least 5 times is decoded by its majority value, rarer sets by a naive-Bayes vote of their words. "With layout" fits one decoder per layout class (handedness and side-length order of the red-green-blue triangle, 12 classes), which every agent can compute in its own frame | Measures what a listener could recover; the layout version detects reference by position. It fits twelve decoders on the same streams, so it needs the default 10,000 test worlds: with 1,000 it reads 0.01 to 0.05 low |
| Non-verbal worlds | Two agents, no utterances, one goal per episode. In half of the episodes the goal is for its holder, in the other half it is "the other agent goes to this location" and the other agent has no goal. Guiding: others' position and velocity visible, the holder's own goals are go-to goals. Pointing: gaze visible too, and the holder's own goals are look-at goals. Pushing: the recipient sees no other agent, bodies collide | The paper does not describe these worlds beyond what each agent can observe. Self-goals make "look at" or "go to this location" an existing skill that the holder can reuse; the consequence is discussed in `docs/RESULTS.md` |
| Contact | Only in the pushing world. Smooth repulsion between bodies of radius 0.1: force 100 × softplus-penetration with margin 0.02 by default. The reported pushing run trains with margin 0.1 (`--contact-margin 0.1`) and is evaluated with both | Functional form of the authors' later particle environments. With the default margin the sender gets no gradient towards the recipient until it already touches it, and pushing was not learned |

Options that exist for experiments and are off in the main runs: `--two-landmarks p` (share of updates with only two of the three landmarks), `--straight-through` (train on one-hot utterances with the relaxed gradient), `--aux-ungrounded` (auxiliary head reads the utterance stream only), `--aux-frame listener|both|color|none`, `--goal-gain`, `--logit-scale`, `--close-start` and `--close-max` (pushing: start some training episodes near contact; never used in evaluation), the `pointing_swap` world. `experiments/suite_ablations.txt` uses the first four.

## Running

Python 3.10 or later and PyTorch 2.x (developed on Python 3.12). From the repository root, after `pip install -e .[dev]`:

```sh
python -m pytest tests -q                       # 51 tests, under a minute on CPU
python -m emlang.probe speaker --init he        # learnability probes
python -m emlang.train --case 1x1x3 --run-dir runs/demo --seed 1,2 --names s1,s2 --init he --horizon 6 \
    --aux-frame speaker --w-goal 0.1 --w-vocab 0.05 --w-utter 0.02 --penalty-start 3000 --penalty-ramp 1000 --iters 7000
python -m emlang.evaluate runs/demo/s1 --baseline runs/main/1x1x3_nocomm/s1
bash scripts/evaluate_all.sh main && bash scripts/evaluate_all.sh ablations   # every run of the reported suites
python -m emlang.report --runs runs/main --out report --ablations runs/ablations
```

The exact arguments of every reported run are in `experiments/suite_main.txt`, `experiments/suite_nonverbal.txt` and `experiments/suite_ablations.txt`, and in each run's `config.json`. In those lists `--run-dir /proj/results --device cuda` are the paths and device inside the training container; replace them to run a line directly.

The reported runs were trained on a shared GPU server with Docker. `scripts/cluster/sync_up.sh` copies the source, `job.sh` starts one labelled, resource-limited, stoppable container (CPU-only unless `GPU=` is given), `queue.sh` runs a job list with bounded parallelism, `reap.sh` archives logs and removes exited containers, and `sync_down.sh` fetches runs. They are configured through environment variables documented at the top of `job.sh`. Training checkpoints every 250 updates, exits with code 75 on SIGTERM and resumes when the same command is run again (`--iters` may be raised on resume).
