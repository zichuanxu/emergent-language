"""Build Table 1, Figures 4-7 and the per-claim tables from evaluated runs.

    python -m emlang.report --runs runs/main --out report [--ablations runs/ablation]

Every measured number in the output is read from a run's eval.json / metrics.npz. The paper's
numbers are the constants PAPER_TABLE1 and PAPER_VOCAB below.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import ACTIVE_THRESHOLD

PAPER_TABLE1 = {"no_communication": (-0.919, -0.920), "communication": (-0.332, -0.392)}
PAPER_VOCAB = {"1x1x3": 3, "1x2x3": 6, "3x3x3": 9}
VERBAL = ("1x1x3", "1x2x3", "3x3x3")
NONVERBAL = ("pointing", "guiding", "pushing")
CASE_COLOR = {"1x1x3": "#d62728", "1x2x3": "#1f77b4", "3x3x3": "#9467bd"}
RGB = {0: "#d9534f", 1: "#5cb85c", 2: "#428bca"}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(p.parent for p in root.rglob("metrics.npz")):      # checkpoints are not needed
        if not (path / "config.json").exists():
            continue
        cfg = json.loads((path / "config.json").read_text())["train"]
        run = {"name": str(path.relative_to(root)), "path": path, "cfg": cfg, "case": cfg["case"], "seed": cfg["seed"],
               "comm": cfg["comm"] and cfg["env"]["comm"], "metrics": np.load(path / "metrics.npz")}
        if (path / "eval.json").exists():
            run["eval"] = json.loads((path / "eval.json").read_text())
        if (path / "eval_episodes.npz").exists():
            run["episodes"] = np.load(path / "eval_episodes.npz")
        runs.append(run)
    return runs


def pick(runs, case, comm=True):
    return [r for r in runs if r["case"] == case and r["comm"] == comm and "eval" in r]


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def fmt_ci(v, digits=3, sign=False):
    plus = "+" if sign else ""
    return "—" if v is None else f"{v[0]:{plus}.{digits}f} [{v[1]:{plus}.{digits}f}, {v[2]:{plus}.{digits}f}]"


def mean_sd(values, digits=3, sign=False) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    plus = "+" if sign else ""
    sd = f" ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else ""
    return f"{np.mean(values):{plus}.{digits}f}{sd}"


def baseline_of(run, bases):
    """The no-communication run a communicating run was evaluated against (None if there was none)."""
    if not bases or "no_communication" not in run["eval"]["reward"]:
        return None
    recorded = run["eval"].get("baseline")
    if recorded is None:                   # older eval.json: the pairing rule of scripts/evaluate_all.sh
        return bases[(run["seed"] - 1) % len(bases)]
    # the recorded path ends in <group>/<replica>, e.g. 1x1x3_nocomm/s2
    match = [b for b in bases if b["path"].parts[-2:] == Path(recorded).parts[-2:]]
    if len(match) != 1:
        print(f"warning: baseline {recorded} of {run['name']} is not among the evaluated runs")
        return None
    return match[0]


def active_curve(metrics, window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Active non-silence symbols per training iteration, main context only.

    `window` > 1 returns the running median over that many main-context updates, which removes the
    flicker of symbols that sit at the 0.5% threshold.
    """
    main = metrics["ctx"] == 0
    its = np.nonzero(main)[0] + 1
    active = (metrics["freq"][main][:, 1:] >= ACTIVE_THRESHOLD).sum(1)
    if window > 1 and len(active) >= window:
        padded = np.concatenate([np.full(window - 1, active[0]), active])
        active = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    return its, active


def final_active(metrics, window=200) -> float:
    """Median active-word count over the last `window` updates on the main world."""
    return float(np.median(active_curve(metrics)[1][-window:]))


# ---------------------------------------------------------------- figures

def figure6(runs, out: Path, reference: Path | None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if reference and reference.exists():
        paper = {}
        for row in csv.DictReader(reference.open()):
            paper.setdefault(row["case"], []).append((int(row["iteration_approx"]), int(row["active_vocabulary"])))
        for case, points in paper.items():
            x, y = zip(*points)
            ax.step(x, y, where="post", color=CASE_COLOR[case], lw=0.6, ls=":", alpha=0.7)
    for case in VERBAL:
        for i, run in enumerate(sorted(pick(runs, case), key=lambda r: r["seed"])):
            its, active = active_curve(run["metrics"], window=25)
            ax.step(its, active, where="post", color=CASE_COLOR[case], lw=0.9, alpha=0.9 if i == 0 else 0.45,
                    label=case if i == 0 else None)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("active vocabulary size")
    ax.set_ylim(0, 20)
    ax.set_yticks(range(0, 21, 5))
    ax.grid(alpha=0.3)
    ax.legend(title="solid: this reproduction, one line per seed\n(running median over 25 updates on the main world)\ndotted: paper, digitized", fontsize=8, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure6.png", dpi=170)
    plt.close(fig)


def stream_panel(ax, symbols, vocab, title, max_t=None):
    """symbols: B,T,N. Lines between consecutive symbols, opacity proportional to frequency."""
    b, t, n = symbols.shape
    t = min(t, max_t or t)
    streams = symbols.transpose(0, 2, 1).reshape(-1, symbols.shape[1])[:, :t].astype(np.int64)
    for step in range(t - 1):
        pair = streams[:, step] * vocab + streams[:, step + 1]
        counts = np.bincount(pair, minlength=vocab * vocab).reshape(vocab, vocab) / len(streams)
        for a, c in zip(*np.nonzero(counts > 2e-4)):
            share = counts[a, c]
            ax.plot([step, step + 1], [a, c], color="black", alpha=float(min(1.0, 0.05 + 3 * share)),
                    lw=0.4 + 2.5 * share)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylim(-0.5, vocab - 0.5)
    ax.set_yticks(range(0, vocab, 2))
    ax.set_xticks(range(t))


def figure5(run, out: Path):
    ep = run["episodes"]
    vocab = run["cfg"]["policy"]["vocab"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    stream_panel(axes[0], ep["symbols_before_training"], vocab, "Before training")
    stream_panel(axes[1], ep["symbols"], vocab, "After training")
    axes[0].set_ylabel("vocabulary symbol (0 = silence)")
    fig.tight_layout()
    fig.savefig(out / "figure5.png", dpi=170)
    plt.close(fig)


def _rgb(color_row):
    return RGB.get(int(np.argmax(color_row)), "#999999")


UNCLEAR = re.compile(r"S\d+$")          # label of a symbol without a clear meaning
COLOR_WORD = ("red", "green", "blue")
ACTION_WORD = ("go to", "look at", "stay", "")


def figure4(rows, out: Path, panels: int = 4):
    """rows: list of (run, episode index, caption). Panels are evenly spaced through the episode.

    Each panel shows the positions at that step, the path so far, and next to every agent the words
    it has uttered so far. Dotted lines in the first panel join each addressee to the location its
    instruction refers to, drawn in the colour of the agent that holds the instruction.
    """
    fig, axes = plt.subplots(len(rows), panels, figsize=(2.6 * panels, 2.75 * len(rows)), squeeze=False)
    for r, (run, e, caption) in enumerate(rows):
        ep, labels = run["episodes"], run["eval"]["symbols"]["labels"]
        pos, lm, sym = ep["sample_pos"][e], ep["sample_lm_pos"][e], ep["symbols"][e]
        steps = np.linspace(0, pos.shape[0] - 1, panels).round().astype(int)
        points = np.concatenate([pos.reshape(-1, 2), lm])
        centre, half = (points.max(0) + points.min(0)) / 2, float(np.ptp(points, axis=0).max()) / 2 + 0.3
        which = lambda row: COLOR_WORD[int(np.argmax(row))]
        goals = []
        for i in range(pos.shape[1]):
            goal_type, rcpt = int(ep["sample_goal_type"][e][i]), int(ep["sample_goal_rcpt"][e][i])
            if goal_type > 2:
                continue
            place = f" {which(ep['sample_lm_color'][e][ep['sample_goal_lm'][e][i]])}" if goal_type < 2 else ""
            goals.append(f"{which(ep['sample_agent_color'][e][i])} holds: {which(ep['sample_agent_color'][e][rcpt])} "
                         f"agent, {ACTION_WORD[goal_type]}{place}")
        for c, t in enumerate(steps):
            ax = axes[r][c]
            for j in range(lm.shape[0]):
                ax.scatter(*lm[j], s=30, marker="s", color=_rgb(ep["sample_lm_color"][e][j]), zorder=3)
            for i in range(pos.shape[1]):
                color = _rgb(ep["sample_agent_color"][e][i])
                ax.plot(pos[: t + 1, i, 0], pos[: t + 1, i, 1], color=color, lw=0.7, alpha=0.5)
                ax.scatter(*pos[t, i], s=170, color=color, alpha=0.35, zorder=2)
                goal_type, rcpt = ep["sample_goal_type"][e][i], ep["sample_goal_rcpt"][e][i]
                if c == 0 and goal_type < 2:
                    target = lm[ep["sample_goal_lm"][e][i]]
                    ax.plot([pos[0, rcpt, 0], target[0]], [pos[0, rcpt, 1], target[1]], ls=":", lw=0.8, color=color)
                said = [labels.get(str(int(k)), f"S{int(k)}") for k in sym[: max(t, 1), i] if k != 0]
                ax.annotate(" ".join(said) if said else "…", pos[t, i], xytext=(6, 7), textcoords="offset points",
                            fontsize=5.5, color="#222222")
            ax.set_xlim(centre[0] - half, centre[0] + half), ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(f"t={t}", fontsize=8)
            if c == 0:
                ax.set_ylabel(caption, fontsize=7)
                ax.set_title("\n".join(goals), fontsize=5.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "figure4.png", dpi=170)
    plt.close(fig)


def figure7(runs, out: Path):
    cases = [c for c in NONVERBAL if pick(runs, c, comm=False)]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.6), squeeze=False)
    for ax, case in zip(axes[0], cases):
        run = sorted(pick(runs, case, comm=False), key=lambda r: r["seed"])[0]
        ep = run["episodes"]
        gain = ep["paired_increase"][: len(ep["sample_pos"])]
        e = int(np.argsort(gain)[len(gain) // 2])  # episode with the median ablation effect among the samples
        pos, ablated, lm = ep["sample_pos"][e], ep["sample_pos_ablated"][e], ep["sample_lm_pos"][e]
        s = int(ep["sample_sender"][e])
        r = 1 - s
        for j in range(lm.shape[0]):
            ax.scatter(*lm[j], s=40, color=_rgb(ep["sample_lm_color"][e][j]), zorder=3)
        ax.scatter(*ep["sample_target"][e], s=220, facecolors="none", edgecolors="black", lw=0.8, zorder=2)
        ax.plot(pos[:, s, 0], pos[:, s, 1], color="#444444", lw=1.2, label="sender")
        ax.plot(pos[:, r, 0], pos[:, r, 1], color="#d9534f", lw=1.6, label="recipient")
        ax.plot(ablated[:, r, 0], ablated[:, r, 1], color="#f0ad4e", lw=1.2, ls="--",
                label=f"recipient, {run['eval']['ablation']}")
        for who, color in ((s, "#444444"), (r, "#d9534f")):
            ax.scatter(*pos[0, who], s=50, facecolors="none", edgecolors=color)
            ax.scatter(*pos[-1, who], s=50, color=color)
        if case == "pointing":
            gaze = ep["sample_gaze"][e]
            for t in range(1, pos.shape[0], 2):      # the gaze point at index t was set from the position at t - 1
                ax.plot([pos[t - 1, s, 0], gaze[t, s, 0]], [pos[t - 1, s, 1], gaze[t, s, 1]], color="#444444", lw=0.5, ls=":")
        ax.set_title(case, fontsize=10)
        points = np.concatenate([pos.reshape(-1, 2), ablated.reshape(-1, 2), lm])
        centre, half = (points.max(0) + points.min(0)) / 2, max(1.1, float(np.ptp(points, axis=0).max()) / 2 + 0.15)
        ax.set_xlim(centre[0] - half, centre[0] + half), ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure7.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- tables

def table1(runs) -> list[str]:
    (p_nc_train, p_nc_test), (p_c_train, p_c_test) = PAPER_TABLE1["no_communication"], PAPER_TABLE1["communication"]
    lines = ["| World | Seed | No comm. train | Comm. train | No comm. test | Comm. test | Test gain (paired) | "
             "Comm. test / no comm. test |", "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| Paper (world unspecified) | — | {p_nc_train:.3f} | {p_c_train:.3f} | {p_nc_test:.3f} | {p_c_test:.3f} | "
             f"{p_c_test - p_nc_test:+.3f} | {p_c_test / p_nc_test:.2f} |"]
    for case in VERBAL:
        bases = sorted(pick(runs, case, comm=False), key=lambda r: r["seed"])
        ratios, gains, used = [], [], set()
        for run in sorted(pick(runs, case), key=lambda r: r["seed"]):
            rew = run["eval"]["reward"]
            b = baseline_of(run, bases)
            comm_test = rew["communication"]["step_mean"][0]
            nocomm_test = rew.get("no_communication", {}).get("step_mean", [None])[0]
            gain = rew.get("paired_gain_step_mean")
            ratio = comm_test / nocomm_test if nocomm_test else None
            if ratio:
                ratios.append(ratio), gains.append(gain[0]), used.add(run["eval"].get("baseline", b and b["name"]))
            lines.append(
                f"| {case} | {run['seed']} | {fmt(b['eval']['train_reward_last_200_iters']) if b else '—'} | "
                f"{fmt(run['eval']['train_reward_last_200_iters'])} | {fmt(nocomm_test)} | {fmt(comm_test)} | "
                f"{fmt_ci(gain, sign=True)} | {fmt(ratio, 2)} |")
        if len(ratios) > 1:
            lines.append(f"| {case} | mean ± sd, {len(ratios)} comm. seeds against {len(used)} baselines | | | | | "
                         f"{mean_sd(gains, sign=True)} | {mean_sd(ratios, 2)} |")
    return lines


def reductions_table(runs) -> list[str]:
    lines = ["| World | Seed | Condition | Step mean | Terminal | Step mean, summed over goals | Terminal, summed over goals |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for case in VERBAL:
        for run in sorted(pick(runs, case), key=lambda r: r["seed"]):
            for cond in ("no_communication", "communication", "communication_relaxed_symbols", "messages_silenced"):
                v = run["eval"]["reward"].get(cond)
                if v:
                    lines.append(f"| {case} | {run['seed']} | {cond.replace('_', ' ')} | {fmt(v['step_mean'][0])} | "
                                 f"{fmt(v['terminal'][0])} | {fmt(v['step_mean_sum_over_goals'][0])} | "
                                 f"{fmt(v['terminal_sum_over_goals'][0])} |")
    return lines


def outcome_table(runs) -> list[str]:
    """End-of-episode outcome: terminal error and whether the addressee is at the right landmark."""
    lines = ["| World | Seed | Terminal reward: no comm. | comm. (exact symbols) | comm. (relaxed symbols, as in training) | "
             "messages silenced | Addressee ends nearest the target landmark: no comm. | comm. | silenced |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in VERBAL:
        rows, base_ids = [], []
        bases = sorted(pick(runs, case, comm=False), key=lambda r: r["seed"])
        for run in sorted(pick(runs, case), key=lambda r: r["seed"]):
            b = baseline_of(run, bases)
            base_ids.append(run["eval"].get("baseline", b and b["name"]) if "no_communication" in run["eval"]["reward"] else None)
            rew, lc = run["eval"]["reward"], run["eval"].get("landmark_choice", {})
            term = lambda k: rew.get(k, {}).get("terminal", [None])[0]
            choice = lambda k: lc.get(k, {}).get("addressee_ends_nearest_to_target_landmark", [None])[0]
            row = [term("no_communication"), term("communication"), term("communication_relaxed_symbols"),
                   term("messages_silenced"), choice("no_communication"), choice("communication"),
                   choice("messages_silenced")]
            rows.append(row)
            lines.append(f"| {case} | {run['seed']} | " + " | ".join(fmt(x) for x in row) + " |")
        if len(rows) > 1:
            # a baseline shared by several communicating seeds counts once in the no-communication columns
            first = [base_ids.index(i) for i in dict.fromkeys(base_ids) if i is not None]
            mean = [mean_sd([rows[i][c] for i in first]) if c in (0, 4) else mean_sd([row[c] for row in rows])
                    for c in range(len(rows[0]))]
            lines.append(f"| {case} | mean ± sd ({len(rows)} comm. seeds, {len(first)} baselines) | " + " | ".join(mean) + " |")
    return lines


def vocab_table(runs) -> list[str]:
    """Active words: during training (relaxed messages) and on test episodes (exact messages)."""
    lines = ["| World | Paper | Seed | Final active words in training (median over the last 200 updates on the main "
             "world) | Active words on test | Words above 1% of utterance slots on test | Words with a clear meaning "
             "(precision ≥ 0.8, at least 0.2 above the base rate) |", "|---|---:|---:|---:|---:|---:|---|"]
    for case in VERBAL:
        train_counts, test_counts, above = [], [], []
        for run in sorted(pick(runs, case), key=lambda r: r["seed"]):
            sym = run["eval"]["symbols"]
            clear = sorted(lab for lab in sym["labels"].values() if not UNCLEAR.match(lab))
            train_counts.append(final_active(run["metrics"])), test_counts.append(sym["n_active"])
            above.append(sum(share >= 0.01 for share in sym["symbol_slot_share"][1:]))
            lines.append(f"| {case} | {PAPER_VOCAB[case]} | {run['seed']} | {train_counts[-1]:g} | "
                         f"{sym['n_active']} | {above[-1]} | {', '.join(clear) or '—'} |")
        if len(train_counts) > 1:
            lines.append(f"| {case} | {PAPER_VOCAB[case]} | mean ± sd over {len(train_counts)} seeds | "
                         f"{mean_sd(train_counts, 1)} | {mean_sd(test_counts, 1)} | {mean_sd(above, 1)} | |")
    return lines


def symbol_tables(runs) -> list[str]:
    lines = []
    for case in VERBAL:
        for run in sorted(pick(runs, case), key=lambda r: r["seed"]):
            sym = run["eval"]["symbols"]
            lines += [f"### {case} · seed {run['seed']}", "",
                      "| Symbol | Meaning | Precision | Recall (directed goals) | Share of utterance slots | Mean first step |",
                      "|---:|---|---:|---:|---:|---:|"]
            for k, row in sorted(sym["table"].items(), key=lambda kv: (kv[1]["factor"], kv[1]["label"])):
                lines.append(f"| {k} | {sym['labels'][k]} ({row['label']}) | {row['precision']:.3f} | "
                             f"{row['recall_directed']:.3f} | {row['slot_share']:.3f} | {row['mean_first_step']:.2f} |")
            lines.append("")
            lines += ["| Goal factor decoded from the set of words in a stream (held-out) | Accuracy | Majority guess | "
                      "Reference |", "|---|---:|---:|---|"]
            rare = sym["decode"].get("landmark", {}).get("share_decoded_by_single_word_fallback")
            rare_layout = sym["decode"].get("landmark", {}).get("share_decoded_by_single_word_fallback_given_layout")
            for name, d in sym["decode"].items():
                ref = ""
                if "accuracy_given_landmark_layout" in d:
                    ref = (f"{d['accuracy_given_landmark_layout']:.3f} when the decoder also sees the landmark layout "
                           f"(layout alone: {d['accuracy_from_layout_alone']:.3f})")
                if "accuracy_from_landmark_colour" in d:
                    ref = f"a pure colour vocabulary would score {d['accuracy_from_landmark_colour']:.3f}"
                lines.append(f"| {name.replace('_', ' ')} | {d['accuracy']:.3f} | {d['majority_baseline']:.3f} | {ref} |")
            lines.append("")
            for name, o in sym["order"].items():
                if o["share"] is not None:
                    lines.append(f"- `{name}`: {o['share']:.3f} of {o['streams_with_both']:,} streams containing both.")
            g = sym.get("goto_before_landmark")
            if g and g["share"] is not None:
                lines.append(f"- GOTO word before the landmark word: {g['share']:.3f} of {g['streams_with_both']:,} streams.")
            if rare is not None:
                with_layout = f" ({rare_layout:.3f} when decoding within a layout class)" if rare_layout is not None else ""
                lines.append(f"- Word sets too rare to decode as a whole (decoded word by word instead): {rare:.3f} "
                             f"of the landmark streams{with_layout}.")
            u = sym["utterances_per_stream"]
            own = (f", {fmt_ci(u['self_goal'], 2)} when the speaker is the subject of its own goal"
                   if u["self_goal"] else "")
            lines.append(f"- Utterances per stream: {fmt_ci(u['directed_goal'], 2)} for goals addressed to another "
                         f"agent{own}.")
            share = ", ".join(f"t={t}: {s:.2f}" for t, s in enumerate(sym["non_silent_share_by_step"]))
            lines += [f"- Share of non-silent utterances by step: {share}.", ""]
            ctx = run["eval"].get("context_usage")
            if ctx:
                lines += ["| Test world | Streams using an action word | a landmark colour word | a landmark position word | "
                          "an addressee word | Utterances per stream |", "|---|---:|---:|---:|---:|---:|"]
                for name, c in ctx.items():
                    s = c["share_of_directed_streams_using_word_class"]
                    lines.append(f"| {name.replace('_', ' ')} | {fmt(s.get('action'))} | {fmt(s.get('landmark'))} | "
                                 f"{fmt(s.get('position'))} | {fmt(s.get('addressee'))} | "
                                 f"{c['utterances_per_directed_stream']:.2f} |")
                lines.append("")
    return lines


def generalization_table(runs) -> list[str]:
    lines = []
    for run in sorted(pick(runs, "3x3x3"), key=lambda r: r["seed"]):
        g = run["eval"].get("generalization")
        if not g:
            continue
        ea, cf, lm = g["extra_agent"], g["conflicting_goals"], g["extra_landmark"]
        shares = lambda c: " / ".join(f"{c[k]:.3f}" for k in ("share_at_first_target", "share_at_second_target",
                                                                "share_at_midpoint", "share_at_landmark_centroid"))
        lines += [f"### 3x3x3 · seed {run['seed']}", "", "| Probe | Measurement | Value [95% CI] |", "|---|---|---:|",
                  f"| Extra agent | progress toward the instructed target, clone has the addressee's colour | {fmt_ci(ea['clone_progress_to_target_same_color'])} |",
                  f"| | same clone recoloured to an unseen colour | {fmt_ci(ea['clone_progress_to_target_novel_color'])} |",
                  f"| | paired difference | {fmt_ci(ea['paired_difference'])} |",
                  f"| | clone's final distance to the target (addressee's: {ea['addressee_final_distance'][0]:.3f}) | {fmt_ci(ea['clone_final_distance_same_color'])} |",
                  f"| Conflicting goals | share ending at the first target / the second target / their midpoint / the centroid of all landmarks (each: within a quarter of the targets' separation; the regions overlap) | {shares(cf['both'])} |",
                  f"| | same shares, only the first command given | {shares(cf['only_first'])} |",
                  f"| | same shares, only the second command given | {shares(cf['only_second'])} |",
                  f"| | same shares, both commands given, all messages silenced | {shares(cf['both_messages_silenced'])} |",
                  f"| | final distance to the midpoint of the two targets | {fmt_ci(cf['both']['to_midpoint'])} |",
                  f"| | final distance to the first target, both commands given | {fmt_ci(cf['both']['to_first_target'])} |",
                  f"| | final distance to its target, first command only | {fmt_ci(cf['only_first']['to_first_target'])} |",
                  f"| Novel-colour distractor | share of addressees ending nearest the distractor | {lm['distractor']['share_ending_nearest_distractor']:.3f} |",
                  f"| | share ending nearest their target | {lm['distractor']['share_ending_nearest_target']:.3f} |",
                  f"| | change in final distance to the target vs no distractor | {fmt_ci(lm['distractor']['paired_change_in_target_distance'])} |",
                  f"| Second landmark in the target's colour | position along target→twin segment (0 = target, 0.5 = midpoint of the pair) | {fmt_ci(lm['same_color_twin']['projection_along_target_to_twin'])} |",
                  f"| | share ending between 30% and 70% | {lm['same_color_twin']['share_between_30_and_70_percent']:.3f} |",
                  f"| | final distance to the pair's midpoint | {fmt_ci(lm['same_color_twin']['final_distance_to_pair_midpoint'])} |",
                  f"| | same, if the extra landmark has an unseen colour | {fmt_ci(lm['same_color_twin']['midpoint_distance_if_twin_were_novel_color'])} |", ""]
    return lines


def nonverbal_table(runs) -> list[str]:
    lines = ["| Condition (cue removed in the ablation) | Seed | Recipient's distance to target: start | end | "
             "end with cue removed | Paired increase | Sender's gaze point to target, end | Sender's distance to target: "
             "start → end | Self-goal episodes: holder's miss of its own goal | Self-goal episodes: other agent's "
             "distance to the holder's goal location, start → end | Episodes with contact |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    stiff = []
    for case in NONVERBAL:
        for run in sorted(pick(runs, case, comm=False), key=lambda r: r["seed"]):
            e = run["eval"]
            by0 = e.get("bystander_initial_distance_to_senders_target_in_self_goal_episodes")
            by1 = e.get("bystander_final_distance_to_senders_target_in_self_goal_episodes")
            own = e.get("self_goal_final_distance")
            contact = e.get("share_of_episodes_with_contact")
            lines.append(
                f"| {case} ({e['ablation']}) | {run['seed']} | {e['recipient_initial_distance'][0]:.3f} | "
                f"{e['recipient_final_distance'][0]:.3f} | {e['recipient_final_distance_ablated'][0]:.3f} | "
                f"{fmt_ci(e['paired_increase_when_ablated'], sign=True)} | {e['sender_final_gaze_to_target_distance'][0]:.3f} | "
                f"{e['sender_initial_distance_to_target'][0]:.3f} → {e['sender_final_distance_to_target'][0]:.3f} | "
                f"{fmt(own[0]) if own else '—'} | {fmt(by0[0]) if by0 else '—'} → {fmt(by1[0]) if by1 else '—'} | "
                f"{fmt(contact) if contact is not None else 'no collisions'} |")
            d = e.get("default_contact_force")
            if d:
                stiff.append(
                    f"| {case} | {run['seed']} | {d['contact_margin_trained']:g} → {d['contact_margin_default']:g} | "
                    f"{e['recipient_initial_distance'][0]:.3f} | {d['recipient_final_distance'][0]:.3f} | "
                    f"{d['recipient_final_distance_ablated'][0]:.3f} | {fmt_ci(d['paired_increase_when_ablated'], sign=True)} | "
                    f"{d['share_of_episodes_with_contact']:.3f} |")
    if stiff:
        lines += ["", "The same policies evaluated with the default (stiffer) contact force instead of the one they "
                  "were trained with:", "",
                  "| Condition | Seed | Contact margin: trained → evaluated | Recipient's distance to target: start | end | "
                  "end with contact disabled | Paired increase | Episodes with contact |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"] + stiff
    return lines


def ablation_table(runs, reference=()) -> list[str]:
    """One row per ablation replica; `reference` are the unablated runs of the same world."""
    lines = ["| Run | World | Seed | Updates | Train reward | Test reward (step mean) | Test terminal reward | "
             "Change in test reward when messages are silenced | Addressee at the target landmark | Final active words | "
             "Landmark colour decoded from words | with layout |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [(f"reference: {r['name']}", r) for r in sorted(reference, key=lambda r: r["seed"])]
    rows += [(r["name"], r) for r in sorted(runs, key=lambda r: r["name"])]
    for name, run in rows:
        if "eval" not in run or not run["comm"]:
            continue
        e = run["eval"]
        dec = e["symbols"]["decode"].get("landmark", {})
        choice = e.get("landmark_choice", {}).get("communication", {}).get("addressee_ends_nearest_to_target_landmark", [None])[0]
        lines.append(f"| {name} | {run['case']} | {run['seed']} | {len(run['metrics']['reward']):,} | "
                     f"{e['train_reward_last_200_iters']:.3f} | "
                     f"{e['reward']['communication']['step_mean'][0]:.3f} | {e['reward']['communication']['terminal'][0]:.3f} | "
                     f"{fmt_ci(e['reward']['paired_loss_when_silenced_step_mean'], sign=True)} | {fmt(choice)} | "
                     f"{final_active(run['metrics']):g} | {fmt(dec.get('accuracy'))} | "
                     f"{fmt(dec.get('accuracy_given_landmark_layout'))} |")
    return lines


def choose_examples(runs) -> list:
    """First sampled test episodes that show a full exchange, one or two per verbal world."""
    rows = []
    for case, count in (("1x2x3", 1), ("3x3x3", 3)):
        candidates = sorted(pick(runs, case), key=lambda r: r["seed"])
        if not candidates:
            continue
        run = candidates[0]
        ep = run["episodes"]
        n = ep["sample_goal_rcpt"].shape[1]
        directed = (ep["sample_goal_rcpt"] != np.arange(n)).sum(1)
        spatial = (ep["sample_goal_type"] < 2).sum(1)
        order = np.argsort(-(directed * 10 + spatial), kind="stable")[:count]
        rows += [(run, int(e), f"{case} seed {run['seed']}\ntest episode {int(e)}") for e in order]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs/main"))
    p.add_argument("--ablations", type=Path)
    p.add_argument("--out", type=Path, default=Path("report"))
    p.add_argument("--reference", type=Path, default=Path("reference/figure6_vertices.csv"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.runs)
    figure6(runs, a.out, a.reference)
    lead = sorted(pick(runs, "3x3x3"), key=lambda r: r["seed"])
    if lead:
        figure5(lead[0], a.out)
    examples = choose_examples(runs)
    if examples:
        figure4(examples, a.out)
    figure7(runs, a.out)

    count = lambda key: " or ".join(f"{n:,}" for n in sorted({r["eval"][key] for r in runs if key in r.get("eval", {})})) or "—"
    trials, probes = count("trials"), count("probe_trials")
    md = ["# Measured results", "",
          "Generated by `python -m emlang.report`. Rewards are negative squared goal error averaged over goals and "
          f"time steps unless a column says otherwise. Test numbers use {trials} held-out worlds ({probes} for the "
          "probes of unseen configurations and of word omission) and exact categorical utterances. Brackets are 95% "
          "intervals over test episodes for one frozen checkpoint; \"mean ± sd\" is over training seeds.", "",
          "## Table 1 · physical reward with and without communication", ""] + table1(runs) + ["",
          "### Outcome at the end of the episode", ""] + outcome_table(runs) + ["",
          "### The same test episodes under other reward reductions", ""] + reductions_table(runs) + ["",
          "## Figure 6 · active vocabulary size", "", "![Figure 6](figure6.png)", ""] + vocab_table(runs) + ["",
          "## Figure 5 · symbol streams before and after training", "", "![Figure 5](figure5.png)", "",
          "## Figure 4 · example test episodes", "", "![Figure 4](figure4.png)", "",
          "## Word meanings, order and omission", ""] + symbol_tables(runs) + [
          "## Generalization to unseen configurations", ""] + generalization_table(runs) + [
          "## Figure 7 · non-verbal strategies", "", "![Figure 7](figure7.png)", ""] + nonverbal_table(runs) + [""]
    if a.ablations and a.ablations.exists():
        ablations = load_runs(a.ablations)
        cases = {r["case"] for r in ablations}
        reference = [r for r in runs if r["comm"] and "eval" in r and r["case"] in cases]
        md += ["## Ablations", ""] + ablation_table(ablations, reference) + [""]
    (a.out / "results.md").write_text("\n".join(md))
    print(f"wrote {a.out / 'results.md'}")


if __name__ == "__main__":
    main()
