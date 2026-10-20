"""
Publication-quality figures for the final prediction and model validation.

Reads the artifacts written by train_model.py, backtest_knockouts.py and
predict_final.py and saves
PNGs to results/figures/:

  1. score_heatmap.png     90-minute scoreline probabilities (knockout-adjusted)
  2. outcome_by_method.png raw 90-minute W/D/L per method + title odds after
                           the knockout adjustment
  3. calibration.png       reliability diagram of the deployed combiner
                           (competitive holdout, >= 2019)
  4. logloss_ci.png        holdout log-loss per candidate with 95% bootstrap
                           CIs, and paired difference vs. the deployed model
  5. rating_history.png    Elo and Glicko-2 (+-2 RD band) of both finalists
  6. knockout_methods.png  knockout backtest: who-advances log-loss per title
                           method, paired difference vs. the old ensemble
  7. knockout_submodels.png knockout backtest: P(level after 90') and
                           P(penalties | extra time), observed vs. predicted
                           before and after the knockout adjustment

Colors: Spain = orange, Argentina = blue, draw = neutral gray (the pair
passes the CVD / contrast checks); every series is also direct-labeled.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from combine import CLASSES, EPS  # noqa: E402
from data_io import RESULTS_DIR  # noqa: E402
from knockout import recalibrate_draw, title_prob  # noqa: E402
from predict_final import FINAL  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
HOME, AWAY = FINAL["home"], FINAL["away"]
N_BOOT = 2000
BOOT_RNG = np.random.default_rng(7)
HISTORY_START = "2018-01-01"

# Palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DRAW = "#c3c2b7"
TEAM_COLOR = {HOME: "#eb6834", AWAY: "#2a78d6"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

METHOD_LABELS = {
    "logistic": "Logistic",
    "catboost": "CatBoost",
    "poisson_wdl": "Poisson W/D/L",
    "fixed_blend_lr_cb": "Fixed 0.65/0.35 blend",
    "equal_blend_3": "Average of 3 bases",
    "stacking_v2": "Stacking v2",
    "stack_blend_avg": "Stacking + blend avg",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "legend.frameon": False,
    })


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> results/figures/{name}")


def _header(fig, title: str, subtitle: str, extra_pt: float = 0.0):
    """Left-aligned title + subtitle stacked above the top of the axes grid,
    offset in points so spacing does not depend on the figure height."""
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in fig.axes)
    left = min(ax.get_tightbbox().x0 for ax in fig.axes) / fig.bbox.width
    base = fig.transFigure
    fig.text(left, top, subtitle, ha="left", va="bottom", color=INK_2, fontsize=9.5,
             transform=offset_copy(base, fig=fig, y=26 + extra_pt, units="points"))
    fig.text(left, top, title, ha="left", va="bottom", color=INK, fontsize=12.5,
             fontweight="bold",
             transform=offset_copy(base, fig=fig, y=30 + 13 * (subtitle.count("\n") + 1) + extra_pt,
                                   units="points"))


def _dodge(values, min_gap):
    """Spread label positions so neighbours are at least `min_gap` apart."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            mid = (out[a] + out[b]) / 2
            out[a], out[b] = mid - min_gap / 2, mid + min_gap / 2
    return out


# --------------------------------------------------------------------------- 1
def plot_score_heatmap(pred: dict):
    M = np.array(pred["score_matrix_90min"]["prob"])[:6, :6]  # 0..5 goals
    n = M.shape[0]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=M.max())
    # 2px surface gaps between cells
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=SURFACE, lw=2)
        ax.axvline(k - 0.5, color=SURFACE, lw=2)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.1%}" if v >= 0.0005 else "<0.1%", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 0.55 * M.max() else INK)
    # outline the draw diagonal (the scorelines that send the final to extra time)
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.46, k - 0.46), 0.92, 0.92, fill=False,
                                   ec=INK, lw=1.2, ls=(0, (2, 1.5))))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel(f"{AWAY} goals")
    ax.set_ylabel(f"{HOME} goals")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(-0.45, n - 0.55, f"{HOME} wins", ha="left", va="top", fontsize=9,
            color=INK, fontweight="bold")
    ax.text(n - 0.55, -0.45, f"{AWAY} wins", ha="right", va="bottom", fontsize=9,
            color=INK, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK_2, length=2)
    xg = pred["poisson_expected_goals"]
    _header(fig, f"Most likely 90-minute scorelines — {HOME} vs {AWAY}",
            f"Poisson scoreline shape, rescaled to the knockout-adjusted W/D/L · expected goals "
            f"{HOME} {xg[HOME]:.2f} – {xg[AWAY]:.2f} {AWAY}\n"
            f"Dashed cells = draw after 90' → extra time ({pred['p_extra_time']:.1%})")
    _save(fig, "score_heatmap.png")


# --------------------------------------------------------------------------- 2
def plot_outcome_by_method(pred: dict, deployed: str):
    breakdown = pred["method_breakdown_90min"]
    ka = pred["knockout_adjustment"]
    order = ["logistic", "catboost", "poisson_wdl", "fixed_blend_lr_cb",
             "equal_blend_3", "stack_blend_avg", "stacking_v2"]
    order = [m for m in order if m in breakdown]

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(order) + 1.6))
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    h = 0.62
    for y, name in enumerate(order):
        p = breakdown[name]
        segs = [(p["home_win"], TEAM_COLOR[HOME], f"{HOME} win"),
                (p["draw"], DRAW, "Draw"),
                (p["away_win"], TEAM_COLOR[AWAY], f"{AWAY} win")]
        left = 0.0
        for val, color, label in segs:
            ax.barh(y, val, left=left, height=h, color=color, edgecolor=SURFACE, linewidth=2,
                    label=label if y == 0 else None)
            ax.text(left + val / 2, y, f"{val:.1%}", ha="center", va="center", fontsize=9,
                    color=INK if color == DRAW else "white", fontweight="bold")
            left += val
        adj = recalibrate_draw(np.array([[p["away_win"], p["draw"], p["home_win"]]]),
                               ka["draw_odds_factor"])
        title_home = float(title_prob(adj, ka["p_home_given_level_after_90"])[0])
        ax.text(1.03, y, f"{title_home:.1%}", ha="left", va="center", fontsize=9.5,
                color=INK_2)
        ax.text(1.13, y, f"{1 - title_home:.1%}", ha="left", va="center", fontsize=9.5,
                color=INK_2)
    ax.text(1.03, len(order) - 0.35, f"Title\n{HOME}", ha="left", va="bottom", fontsize=8.5,
            color=MUTED)
    ax.text(1.13, len(order) - 0.35, f"Title\n{AWAY}", ha="left", va="bottom", fontsize=8.5,
            color=MUTED)
    ax.set_yticks(range(len(order)))
    labels = [METHOD_LABELS.get(m, m) + ("  (deployed)" if m == deployed else "") for m in order]
    ax.set_yticklabels(labels)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == deployed:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.22)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, handlelength=1.2)
    final = pred["final_title_probability"]
    _header(fig, "90-minute outcome by method",
            f"Bars: raw 90' forecast. Title columns: draw odds ×{ka['draw_odds_factor']:.2f} (knockout "
            f"recalibration), then {HOME} wins a level tie {ka['p_home_given_level_after_90']:.1%} "
            f"(extra time + shootout).\nPublished (Stacking v2): {HOME} {final[HOME]:.1%} · "
            f"{AWAY} {final[AWAY]:.1%}",
            extra_pt=22)
    _save(fig, "outcome_by_method.png")


# --------------------------------------------------------------------------- 3
def _wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def plot_calibration(ho: pd.DataFrame, deployed: str, n_bins: int = 10):
    comp = ho[ho["competitive"]]
    titles = {"H": "Home win", "D": "Draw", "A": "Away win"}
    color = {"H": TEAM_COLOR[HOME], "D": INK_2, "A": TEAM_COLOR[AWAY]}

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08, "wspace": 0.18})
    for col, c in enumerate(["H", "D", "A"]):
        p = comp[f"{deployed}_{c}"].to_numpy()
        y = (comp["outcome"] == c).to_numpy().astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        mean_p, freq, lo, hi, cnt = [], [], [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            k, n = y[m].sum(), m.sum()
            mean_p.append(p[m].mean())
            freq.append(k / n)
            l, u = _wilson(k, n)
            lo.append(l)
            hi.append(u)
            cnt.append(n)
        mean_p, freq = np.array(mean_p), np.array(freq)
        ece = np.sum(np.array(cnt) * np.abs(mean_p - freq)) / np.sum(cnt)

        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", zorder=1)
        ax.vlines(mean_p, lo, hi, color=color[c], lw=1.5, alpha=0.55, zorder=2)
        ax.plot(mean_p, freq, color=color[c], lw=2, zorder=3)
        ax.scatter(mean_p, freq, s=40, color=color[c], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.set_title(titles[c], fontsize=11)
        ax.text(0.04, 0.94, f"ECE {ece:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                color=INK_2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        if col == 0:
            ax.set_ylabel("Observed frequency")

        hax = axes[1, col]
        hax.hist(p, bins=np.linspace(0, 1, 41), color=color[c], alpha=0.75,
                 edgecolor=SURFACE, linewidth=0.8)
        hax.set_yticks([])
        hax.grid(False)
        hax.spines["left"].set_visible(False)
        hax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        hax.set_xlabel("Predicted probability")
    _header(fig, f"Calibration of {METHOD_LABELS.get(deployed, deployed)} — competitive holdout",
            f"Matches from 2019 on, never seen in training (n = {len(comp):,}). Points = {n_bins} "
            f"equal-count bins with 95% Wilson intervals; dashed = perfect calibration.\n"
            f"Bottom: distribution of predicted probabilities. ECE = expected calibration error.",
            extra_pt=22)
    _save(fig, "calibration.png")


# --------------------------------------------------------------------------- 4
def _row_log_loss(ho: pd.DataFrame, name: str) -> np.ndarray:
    P = np.clip(ho[[f"{name}_{c}" for c in CLASSES]].to_numpy(), EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = ho["outcome"].map({c: j for j, c in enumerate(CLASSES)}).to_numpy()
    return -np.log(P[np.arange(len(y)), y])


def plot_logloss_ci(ho: pd.DataFrame, deployed: str):
    comp = ho[ho["competitive"]].reset_index(drop=True)
    names = [m for m in METHOD_LABELS if f"{m}_H" in comp.columns]
    losses = {m: _row_log_loss(comp, m) for m in names}
    boot_idx = BOOT_RNG.integers(0, len(comp), size=(N_BOOT, len(comp)))

    rows = []
    for m in names:
        bl = losses[m][boot_idx].mean(axis=1)
        diff = (losses[m] - losses[deployed])[boot_idx].mean(axis=1)
        rows.append({
            "name": m, "ll": losses[m].mean(),
            "lo": np.percentile(bl, 2.5), "hi": np.percentile(bl, 97.5),
            "d": (losses[m] - losses[deployed]).mean(),
            "dlo": np.percentile(diff, 2.5), "dhi": np.percentile(diff, 97.5),
        })
    df = pd.DataFrame(rows).sort_values("ll", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.5 * len(df) + 1.8), sharey=True,
                                   gridspec_kw={"wspace": 0.08})
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ys = np.arange(len(df))
    for y, r in df.iterrows():
        is_dep = r["name"] == deployed
        c = INK if is_dep else TEAM_COLOR[AWAY]
        ax1.hlines(y, r["lo"], r["hi"], color=c, lw=2, alpha=0.5)
        ax1.scatter(r["ll"], y, s=60 if is_dep else 44, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax1.text(r["hi"] + 0.0007, y, f"{r['ll']:.4f}", va="center", fontsize=8.5,
                 color=INK if is_dep else INK_2, fontweight="bold" if is_dep else "normal")
        if not is_dep:
            ax2.hlines(y, r["dlo"], r["dhi"], color=c, lw=2, alpha=0.5)
            ax2.scatter(r["d"], y, s=44, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax2.text(r["dhi"] + 0.0004, y, f"+{r['d']:.4f}" if r["d"] >= 0 else f"{r['d']:.4f}",
                     va="center", fontsize=8.5, color=INK_2)
        else:
            ax2.text(0.0004, y, "reference", va="center", fontsize=8.5, color=MUTED)
    ax1.set_yticks(ys)
    labels = [METHOD_LABELS[m] + ("  (deployed)" if m == deployed else "") for m in df["name"]]
    ax1.set_yticklabels(labels)
    for t, m in zip(ax1.get_yticklabels(), df["name"]):
        if m == deployed:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax1.set_xlabel("Log-loss (lower is better)")
    ax1.set_xlim(df["lo"].min() - 0.004, df["hi"].max() + 0.008)
    ax1.set_title("Holdout log-loss, 95% CI", fontsize=11)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel(f"Paired difference vs. {METHOD_LABELS[deployed]}  (> 0 = worse)")
    ax2.set_title("Paired bootstrap difference, 95% CI", fontsize=11)
    ax2.set_xlim(min(df["dlo"].min(), 0) - 0.002, df["dhi"].max() + 0.008)
    _header(fig, "Is the deployed combiner really better?",
            f"Competitive matches since 2019 (n = {len(comp):,}), {N_BOOT:,} match-level bootstrap "
            f"resamples. Overlapping absolute CIs can hide a consistent paired gap —\n"
            f"the right panel is the relevant test. Caveat: the deployed model was chosen on this "
            f"same holdout, so its edge is mildly optimistic.", extra_pt=30)
    _save(fig, "logloss_ci.png")


# --------------------------------------------------------------------------- 5
def plot_rating_history(hist: pd.DataFrame, pred: dict):
    # append the post-semifinal ratings on the eve of the final
    eve = pd.DataFrame([{"date": pd.Timestamp(FINAL["date"]), "team": t, "elo": pred["elo"][t],
                         "glicko": pred["glicko2"][t]["rating"], "glicko_rd": pred["glicko2"][t]["rd"]}
                        for t in (HOME, AWAY)])
    hist = pd.concat([hist[hist["date"] >= HISTORY_START], eve], ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                                   gridspec_kw={"hspace": 0.28})
    for team in (HOME, AWAY):
        h = hist[hist["team"] == team]
        c = TEAM_COLOR[team]
        ax1.step(h["date"], h["elo"], where="post", color=c, lw=2, label=team)
        ax2.fill_between(h["date"], h["glicko"] - 2 * h["glicko_rd"], h["glicko"] + 2 * h["glicko_rd"],
                         step="post", color=c, alpha=0.13, lw=0)
        ax2.step(h["date"], h["glicko"], where="post", color=c, lw=2, label=team)
    last = hist.sort_values("date").groupby("team").last().loc[[HOME, AWAY]]
    for ax, col, fmt in ((ax1, "elo", "{team} {v:.0f}"), (ax2, "glicko", "{team} {v:.0f} ±{rd:.0f}")):
        lo, hi = ax.get_ylim()
        ys = _dodge(last[col].to_numpy(), 0.06 * (hi - lo))
        for (team, r), y in zip(last.iterrows(), ys):
            ax.annotate(fmt.format(team=team, v=r[col], rd=2 * r["glicko_rd"]),
                        (r["date"], r[col]), xytext=(r["date"] + pd.Timedelta(days=25), y),
                        textcoords="data", va="center", fontsize=9, color=TEAM_COLOR[team],
                        fontweight="bold", annotation_clip=False)
    ax1.set_title("Elo (World Football Elo rules)", fontsize=11)
    ax2.set_title("Glicko-2 rating, shaded band = ±2 RD (uncertainty)", fontsize=11)
    ax1.set_ylabel("Elo")
    ax2.set_ylabel("Glicko-2")
    ax1.legend(loc="upper left", ncol=2, fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.01)
    fig.subplots_adjust(right=0.86)
    _header(fig, f"Strength ratings of the finalists since {HISTORY_START[:4]}",
            "Pre-match ratings at each international match, last point = eve of the final. "
            "Elo and Glicko-2 disagree on who is stronger.")
    _save(fig, "rating_history.png")


KO_METHOD_LABELS = {
    "calibrated": "Knockout-adjusted stacking",
    "hybrid": "Stacking + naive extra time",
    "classifier": "Stacking + shootout tilt",
    "elo_only": "Elo only",
    "ensemble": "Old ensemble (MC + stacking)",
    "monte_carlo": "Monte Carlo",
}


# --------------------------------------------------------------------------- 6
def plot_knockout_methods(bt: dict):
    block = bt["advance_all_knockouts"]
    rows = sorted(block["methods"].items(), key=lambda kv: kv[1]["diff_vs_ensemble"],
                  reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(rows) + 1.4))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK, lw=0.8)
    for y, (m, r) in enumerate(rows):
        dep = m == "calibrated"
        c = INK if dep else MUTED
        if m == "ensemble":
            ax.text(0.0006, y, f"reference · log-loss {r['log_loss']:.4f}", va="center",
                    fontsize=8.5, color=MUTED)
            continue
        lo, hi = r["diff_ci95"]
        ax.hlines(y, lo, hi, color=c, lw=2, alpha=0.6)
        ax.scatter(r["diff_vs_ensemble"], y, s=60 if dep else 44, color=c, edgecolor=SURFACE,
                   lw=1.5, zorder=3)
        ax.text(hi + 0.0008, y, f"{r['diff_vs_ensemble']:+.4f}  (log-loss {r['log_loss']:.4f})",
                va="center", fontsize=8.5, color=INK if dep else INK_2,
                fontweight="bold" if dep else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([KO_METHOD_LABELS.get(m, m) + ("  (deployed)" if m == "calibrated" else "")
                        for m, _ in rows])
    for t, (m, _) in zip(ax.get_yticklabels(), rows):
        if m == "calibrated":
            t.set_fontweight("bold")
            t.set_color(INK)
    lo = min(r["diff_ci95"][0] for _, r in rows)
    hi = max(r["diff_ci95"][1] for _, r in rows)
    ax.set_xlim(lo - 0.002, hi + 0.016)
    ax.set_xlabel("Paired log-loss difference vs. the old ensemble  (< 0 = better)")
    _header(fig, "Knockout backtest: which title method predicts who advances?",
            f"{block['n']} World Cup / Euro / AFCON / Asian Cup knockouts 2004–2026, every model refit "
            f"before each edition. 95% paired bootstrap CIs.\nDropping the Monte Carlo is what beats "
            f"the old ensemble; the remaining methods are within noise of each other.")
    _save(fig, "knockout_methods.png")


# --------------------------------------------------------------------------- 7
def plot_knockout_submodels(bt: dict):
    panels = [
        ("level_after_90", "Level after 90'", "p_level90_stacking", "p_level90_calibrated",
         "Before (raw stacking)", "After (draw odds ×{lam:.2f})"),
        ("penalties_given_extra_time", "Penalties, given extra time", "p_pens_given_et",
         "p_pens_given_et_calibrated", "Before (ET factor 1/3)",
         "After (ET factor {phi:.3f})"),
    ]
    cal = bt["deployed_calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), gridspec_kw={"wspace": 0.45})
    for ax, (key, title, before, after, lb, la) in zip(axes, panels):
        block = bt[key]
        n, obs = block["n"], block["observed_rate"]
        lo, hi = _wilson(obs * n, n)
        rows = [("Observed", obs, INK_2), (lb, block["methods"][before]["mean_pred"], MUTED),
                (la.format(lam=cal["draw_odds_factor"], phi=cal["et_goal_factor"]),
                 block["methods"][after]["mean_pred"], INK)]
        ax.axvspan(lo, hi, color=DRAW, alpha=0.35, lw=0)
        ax.axvline(obs, color=INK_2, lw=0.8, ls="--")
        for y, (label, v, c) in enumerate(reversed(rows)):
            ax.scatter(v, y, s=60, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.annotate(f"{v:.1%}", (v, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK,
                        fontweight="bold" if c == INK else "normal",
                        bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)])
        ax.set_ylim(-0.5, len(rows) - 0.1)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlim(min(r[1] for r in rows + [("", lo, "")]) - 0.04,
                    max(r[1] for r in rows + [("", hi, "")]) + 0.04)
        ax.set_title(f"{title}  (n = {n})", fontsize=11)
    _header(fig, "Knockout backtest: draws and extra time, observed vs. predicted",
            "Mean predicted rate before and after the knockout adjustment; shaded band = 95% "
            "Wilson interval of the observed rate. Level after 90': knockouts 2004+;\n"
            "penalties: extra times 1986+. Parameters re-estimated before each edition from "
            "earlier knockouts only.", extra_pt=6)
    _save(fig, "knockout_submodels.png")


def main():
    _style()
    with open(RESULTS_DIR / "final_prediction.json") as f:
        pred = json.load(f)[f"{HOME} vs {AWAY}"]
    with open(RESULTS_DIR / "classifier_metrics.json") as f:
        deployed = json.load(f)["deployed"]
    ho = pd.read_csv(RESULTS_DIR / "holdout_predictions.csv", parse_dates=["date"])
    hist = pd.read_csv(RESULTS_DIR / "rating_history.csv", parse_dates=["date"])

    plot_score_heatmap(pred)
    plot_outcome_by_method(pred, deployed)
    plot_calibration(ho, deployed)
    plot_logloss_ci(ho, deployed)
    plot_rating_history(hist, pred)
    with open(RESULTS_DIR / "knockout_backtest.json") as f:
        bt = json.load(f)
    plot_knockout_methods(bt)
    plot_knockout_submodels(bt)


if __name__ == "__main__":
    main()
