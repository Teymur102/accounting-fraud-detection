"""Reproduce every result, table and figure in this repo.

    python -m frauddetect.run

Writes results/*.csv|json|md and figures/*.png.
"""
from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import beneish, models
from .data import RATIO14, RAW28, ROOT, load
from .metrics import evaluate
from .splits import out_of_time_split

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TEST_YEARS = list(range(2003, 2015))
WINDOWS = {"2003-2005": (2003, 2005), "2003-2008": (2003, 2008), "2003-2014": (2003, 2014)}
HEADLINE = "2003-2008"
TOPS = [0.01, 0.02, 0.03, 0.04, 0.05]
RUS_SEEDS = [0, 1, 2, 3, 4]

MODEL_LABELS = {
    "beneish_mscore": "Beneish M-score (fixed 1999 weights)",
    "logit_ratios14": "Logit, 14 ratios (Dechow-style)",
    "logit_raw28": "Logit, 28 raw items",
    "rusboost_raw28": "RUSBoost, 28 raw items",
    "lightgbm_raw28": "LightGBM, 28 raw items",
    "mlp_raw28": "Small neural net, 28 raw items",
}

# Quoted, not computed: Bao, Ke, Li, Yu & Zhang (2021), "A Response to 'Critique of
# an Article on Machine Learning in the Detection of Accounting Fraud'", Econ
# Journal Watch 18(1): 71-78, Table 1. These use the corrected serial-fraud
# handling (by p_aaer) that matches the public dataset and this repo.
PAPER = [
    # window, model, auc, ndcg, sensitivity, precision, hits
    ("2003-2005", "rusboost_raw28", 0.7428, 0.0394, 0.0453, 0.0502, 9),
    ("2003-2005", "logit_raw28", 0.6736, 0.0084, 0.0097, 0.0113, 2),
    ("2003-2005", "logit_ratios14", 0.6483, 0.0109, 0.0137, 0.0128, 2),
    ("2003-2008", "rusboost_raw28", 0.7228, 0.0237, 0.0291, 0.0281, 10),
    ("2003-2008", "logit_raw28", 0.6842, 0.0042, 0.0048, 0.0057, 2),
    ("2003-2008", "logit_ratios14", 0.6711, 0.0273, 0.0399, 0.0262, 8),
    ("2003-2014", "rusboost_raw28", 0.7196, 0.0182, 0.0244, 0.0187, 13),
    ("2003-2014", "logit_raw28", 0.7061, 0.0101, 0.0172, 0.0090, 6),
    ("2003-2014", "logit_ratios14", 0.7015, 0.0223, 0.0345, 0.0185, 11),
]


def _rows(model, year, seed, variant, y, s, n_recoded):
    out = []
    for top in TOPS:
        m = evaluate(y, s, top)
        out.append({"model": model, "variant": variant, "test_year": year, "seed": seed,
                    "top": top, "n_recoded": n_recoded, **m})
    return out


def tune_lightgbm(df):
    """Pick LightGBM size on the paper's own validation design: train 1991-1999,
    validate on 2001 (Bao et al. tuned RUSBoost the same way). No test year is used."""
    sp = out_of_time_split(df, 2001, gap=2)
    Xtr = df.loc[sp.train_idx, RAW28].to_numpy()
    Xva = df.loc[sp.test_idx, RAW28].to_numpy()
    rows = []
    for n_est in (100, 300, 600):
        for leaves in (7, 15, 31):
            m = models.lightgbm(n_estimators=n_est, num_leaves=leaves).fit(Xtr, sp.y_train)
            auc = evaluate(sp.y_test, models.score(m, Xva))["auc"]
            rows.append({"n_estimators": n_est, "num_leaves": leaves, "val_auc_2001": auc})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "lightgbm_tuning.csv", index=False)
    best = tab.sort_values("val_auc_2001", ascending=False).iloc[0]
    return {"n_estimators": int(best.n_estimators), "num_leaves": int(best.num_leaves)}


def run_all(df, lgb_params):
    rows = []
    shap_abs = []          # mean |SHAP| per feature per test year (LightGBM)
    contrib_store = {}     # per test year: (row index, contributions) for headline window
    rus_importance = []
    for t in TEST_YEARS:
        t0 = time.time()
        sp = out_of_time_split(df, t, gap=2, handle_serial_fraud=True)
        sp_raw = out_of_time_split(df, t, gap=2, handle_serial_fraud=False)
        tr, te = df.loc[sp.train_idx], df.loc[sp.test_idx]
        y = sp.y_test

        rows += _rows("beneish_mscore", t, 0, "main", y, te["m_score"].to_numpy(), 0)

        Xr_tr, Xr_te = tr[RATIO14].to_numpy(), te[RATIO14].to_numpy()
        X_tr, X_te = tr[RAW28].to_numpy(), te[RAW28].to_numpy()

        m = models.logit_ratios().fit(Xr_tr, sp.y_train)
        rows += _rows("logit_ratios14", t, 0, "main", y, models.score(m, Xr_te), sp.n_recoded)
        m = models.logit_raw().fit(X_tr, sp.y_train)
        rows += _rows("logit_raw28", t, 0, "main", y, models.score(m, X_te), sp.n_recoded)

        for seed in RUS_SEEDS:
            m = models.rusboost(seed).fit(X_tr, sp.y_train)
            rows += _rows("rusboost_raw28", t, seed, "main", y, models.score(m, X_te), sp.n_recoded)
            rus_importance.append(pd.Series(m.feature_importances_, index=RAW28, name=(t, seed)))
            m = models.rusboost(seed).fit(X_tr, sp_raw.y_train)
            rows += _rows("rusboost_raw28", t, seed, "no_serial_fix", y, models.score(m, X_te), 0)

        m = models.lightgbm(**lgb_params).fit(X_tr, sp.y_train)
        s = models.score(m, X_te)
        rows += _rows("lightgbm_raw28", t, 0, "main", y, s, sp.n_recoded)
        contrib = m.predict(X_te, pred_contrib=True)  # TreeSHAP, log-odds units
        shap_abs.append(pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=RAW28, name=t))
        if 2003 <= t <= 2008:
            contrib_store[t] = (sp.test_idx, s, contrib)
        m = models.lightgbm(**lgb_params).fit(X_tr, sp_raw.y_train)
        rows += _rows("lightgbm_raw28", t, 0, "no_serial_fix", y, models.score(m, X_te), 0)

        m = models.BaggedMLP(seed=0).fit(X_tr, sp.y_train)
        rows += _rows("mlp_raw28", t, 0, "main", y, models.score(m, X_te), sp.n_recoded)

        print(f"  test year {t}: train {len(sp.train_idx):,} rows "
              f"({int(sp.y_train.sum())} frauds, {sp.n_recoded} recoded), "
              f"test {len(y):,} rows ({int(y.sum())} frauds)  [{time.time() - t0:.0f}s]",
              flush=True)
    per_year = pd.DataFrame(rows)
    return per_year, pd.DataFrame(shap_abs), contrib_store, pd.DataFrame(rus_importance)


def summarise(per_year):
    """Average per-year metrics over each window (the paper's convention); for
    RUSBoost, average over seeds too and report the seed range."""
    out = []
    for wname, (a, b) in WINDOWS.items():
        w = per_year[(per_year.test_year >= a) & (per_year.test_year <= b)]
        for (model, variant, top), g in w.groupby(["model", "variant", "top"]):
            by_seed = g.groupby("seed").agg(
                auc=("auc", "mean"), ndcg_at_k=("ndcg_at_k", "mean"),
                precision_topk=("precision_topk", "mean"),
                sensitivity_topk=("sensitivity_topk", "mean"),
                hits=("hits_topk", "sum"), k=("k", "sum"), n_fraud=("n_fraud", "sum"))
            row = {"window": wname, "model": model, "variant": variant, "top": top,
                   "n_seeds": len(by_seed)}
            for c in ["auc", "ndcg_at_k", "precision_topk", "sensitivity_topk", "hits"]:
                row[c] = by_seed[c].mean()
                row[c + "_min"] = by_seed[c].min()
                row[c + "_max"] = by_seed[c].max()
            row["flagged_k"] = int(by_seed["k"].iloc[0])
            row["test_frauds"] = int(by_seed["n_fraud"].iloc[0])
            out.append(row)
    return pd.DataFrame(out)


def mscore_threshold_stats(df):
    w = df[(df.fyear >= 2003) & (df.fyear <= 2008)]
    flag = w["m_score"] > beneish.THRESHOLD
    y = w["misstate"] == 1
    return {
        "window": HEADLINE, "threshold": beneish.THRESHOLD,
        "firm_years": int(len(w)), "flagged": int(flag.sum()),
        "flag_rate": float(flag.mean()),
        "precision": float((flag & y).sum() / flag.sum()),
        "recall": float((flag & y).sum() / y.sum()),
        "base_rate": float(y.mean()),
    }


def explain_examples(df, contrib_store):
    """Two firm-years from the headline window, explained with LightGBM TreeSHAP:
    (a) the single highest-scored firm-year, whatever its label; (b) the
    highest-ranked firm-year that was in fact an AAER fraud year."""
    recs = []
    for t, (idx, s, contrib) in contrib_store.items():
        pct = pd.Series(s).rank(pct=True).to_numpy()
        rank = pd.Series(-s).rank(method="first").to_numpy().astype(int)
        for i in range(len(idx)):
            recs.append((t, idx[i], s[i], pct[i], rank[i], i))
    tab = pd.DataFrame(recs, columns=["test_year", "row", "score", "pct", "rank", "pos"])
    tab["misstate"] = df.loc[tab.row, "misstate"].to_numpy()
    tab["year_n"] = tab.groupby("test_year").row.transform("size")
    picks = {
        "top_scored": tab.sort_values("score", ascending=False).iloc[0],
        "top_true_fraud": tab[tab.misstate == 1].sort_values(["rank", "score"],
                                                             ascending=[True, False]).iloc[0],
    }
    out = {}
    for name, p in picks.items():
        t = int(p.test_year)
        contrib = contrib_store[t][2][int(p.pos)]
        row = df.loc[int(p.row)]
        year_med = df.loc[df.fyear == t, RAW28].median()
        drivers = (pd.DataFrame({"feature": RAW28, "shap_logodds": contrib[:-1],
                                 "value": row[RAW28].astype(float).to_numpy(),
                                 "year_median": year_med.to_numpy()})
                   .assign(abs=lambda d: d.shap_logodds.abs())
                   .sort_values("abs", ascending=False).drop(columns="abs").head(8))
        out[name] = {
            "gvkey": int(row.gvkey), "fyear": t, "misstate": int(row.misstate),
            "p_aaer": None if pd.isna(row.p_aaer) else int(row.p_aaer),
            "score": float(p.score), "rank_in_year": int(p["rank"]),
            "firms_in_year": int(p.year_n),
            "base_value_logodds": float(contrib[-1]),
            "m_score": float(row.m_score),
            "drivers": drivers.round(4).to_dict(orient="records"),
        }
    return out


def make_figures(per_year, summary, shap_abs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    order = list(MODEL_LABELS)
    colors = {"beneish_mscore": "#8c8c8c", "logit_ratios14": "#1f77b4",
              "logit_raw28": "#9ecae1", "rusboost_raw28": "#d62728",
              "lightgbm_raw28": "#ff7f0e", "mlp_raw28": "#2ca02c"}

    # 1. AUC by test year
    main = per_year[(per_year.variant == "main") & (per_year.top == 0.01)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for mname in order:
        g = main[main.model == mname].groupby("test_year").auc.mean()
        ax.plot(g.index, g.values, marker="o", lw=2 if "rus" in mname or "light" in mname else 1.3,
                color=colors[mname], label=MODEL_LABELS[mname])
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.axvspan(2002.5, 2008.5, color="#f0f0f0", zorder=0)
    ax.text(2005.5, 0.37, "paper's test window (2003-2008)", ha="center", fontsize=9, color="#555")
    ax.set_xlabel("Test year (model trained on 1991 to test year minus 2)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 0.9)
    ax.set_title("Out-of-time AUC by test year")
    ax.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "auc_by_year.png", dpi=150)
    plt.close(fig)

    # 2. Precision in the top k% (headline window)
    s = summary[(summary.window == HEADLINE) & (summary.variant == "main")]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.13
    xs = np.arange(len(TOPS))
    for i, mname in enumerate(order):
        g = s[s.model == mname].set_index("top").reindex(TOPS)
        ax.bar(xs + (i - 2.5) * width, 100 * g.precision_topk, width,
               color=colors[mname], label=MODEL_LABELS[mname])
    base = 100 * per_year[(per_year.test_year.between(2003, 2008)) & (per_year.model == "beneish_mscore")
                          & (per_year.top == 0.01)].eval("n_fraud / n").mean()
    ax.axhline(base, color="black", ls="--", lw=1)
    ax.text(-0.45, base + 0.05, f"random pick: {base:.2f}%", ha="left", fontsize=9)
    ax.set_xticks(xs, [f"top {int(t * 100)}%" for t in TOPS])
    ax.set_ylabel("Precision (% of flagged firm-years that were AAER frauds)")
    ax.set_title("Hit rate when an analyst reviews the highest-scored firms, 2003-2008 average")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "topk_precision.png", dpi=150)
    plt.close(fig)

    # 3. Global SHAP importance (LightGBM, headline window)
    imp = shap_abs.loc[2003:2008].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(imp.index, imp.values, color="#ff7f0e")
    ax.set_xlabel("Mean |SHAP| (log-odds), test years 2003-2008")
    ax.set_title("What drives the LightGBM fraud score")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_importance.png", dpi=150)
    plt.close(fig)


def write_markdown(summary, thr, examples, info):
    lines = []
    paper = pd.DataFrame(PAPER, columns=["window", "model", "auc", "ndcg_at_k",
                                         "sensitivity_topk", "precision_topk", "hits"])
    s1 = summary[(summary.top == 0.01)]
    for wname in WINDOWS:
        lines.append(f"\n### Test window {wname} (top 1% cut-off)\n")
        lines.append("| Model | AUC | NDCG@k | Precision top 1% | Frauds caught (sum over years) "
                     "| Paper AUC | Paper NDCG@k | Paper precision | Paper caught |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for mname in MODEL_LABELS:
            r = s1[(s1.window == wname) & (s1.model == mname) & (s1.variant == "main")].iloc[0]
            p = paper[(paper.window == wname) & (paper.model == mname)]
            pa = p.iloc[0] if len(p) else None
            seeds = ""
            if r.n_seeds > 1:
                seeds = f" ({r.auc_min:.3f}-{r.auc_max:.3f})"
            hits = f"{r.hits:.1f}" if r.n_seeds > 1 else f"{int(r.hits)}"
            if r.n_seeds > 1:
                hits += f" ({int(r.hits_min)}-{int(r.hits_max)})"
            lines.append(
                f"| {MODEL_LABELS[mname]} | {r.auc:.3f}{seeds} | {r.ndcg_at_k:.3f} | "
                f"{100 * r.precision_topk:.2f}% | {hits} of {r.test_frauds} | "
                + (f"{pa.auc:.3f} | {pa.ndcg_at_k:.3f} | {100 * pa.precision_topk:.2f}% | {pa.hits} |"
                   if pa is not None else "- | - | - | - |"))
        lines.append(f"\nTop 1% = {int(s1[(s1.window == wname)].flagged_k.iloc[0])} firm-years "
                     f"flagged in total across the window.")
    lines.append("\n### Serial-fraud ablation (top 1%)\n")
    lines.append("| Model | Window | AUC with fix | AUC without fix | Caught with fix | Caught without fix |")
    lines.append("|---|---|---|---|---|---|")
    for mname in ["rusboost_raw28", "lightgbm_raw28"]:
        for wname in WINDOWS:
            a = s1[(s1.window == wname) & (s1.model == mname) & (s1.variant == "main")].iloc[0]
            b = s1[(s1.window == wname) & (s1.model == mname) & (s1.variant == "no_serial_fix")].iloc[0]
            lines.append(f"| {MODEL_LABELS[mname]} | {wname} | {a.auc:.3f} | {b.auc:.3f} | "
                         f"{a.hits:.1f} | {b.hits:.1f} |")
    lines.append("\n### Beneish threshold (M > -1.78), 2003-2008\n")
    lines.append(f"Flags {thr['flagged']:,} of {thr['firm_years']:,} firm-years "
                 f"({100 * thr['flag_rate']:.1f}%); precision {100 * thr['precision']:.2f}% "
                 f"vs base rate {100 * thr['base_rate']:.2f}%; recall {100 * thr['recall']:.1f}%.")
    lines.append("\n### Explained examples (LightGBM TreeSHAP)\n")
    for name, e in examples.items():
        lines.append(f"**{name}**: gvkey {e['gvkey']}, fiscal {e['fyear']}, AAER label "
                     f"{e['misstate']}, score {e['score']:.3f}, rank {e['rank_in_year']} of "
                     f"{e['firms_in_year']}, M-score {e['m_score']:.2f}\n")
        lines.append("| Item | Value ($m) | Year median | SHAP (log-odds) |")
        lines.append("|---|---|---|---|")
        for d in e["drivers"]:
            lines.append(f"| {d['feature']} | {d['value']:,.3f} | {d['year_median']:,.3f} | "
                         f"{d['shap_logodds']:+.3f} |")
        lines.append("")
    lines.append(f"\nRun: {info['runtime_min']:.1f} min, peak RSS {info['peak_rss_mb']:.0f} MB, "
                 f"Python {info['python']}.")
    (RESULTS / "results.md").write_text("# Generated results\n" + "\n".join(lines) + "\n")


def main():
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    print("Loading data")
    df = load()
    comp = beneish.components(df)
    df = df.join(comp[beneish.COMPONENTS + ["m_score", "lag_available", "n_components_computed"]])
    coverage = {c: float(comp[f"{c}_computed"].mean()) for c in beneish.COMPONENTS}

    frauds = df[df.misstate == 1]
    data_summary = {
        "source": "Bao, Ke, Li, Yu & Zhang (2020), github.com/JarFraud/FraudDetection",
        "firm_years": int(len(df)), "fiscal_years": [int(df.fyear.min()), int(df.fyear.max())],
        "fraud_firm_years": int(len(frauds)), "fraud_cases_p_aaer": int(frauds.p_aaer.nunique()),
        "fraud_rate": float(df.misstate.mean()),
        "frauds_by_year": {int(k): int(v) for k, v in df.groupby("fyear").misstate.sum().items()},
        "beneish_prior_year_available": float(df.lag_available.mean()),
        "beneish_all_7_computable_components": float((df.n_components_computed == 7).mean()),
        "beneish_component_coverage": coverage,
    }
    (RESULTS / "data_summary.json").write_text(json.dumps(data_summary, indent=2))

    print("Tuning LightGBM on 1991-1999 -> 2001 validation")
    lgb_params = tune_lightgbm(df)
    print(f"  chosen: {lgb_params}")

    print("Rolling out-of-time evaluation")
    per_year, shap_abs, contrib_store, rus_imp = run_all(df, lgb_params)
    per_year.to_csv(RESULTS / "metrics_by_year.csv", index=False)
    summary = summarise(per_year)
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    pd.DataFrame(PAPER, columns=["window", "model", "auc", "ndcg_at_k", "sensitivity_topk",
                                 "precision_topk", "hits"]).assign(
        source="Bao et al. (2021) Econ Journal Watch 18(1) Table 1 (quoted)"
    ).to_csv(RESULTS / "paper_reported.csv", index=False)
    shap_abs.to_csv(RESULTS / "lightgbm_mean_abs_shap_by_year.csv")
    rus_imp.mean().sort_values(ascending=False).rename("mean_impurity_importance").to_csv(
        RESULTS / "rusboost_feature_importance.csv")

    thr = mscore_threshold_stats(df)
    (RESULTS / "mscore_threshold.json").write_text(json.dumps(thr, indent=2))
    examples = explain_examples(df, contrib_store)
    (RESULTS / "explained_examples.json").write_text(json.dumps(examples, indent=2))

    make_figures(per_year, summary, shap_abs)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    info = {"runtime_min": (time.time() - start) / 60, "peak_rss_mb": peak_mb,
            "python": platform.python_version(), "lightgbm_params": lgb_params,
            "rusboost_seeds": RUS_SEEDS}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=2))
    write_markdown(summary, thr, examples, info)
    print(f"Done in {info['runtime_min']:.1f} min, peak RSS {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
