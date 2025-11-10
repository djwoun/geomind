#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def normalize_modifier(s: pd.Series) -> pd.Series:
    s = s.fillna("None").astype(str).str.strip()
    s = s.replace({"": "None"})
    s = s.apply(lambda x: "None" if x.lower() in {"none", "nan"} else x)
    return s

def main():
    ap = argparse.ArgumentParser(
        description="Welch t-test per Modifier vs 'None' baseline within each AI model."
    )
    ap.add_argument("csv", type=Path, help="CSV with columns: AI, Modifier, Interpretation")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("modifier_vs_none_ttests.csv"))
    ap.add_argument("--only-model", action="append", default=None)
    ap.add_argument("--sort-by", choices=["p","diff","modifier"], default="p",
                    help="Console print order")
    ap.add_argument("--descending", action="store_true",
                    help="Sort descending for console print")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"AI","Modifier","Interpretation"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV must include {need}. Found {list(df.columns)}")

    df["AI"] = df["AI"].astype(str).str.strip()
    df["Modifier"] = normalize_modifier(df["Modifier"])
    df["y"] = df["Interpretation"].astype(str).str.strip().str.lower().eq("correct").astype(int)

    models = sorted(df["AI"].unique().tolist())
    if args.only_model:
        keep = set(args.only_model)
        models = [m for m in models if m in keep]
        if not models:
            raise SystemExit("No models matched --only-model filters.")

    rows = []
    for ai in models:
        d = df[df["AI"] == ai]
        base = d[d["Modifier"] == "None"]["y"].to_numpy()
        if base.size == 0:
            continue
        acc_base = float(base.mean())
        for mod in sorted(m for m in d["Modifier"].unique() if m != "None"):
            grp = d[d["Modifier"] == mod]["y"].to_numpy()
            n_mod = grp.size
            acc_mod = float(grp.mean()) if n_mod else np.nan
            if n_mod == 0:
                tstat, pval = np.nan, np.nan
            else:
                tstat, pval = stats.ttest_ind(grp, base, equal_var=False)
            diff = (acc_mod - acc_base) if n_mod else np.nan
            rows.append({
                "AI": ai,
                "Modifier": mod,
                "n_base_None": int(base.size),
                "acc_base_None": acc_base,
                "n_mod": int(n_mod),
                "acc_mod": acc_mod,
                "diff_mod_minus_base": diff,      # negative => lowers accuracy
                "Welch_tstat": float(tstat) if np.isfinite(tstat) else np.nan,
                "Welch_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "Use_if_avoiding_identifiability": bool((diff < 0) and (np.isfinite(pval) and pval <= args.alpha)),
            })

    out = pd.DataFrame(rows).sort_values(["AI","Modifier"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out)

    # Console: show ALL modifiers
    view = out.copy()
    # round for readability
    for col in ["acc_base_None","acc_mod","diff_mod_minus_base","Welch_pvalue","Welch_tstat"]:
        if col in view:
            view[col] = view[col].astype(float).round(4)

    key = {"p":"Welch_pvalue","diff":"diff_mod_minus_base","modifier":"Modifier"}[args.sort_by]
    view = view.sort_values([ "AI", key ], ascending=not args.descending)
    print("\nAll modifiers (per model):")
    print(view.to_string(index=False))

    # Also list the significant-lowering subset
    cand = out[out["Use_if_avoiding_identifiability"]]
    if not cand.empty:
        print("\nModifiers that significantly LOWER accuracy vs None (p <= alpha):")
        for _, r in cand.iterrows():
            print(f"- {r['AI']} | {r['Modifier']}: diff={r['diff_mod_minus_base']:.3f}, p={r['Welch_pvalue']:.3g}")
    else:
        print("\nNo modifiers met the criterion at alpha =", args.alpha)

if __name__ == "__main__":
    main()
