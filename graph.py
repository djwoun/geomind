#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def find_model_like(values, substrs):
    t = [s.lower() for s in substrs]
    for v in values:
        vl = str(v).lower()
        if all(ss in vl for ss in t):
            return v
    for v in values:
        vl = str(v).lower()
        if t[0] in vl:
            return v
    return None

def normalize_modifier(s: pd.Series) -> pd.Series:
    # Coerce to string, handle NaN/blank/none variants
    s = s.fillna("None").astype(str).str.strip()
    s = s.mask(s.str.len() == 0, "None")
    s = s.where(~s.str.lower().isin({"none", "nan"}), "None")
    return s

def order_modifiers_from(series: pd.Series):
    vals = normalize_modifier(series)
    rest = sorted([v for v in set(vals) if v != "None"])
    return ["None"] + rest

def counts_by_modifier(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    sub = df[df["AI"] == model_label].copy()
    sub["Modifier"] = normalize_modifier(sub["Modifier"])
    sub["Interpretation_norm"] = sub["Interpretation"].astype(str).str.strip().str.lower()
    grp = sub.groupby(["Modifier", "Interpretation_norm"]).size().unstack(fill_value=0)
    grp = grp.rename(columns={"correct": "Correct", "incorrect": "Incorrect"})
    for needed in ["Correct", "Incorrect"]:
        if needed not in grp.columns:
            grp[needed] = 0
    grp = grp[["Correct", "Incorrect"]]
    # Reindex so "None" is first and all categories appear
    wanted = order_modifiers_from(sub["Modifier"])
    grp = grp.reindex(wanted).fillna(0).astype(int)
    return grp

def plot_counts(counts: pd.DataFrame, title: str, outfile: Path):
    ax = counts.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_xlabel("Modifier")
    ax.set_ylabel("Count")
    ax.set_title(title)
    # Legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Plot Correct vs Incorrect by Modifier for Gemini 2.5 Pro and Claude.")
    p.add_argument("csv", type=Path, help="Path to Final_results.csv")
    p.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for charts and summary CSV")
    p.add_argument("--gemini-like", nargs="*", default=["gemini", "2.5", "pro"], help="Substrings to detect Gemini label")
    p.add_argument("--claude-like", nargs="*", default=["claude"], help="Substrings to detect Claude label")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"AI", "Modifier", "Interpretation"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"CSV must include columns {needed}")

    # Normalize modifier column globally
    df["Modifier"] = normalize_modifier(df["Modifier"])
    # Force CSV sort order with 'None' first
    mod_order = order_modifiers_from(df["Modifier"])
    df["Modifier"] = pd.Categorical(df["Modifier"], categories=mod_order, ordered=True)

    ai_values = set(df["AI"].unique().tolist())
    gemini_label = find_model_like(ai_values, args.gemini_like) or "Gemini 2.5 Pro"
    claude_label = find_model_like(ai_values, args.claude_like) or "Claude Sonnet 4.5"

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Summary CSV with enforced order
    df["Interpretation_norm"] = df["Interpretation"].astype(str).str.strip().str.lower()
    summary = (
        df.assign(Interpretation=df["Interpretation_norm"].str.title())
          .groupby(["AI", "Modifier", "Interpretation"]).size()
          .unstack(fill_value=0)
          .reset_index()
          .sort_values(["AI", "Modifier"])
    )
    summary.to_csv(args.outdir / "correct_incorrect_by_model_modifier.csv", index=False)

    # Charts
    g_counts = counts_by_modifier(df, gemini_label)
    c_counts = counts_by_modifier(df, claude_label)

    plot_counts(g_counts, f"{gemini_label}: Correct vs Incorrect by Modifier",
                args.outdir / "gemini_2_5_pro_correct_incorrect_by_modifier.png")
    plot_counts(c_counts, f"{claude_label}: Correct vs Incorrect by Modifier",
                args.outdir / "claude_correct_incorrect_by_modifier.png")

    print("Wrote:")
    print(args.outdir / "correct_incorrect_by_model_modifier.csv")
    print(args.outdir / "gemini_2_5_pro_correct_incorrect_by_modifier.png")
    print(args.outdir / "claude_correct_incorrect_by_modifier.png")

if __name__ == "__main__":
    main()
