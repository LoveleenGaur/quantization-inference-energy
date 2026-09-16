import pathlib
# Standalone use: read the released run records.
RAW = pathlib.Path("data/runs_48.csv")
OUT = pathlib.Path(".")

# CELL 6 (replacement)  analysis across model sizes.
#
# Change of approach from the first version. The first version dropped throttled
# runs, which is wrong here: fp16 draws more power, so fp16 is the condition
# that hits the 70 W cap, and dropping those runs removes the high-power
# condition preferentially. It left n = 2 for 1.5B fp16.
#
# All runs are now reported as the primary result, power capping is disclosed as
# a hardware characteristic, and the exclusion is reported as a sensitivity
# analysis so a reader can see the conclusion does not depend on it.

import pandas as pd, numpy as np
from scipy import stats

d = pd.read_csv(RAW)
d = d[d.ok == True].copy()
d["throttled"] = ~d.throttle.isin(["none", "unavailable"])
# order model sizes by parameter count parsed from the name
d["size_b"] = d.model.str.extract(r"(\d+\.?\d*)B")[0].astype(float)
d = d.sort_values("size_b")

def mean_ci(x):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 2: return x.mean() if n else np.nan, np.nan, np.nan
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n-1) * s / np.sqrt(n)
    return m, m-h, m+h

print("="*78); print("Power capping incidence (disclose, do not discard)"); print("="*78)
print(pd.crosstab([d.model, d.condition], d.throttled,
                  rownames=["model","condition"], colnames=["capped"]).to_string())

print("\n" + "="*78); print("PRIMARY RESULT: all runs included"); print("="*78)
rows = []
for (mid, size), g in d.groupby(["model","size_b"]):
    a, b = g[g.condition=="fp16"], g[g.condition=="nf4"]
    if not (len(a) and len(b)): continue
    ma, la, ha = mean_ci(a.wh_per_1k_tokens)
    mb, lb, hb = mean_ci(b.wh_per_1k_tokens)
    de = (mb-ma)/ma*100
    p = stats.ttest_ind(b.wh_per_1k_tokens, a.wh_per_1k_tokens, equal_var=False).pvalue \
        if len(a)>1 and len(b)>1 else np.nan
    pooled = np.sqrt((a.wh_per_1k_tokens.var(ddof=1)+b.wh_per_1k_tokens.var(ddof=1))/2) \
             if len(a)>1 and len(b)>1 else np.nan
    dcoh = (mb-ma)/pooled if pooled and pooled==pooled else np.nan
    memcol = "model_mem_mb" if "model_mem_mb" in g else "peak_mem_mb"
    rows.append({
        "model": mid.split("/")[-1], "size_b": size,
        "n_fp16": len(a), "n_nf4": len(b),
        "fp16_wh_per_1k": round(ma,4), "nf4_wh_per_1k": round(mb,4),
        "energy_pct": round(de,1), "p": round(p,5) if p==p else None,
        "cohens_d": round(dcoh,2) if dcoh==dcoh else None,
        "memory_pct": round((b[memcol].mean()-a[memcol].mean())/a[memcol].mean()*100,1),
        "throughput_pct": round((b.tokens_per_s.mean()-a.tokens_per_s.mean())/a.tokens_per_s.mean()*100,1),
        "perplexity_pct": round((b.perplexity.mean()-a.perplexity.mean())/a.perplexity.mean()*100,1),
        "fp16_capped": int(a.throttled.sum()), "nf4_capped": int(b.throttled.sum()),
    })
res = pd.DataFrame(rows).sort_values("size_b")
print(res.to_string(index=False))

print("\n" + "="*78); print("SENSITIVITY: does the conclusion depend on the capped runs?"); print("="*78)
for (mid, size), g in d.groupby(["model","size_b"]):
    variants = {"all runs": g,
                "capped excluded": g[~g.throttled],
                "capped only (fp16) vs all nf4":
                    pd.concat([g[(g.condition=="fp16") & g.throttled], g[g.condition=="nf4"]])}
    print(f"\n  {mid.split('/')[-1]}")
    for lbl, sub in variants.items():
        a, b = sub[sub.condition=="fp16"], sub[sub.condition=="nf4"]
        if not (len(a) and len(b)):
            print(f"    {lbl:32s} insufficient data"); continue
        de = (b.wh_per_1k_tokens.mean()-a.wh_per_1k_tokens.mean())/a.wh_per_1k_tokens.mean()*100
        p = stats.ttest_ind(b.wh_per_1k_tokens, a.wh_per_1k_tokens, equal_var=False).pvalue \
            if len(a)>1 and len(b)>1 else np.nan
        print(f"    {lbl:32s} n={len(a)}/{len(b)}  energy {de:+6.1f}%  "
              f"p={p:.3f}" if p==p else
              f"    {lbl:32s} n={len(a)}/{len(b)}  energy {de:+6.1f}%  p=n/a")

print("\n" + "="*78); print("SCALING: does the energy penalty decline with model size?"); print("="*78)
if len(res) >= 2:
    for _, r in res.iterrows():
        bar = "+" * max(0, int(round(r.energy_pct/5))) or ("0" if abs(r.energy_pct) < 2.5 else "")
        print(f"  {r.size_b:4.1f}B  energy {r.energy_pct:+6.1f}%  memory {r.memory_pct:+6.1f}%  "
              f"perplexity {r.perplexity_pct:+6.1f}%  {bar}")
    if len(res) >= 3:
        rho, prho = stats.spearmanr(res.size_b, res.energy_pct)
        print(f"\n  size vs energy penalty: Spearman rho = {rho:.2f}, p = {prho:.3f}")
        print("  monotonically declining:",
              bool(all(np.diff(res.energy_pct.values) < 0)))
    else:
        print("\n  two sizes only: report as a contrast, not a trend. A third size")
        print("  is needed before claiming the penalty declines monotonically.")

res.to_csv(OUT / "results_by_size.csv", index=False)
d.to_csv(OUT / "all_runs_clean.csv", index=False)
print(f"\nwrote {OUT/'results_by_size.csv'} and {OUT/'all_runs_clean.csv'}")
