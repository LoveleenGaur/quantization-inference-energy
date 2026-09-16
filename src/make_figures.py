import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, pathlib, os
os.makedirs("figures", exist_ok=True)
from scipy import stats
MM = 1/25.4
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
 "font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,"xtick.labelsize":7.5,
 "ytick.labelsize":7.5,"legend.fontsize":7.5,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"savefig.dpi":300,
 "savefig.bbox":"tight","savefig.pad_inches":0.05,"pdf.fonttype":42})
DARK, LIGHT = "#3d3d3d", "#b0b0b0"
d = pd.read_csv("data/runs_48.csv")
# Map the harness column names onto the short names used below.
d["size_b"] = d.model.str.extract(r"(\d+\.?\d*)B")[0].astype(float)
d = d.rename(columns={"wh_per_1k_tokens": "wh_per_1k", "tokens_per_s": "tok_s",
                      "model_mem_mb": "mem_mb", "perplexity": "ppl"})
d["capped"] = ~d.throttle.isin(["none", "unavailable"])
sizes = [0.5, 1.5, 3.0]
labels = ["0.5B", "1.5B", "3B"]

def ci(x):
    x=np.asarray(x,float); n=len(x)
    return stats.t.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n)

# ---- Fig 1: energy per 1000 tokens, by size and precision ----
fig, ax = plt.subplots(figsize=(115*MM, 62*MM))
w, x = 0.36, np.arange(3)
for i,(cond,lbl,col) in enumerate([("fp16","16-bit",DARK),("nf4","4-bit",LIGHT)]):
    m=[d[(d.size_b==s)&(d.condition==cond)].wh_per_1k.mean() for s in sizes]
    e=[ci(d[(d.size_b==s)&(d.condition==cond)].wh_per_1k) for s in sizes]
    ax.bar(x+(i-0.5)*w, m, w, yerr=e, capsize=2.5, label=lbl, color=col,
           edgecolor="black", linewidth=0.6, error_kw={"elinewidth":0.7})
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Model size"); ax.set_ylabel("Energy (Wh per 1000 tokens)")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
ax.annotate("+52.5%", xy=(0+0.5*w, 0.60), ha="center", fontsize=7.5)
ax.annotate("$-$0.3%", xy=(1+0.5*w, 0.76), ha="center", fontsize=7.5)
ax.annotate("$-$1.6%", xy=(2+0.5*w, 1.01), ha="center", fontsize=7.5)
ax.set_ylim(0, 1.15)
fig.tight_layout(); fig.savefig("figures/Fig1.png"); plt.close(fig)

# ---- Fig 3: percentage change from 16-bit on four measures ----
fig, ax = plt.subplots(figsize=(167*MM, 62*MM))
measures = [("Energy per 1000 tokens","wh_per_1k"),("Peak memory","mem_mb"),
            ("Throughput","tok_s"),("Perplexity","ppl")]
x = np.arange(len(measures)); w = 0.26
greys = ["#2f2f2f","#7a7a7a","#c4c4c4"]
for i,(s,lbl) in enumerate(zip(sizes,labels)):
    vals=[]
    for _,col in measures:
        a=d[(d.size_b==s)&(d.condition=="fp16")][col].mean()
        b=d[(d.size_b==s)&(d.condition=="nf4")][col].mean()
        vals.append((b-a)/a*100)
    bars=ax.bar(x+(i-1)*w, vals, w, label=lbl, color=greys[i],
                edgecolor="black", linewidth=0.6)
    for bb,v in zip(bars,vals):
        lab = f"{v:+.1f}" if abs(v) < 5 else f"{v:+.0f}"
        ax.text(bb.get_x()+bb.get_width()/2, v + (1.8 if v>=0 else -4.6),
                lab, ha="center", fontsize=6.8)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in measures])
ax.set_ylabel("Change from 16-bit (%)")
ax.legend(frameon=False, title="Model size", ncol=3, loc="upper right",
          bbox_to_anchor=(1.0, 1.02))
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(-80, 78)
fig.tight_layout(); fig.savefig("figures/Fig3.png"); plt.close(fig)

# ---- Fig 2: per-run distributions, showing separation only at 0.5B ----
fig, ax = plt.subplots(figsize=(115*MM, 62*MM))
pos, ticks, tlab = 0, [], []
rng = np.random.default_rng(7)
for s,lbl in zip(sizes,labels):
    for cond,mark,col in [("fp16","o",DARK),("nf4","s",LIGHT)]:
        y=d[(d.size_b==s)&(d.condition==cond)].wh_per_1k.values
        ax.scatter(pos+rng.uniform(-0.10,0.10,len(y)), y, s=13, marker=mark,
                   facecolor=col, edgecolor="black", linewidth=0.5, zorder=3)
        ax.plot([pos-0.22,pos+0.22],[y.mean()]*2, color="black", linewidth=1.1, zorder=4)
        ticks.append(pos); tlab.append(f"{lbl}\n{'16-bit' if cond=='fp16' else '4-bit'}")
        pos += 1
    pos += 0.45
ax.set_xticks(ticks); ax.set_xticklabels(tlab, fontsize=6.8)
ax.set_ylabel("Energy (Wh per 1000 tokens)")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/Fig2.png"); plt.close(fig)

from PIL import Image
for n in ["Fig1","Fig2","Fig3"]:
    im=Image.open(f"figures/{n}.png")
    print(f"{n}: {im.size[0]}x{im.size[1]} px -> {im.size[0]/300*25.4:.0f} mm at 300 dpi")
