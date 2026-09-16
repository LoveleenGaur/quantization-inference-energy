# %% [markdown]
# # Does quantization save energy, or only memory? (lite)
#
# Measures energy per 1000 generated tokens at full precision against 4-bit,
# on one small model, with replicates. Designed to finish in about 10 minutes
# on a free T4 and to survive a dropped session.
#
# **What makes this the lite version**
#
# * One 0.5B model instead of two 1.5B models.
# * Two conditions (fp16, nf4) instead of three. int8 on Turing runs about
#   three times slower than fp16 and dominated the runtime.
# * 8 prompts x 64 new tokens = 512 measured tokens per run, down from 3072.
# * **The model loads 4 times in total, not 12.** The old design reloaded for
#   every replicate, which is where the RAM and disk churn came from. Here the
#   replicates for a condition run against one loaded copy.
# * Every run appends to CSV the moment it finishes, and re-running the cell
#   skips work already recorded. A dropped session costs one run.
#
# **What is kept, because the paper depends on it**
#
# Energy from the GPU's cumulative energy counter, an idle baseline before each
# block, throttle detection, measured rather than assumed token counts, a
# discarded warm-up, and condition order reversed between blocks so thermal
# drift does not land on one condition.

# %%
# CELL 1  install (Colab). Set Runtime > Change runtime type > T4 GPU first.
# !pip -q install transformers accelerate bitsandbytes nvidia-ml-py
# !nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv

# %%
# CELL 2  preconditions, config, output location
import os, gc, json, time, math, csv, random, statistics, platform, pathlib, threading
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA device. Set Runtime > Change runtime type > T4 GPU.")
try:
    import pynvml; pynvml.nvmlInit()
except Exception as e:
    raise SystemExit(f"pynvml missing ({e}). Run the install cell.")

H = pynvml.nvmlDeviceGetHandleByIndex(0)
dec = lambda x: x.decode() if isinstance(x, bytes) else x
GPU, DRIVER = dec(pynvml.nvmlDeviceGetName(H)), dec(pynvml.nvmlSystemGetDriverVersion())
try:
    pynvml.nvmlDeviceGetTotalEnergyConsumption(H); COUNTER = True
except Exception:
    COUNTER = False

CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",   # ~1 GB in fp16; swap if you prefer
    "conditions": ["fp16", "nf4"],
    "n_prompts": 8,
    "max_new_tokens": 64,
    "reps_per_block": 2,        # 2 blocks x 2 reps = 4 runs per condition
    "blocks": 2,                # block 2 reverses condition order
    "idle_settle_s": 6,
    "power_hz": 10,
    "seed": 20260915,
    # Write to Drive if it is mounted, so results survive a runtime recycle.
    "out_dir": "/content/drive/MyDrive/energy_lite"
               if pathlib.Path("/content/drive/MyDrive").exists() else "/content/energy_lite",
}
random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
OUT = pathlib.Path(CONFIG["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw_runs.csv"

PROMPTS = [
    "Explain in two sentences why rivers meander.",
    "Summarise the difference between weather and climate.",
    "What causes urban heat islands?",
    "Explain thermal stratification in a lake.",
    "Why does deforestation affect local rainfall?",
    "What is albedo and why does it matter?",
    "Describe how a wetland filters water.",
    "Explain evapotranspiration briefly.",
][:CONFIG["n_prompts"]]

HELD_OUT = ("A catchment is the area of land where precipitation collects and drains "
            "into a common outlet. Water moves through it by surface runoff, infiltration "
            "into soil, and shallow subsurface flow, and the relative share of each pathway "
            "governs how quickly a storm produces a rise in river level.")

print(f"GPU {GPU} | driver {DRIVER} | energy counter {COUNTER}")
print(f"model {CONFIG['model']}")
print(f"runs: {CONFIG['blocks']} blocks x {len(CONFIG['conditions'])} conditions x "
      f"{CONFIG['reps_per_block']} reps = "
      f"{CONFIG['blocks']*len(CONFIG['conditions'])*CONFIG['reps_per_block']} measured")
print(f"model loads: {CONFIG['blocks']*len(CONFIG['conditions'])}")
print(f"output -> {OUT}")

# %%
# CELL 3  measurement primitives

class Power(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.s, self.t, self._q = [], [], threading.Event()
    def run(self):
        t0 = time.perf_counter()
        while not self._q.is_set():
            try:
                self.s.append((time.perf_counter()-t0,
                               pynvml.nvmlDeviceGetPowerUsage(H)/1000.0))
                self.t.append(pynvml.nvmlDeviceGetTemperature(H, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            time.sleep(1.0/CONFIG["power_hz"])
    def stop(self): self._q.set(); self.join(timeout=2)
    def wh(self):
        if len(self.s) < 2: return float("nan")
        j = sum((w1+w2)/2*(t2-t1) for (t1,w1),(t2,w2) in zip(self.s, self.s[1:]))
        return j/3600.0
    def mean_w(self): return statistics.mean(w for _, w in self.s) if self.s else float("nan")

def throttling():
    try:
        b = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(H)
    except Exception:
        return "unavailable"
    f = {"sw_power_cap": pynvml.nvmlClocksThrottleReasonSwPowerCap,
         "hw_slowdown": pynvml.nvmlClocksThrottleReasonHwSlowdown,
         "sw_thermal": pynvml.nvmlClocksThrottleReasonSwThermalSlowdown,
         "hw_thermal": pynvml.nvmlClocksThrottleReasonHwThermalSlowdown}
    hit = [n for n, bit in f.items() if b & bit]
    return ";".join(hit) if hit else "none"

def idle_baseline():
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    p = Power(); p.start(); time.sleep(CONFIG["idle_settle_s"]); p.stop()
    return p.mean_w(), (statistics.mean(p.t) if p.t else float("nan"))

class Measure:
    def __enter__(self):
        torch.cuda.synchronize()
        self.p = Power(); self.p.start()
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(H) if COUNTER else None
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.secs = time.perf_counter() - self.t0
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(H) if COUNTER else None
        self.p.stop()
        self.counter_wh = ((e1-self.e0)/1000.0/3600.0) if COUNTER else float("nan")
        self.integ_wh = self.p.wh()
        self.mean_power = self.p.mean_w()
        self.max_temp = max(self.p.t) if self.p.t else float("nan")
        self.throttle = throttling()
        return False

COLS = ["model","condition","block","rep","warmup","new_tokens","seconds","tokens_per_s",
        "energy_wh_counter","energy_wh_integrated","counter_vs_integrated_pct",
        "wh_per_1k_tokens","energy_above_idle_wh","mean_power_w","idle_power_w",
        "peak_mem_mb","perplexity","max_temp_c","idle_temp_c","throttle","ok","error"]

def save_row(rec):
    new = not RAW.exists()
    with open(RAW, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow({c: rec.get(c) for c in COLS})

print("primitives ready")

# %%
# CELL 4  load, generate, perplexity

from transformers import AutoModelForCausalLM, AutoTokenizer

def load(cond):
    tok = AutoTokenizer.from_pretrained(CONFIG["model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"device_map": {"": 0}, "low_cpu_mem_usage": True}
    if cond == "fp16":
        kw["torch_dtype"] = torch.float16
    elif cond == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        raise ValueError(cond)
    m = AutoModelForCausalLM.from_pretrained(CONFIG["model"], **kw); m.eval()
    return m, tok

def release(*objs):
    """Free VRAM and host RAM between conditions. This is what the old version missed."""
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

@torch.inference_mode()
def run_prompts(m, tok, prompts):
    n = 0
    for p in prompts:
        try:
            txt = tok.apply_chat_template([{"role":"user","content":p}],
                                          tokenize=False, add_generation_prompt=True)
        except Exception:
            txt = p
        enc = tok(txt, return_tensors="pt").to(m.device)
        out = m.generate(**enc, max_new_tokens=CONFIG["max_new_tokens"],
                         min_new_tokens=CONFIG["max_new_tokens"],
                         do_sample=False, pad_token_id=tok.pad_token_id)
        n += out.shape[-1] - enc["input_ids"].shape[-1]
    return n

@torch.inference_mode()
def ppl(m, tok):
    enc = tok(HELD_OUT, return_tensors="pt").to(m.device)
    return float(torch.exp(m(**enc, labels=enc["input_ids"]).loss))

print("loader ready")

# %%
# CELL 5  measurement loop. Safe to re-run: completed runs are skipped.

done = set()
if RAW.exists():
    import pandas as pd
    for r in pd.read_csv(RAW).itertuples():
        done.add((r.condition, int(r.block), int(r.rep)))
    print(f"resuming, {len(done)} runs already recorded")

total = CONFIG["blocks"] * len(CONFIG["conditions"]) * CONFIG["reps_per_block"]
n_done, t_start = len(done), time.perf_counter()

for block in range(CONFIG["blocks"]):
    order = CONFIG["conditions"] if block % 2 == 0 else CONFIG["conditions"][::-1]
    for cond in order:
        todo = [r for r in range(CONFIG["reps_per_block"]) if (cond, block, r) not in done]
        if not todo:
            print(f"block {block} | {cond}: all reps recorded, skipping"); continue

        model = tok = None
        try:
            idle_w, idle_t = idle_baseline()
            model, tok = load(cond)                 # ONE load serves every rep here
            run_prompts(model, tok, PROMPTS[:2])    # untimed warm-up

            for rep in todo:
                torch.cuda.reset_peak_memory_stats()
                with Measure() as mz:
                    ntok = run_prompts(model, tok, PROMPTS)
                peak = torch.cuda.max_memory_allocated()/1e6
                q = ppl(model, tok)
                e = mz.counter_wh if COUNTER else mz.integ_wh
                dis = (abs(mz.counter_wh-mz.integ_wh)/max(mz.counter_wh,1e-9)*100
                       if COUNTER and not math.isnan(mz.counter_wh) else float("nan"))
                rec = {"model": CONFIG["model"], "condition": cond, "block": block, "rep": rep,
                       "warmup": False, "new_tokens": ntok, "seconds": round(mz.secs,2),
                       "tokens_per_s": round(ntok/mz.secs,2),
                       "energy_wh_counter": round(mz.counter_wh,6),
                       "energy_wh_integrated": round(mz.integ_wh,6),
                       "counter_vs_integrated_pct": round(dis,1) if not math.isnan(dis) else None,
                       "wh_per_1k_tokens": round(e/ntok*1000,4),
                       "energy_above_idle_wh": round(max(0.0, e - idle_w*mz.secs/3600.0),6),
                       "mean_power_w": round(mz.mean_power,1), "idle_power_w": round(idle_w,1),
                       "peak_mem_mb": round(peak,1), "perplexity": round(q,3),
                       "max_temp_c": mz.max_temp, "idle_temp_c": round(idle_t,1),
                       "throttle": mz.throttle, "ok": True, "error": None}
                save_row(rec); n_done += 1
                flag = ""
                if rec["throttle"] not in ("none","unavailable"): flag += "  THROTTLED"
                if rec["counter_vs_integrated_pct"] and rec["counter_vs_integrated_pct"] > 10:
                    flag += f"  ENERGY MISMATCH {rec['counter_vs_integrated_pct']}%"
                el = time.perf_counter()-t_start
                print(f"b{block} {cond:5s} r{rep}  {rec['wh_per_1k_tokens']:7.4f} Wh/1k  "
                      f"{rec['tokens_per_s']:6.1f} tok/s  {peak:6.0f} MB  ppl {q:6.2f}"
                      f"{flag}   [{n_done}/{total}]{flag}")
        except Exception as ex:
            print(f"b{block} {cond:5s} FAILED: {type(ex).__name__}: {ex}")
            save_row({"model": CONFIG["model"], "condition": cond, "block": block,
                      "rep": -1, "ok": False, "error": f"{type(ex).__name__}: {ex}"})
        finally:
            release(model, tok)
            time.sleep(2)

print(f"\ndone. results in {RAW}")

# %%
# CELL 6  analysis

import pandas as pd
from scipy import stats

d = pd.read_csv(RAW)
good = d[(d.ok == True)].copy()
thr = good[~good.throttle.isin(["none","unavailable"])]
if len(thr):
    print(f"excluding {len(thr)} throttled run(s); report this exclusion"); 
    good = good[good.throttle.isin(["none","unavailable"])]

if good.empty:
    print("no usable runs")
else:
    print(good.groupby("condition").agg(
        n=("wh_per_1k_tokens","size"),
        wh_per_1k_mean=("wh_per_1k_tokens","mean"),
        wh_per_1k_sd=("wh_per_1k_tokens","std"),
        tok_s=("tokens_per_s","mean"),
        peak_mb=("peak_mem_mb","mean"),
        ppl=("perplexity","mean"),
        max_temp=("max_temp_c","max")).round(4).to_string())

    a = good[good.condition=="fp16"]; b = good[good.condition=="nf4"]
    if len(a) and len(b):
        de = (b.wh_per_1k_tokens.mean()-a.wh_per_1k_tokens.mean())/a.wh_per_1k_tokens.mean()*100
        dm = (b.peak_mem_mb.mean()-a.peak_mem_mb.mean())/a.peak_mem_mb.mean()*100
        dp = (b.perplexity.mean()-a.perplexity.mean())/a.perplexity.mean()*100
        print(f"\n4-bit relative to fp16:")
        print(f"  energy per 1k tokens {de:+.1f}%   (negative = a saving)")
        print(f"  peak memory          {dm:+.1f}%")
        print(f"  perplexity           {dp:+.1f}%   (positive = worse output)")
        if len(a)>1 and len(b)>1:
            t,p = stats.ttest_ind(b.wh_per_1k_tokens, a.wh_per_1k_tokens, equal_var=False)
            print(f"  Welch t-test on energy: p = {p:.4f}")
        else:
            print("  too few replicates for a test")
    good.to_csv(OUT/"summary_runs.csv", index=False)

# %%
# CELL 7  manifest

import datetime
man = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "config": CONFIG,
       "hardware": {"gpu": GPU, "driver": DRIVER, "energy_counter": COUNTER,
                    "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(H)/1000},
       "software": {"python": platform.python_version(), "torch": torch.__version__}}
try:
    import transformers, bitsandbytes
    man["software"]["transformers"] = transformers.__version__
    man["software"]["bitsandbytes"] = bitsandbytes.__version__
except Exception:
    pass
man["counts"] = {"rows": int(len(d)), "successful": int((d.ok==True).sum())}
with open(OUT/"manifest.json","w") as f: json.dump(man, f, indent=2, default=str)
print(json.dumps(man["hardware"], indent=2)); print(f"\nwrote {OUT/'manifest.json'}")
