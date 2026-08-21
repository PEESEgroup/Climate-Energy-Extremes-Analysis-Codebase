"""#37 core: produce the SEDS-anchored REAL historical load (1980-2019).
anchored(state,y) = model(state,2019) * SEDS_ESTCP(state,y)/SEDS_ESTCP(state,2019)  -> real growth,
preserves the EIA-930-calibrated 2019 demand level + the TELL hourly shape within each year.
Applies per-county (by state), writes county + 18-subregion hourly + annual summary + scale table."""
import numpy as np, pandas as pd, os

HF = "/data/tell_pred/future/hist_full40"
OUT = "/data/tell_pred/future/hist_full40_seds"; os.makedirs(OUT, exist_ok=True)
FMAP = "/data/datasets/grid/fips_to_subregion_mapping.csv"   # was /tmp, which had no producer
SEDS = "/data/loads_measured/seds_use_all_phy.csv"
FIPS2ST = {"01":"AL","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
 "13":"GA","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
 "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ",
 "35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC",
 "46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"}

meta = np.load(f"{HF}/meta.npz", allow_pickle=True)
fips = meta["fips"].astype(str)
NH = int(meta["NH"]); tidx = pd.date_range(str(meta["t0"]), periods=NH, freq="h")
yr = np.asarray(tidx.year); state_of = np.array([f.zfill(5)[:2] for f in fips])
cl = np.load(f"{HF}/county_load_hourly.npy", mmap_mode="r")

# SEDS ESTCP state x year (GWh)
s = pd.read_csv(SEDS); es = s[s["MSN"] == "ESTCP"].set_index("State")

# model annual per state (GWh)
mstate = {}
for y in range(1980, 2020):
    cols = np.where(yr == y)[0]; ce = np.asarray(cl[:, cols]).sum(1) / 1000.0
    for st in set(FIPS2ST.values()):
        fst = [k for k, v in FIPS2ST.items() if v == st][0]
        mstate.setdefault(st, {})[y] = float(ce[state_of == fst].sum())

# scale[state][year] = anchored/model = (model_2019/model_y) * (SEDS_y/SEDS_2019)
scale = {}
for st in mstate:
    m19 = mstate[st][2019]; sd19 = float(es.loc[st, "2019"])
    scale[st] = {y: (m19 / mstate[st][y]) * (float(es.loc[st, str(y)]) / sd19) for y in range(1980, 2020)}
pd.DataFrame(scale).T.to_csv(f"{OUT}/seds_scale_state_year.csv")

# per-county scale vector by year, apply -> write county memmap
out = np.lib.format.open_memmap(f"{OUT}/county_load_hourly.npy", mode="w+", dtype="float32", shape=(cl.shape[0], NH))
cty_st = np.array([FIPS2ST.get(st, None) for st in state_of])
for y in range(1980, 2020):
    cols = np.where(yr == y)[0]
    sv = np.array([scale[st][y] if st in scale else 1.0 for st in cty_st], "float32")
    out[:, cols] = np.asarray(cl[:, cols]) * sv[:, None]
    if y % 10 == 0: print(f"  applied {y}", flush=True)
out.flush()

# 18-subregion aggregate (sum of counties)
fm = pd.read_csv(FMAP); fm["FIPS"] = fm["FIPS"].astype(str).str.zfill(5)
f2sub = dict(zip(fm["FIPS"], fm["Subregion_Code"]))
subcode = np.array([f2sub.get(f.zfill(5), 0) for f in fips])
sub = np.zeros((18, NH), "float32")
for sc in range(1, 19):
    idx = np.where(subcode == sc)[0]
    if idx.size:
        for i0 in range(0, NH, 50000):
            sub[sc - 1, i0:i0+50000] = np.asarray(out[idx, i0:i0+50000]).sum(0)
np.save(f"{OUT}/subregion_load_hourly.npy", sub)

# annual summary + validation
rows = []
for y in range(1980, 2020):
    cols = np.where(yr == y)[0]
    twh = float(np.asarray(out[:, cols]).sum()) / 1e6
    pk = float(np.asarray(out[:, cols]).sum(0).max()) / 1000.0
    seds_y = float(es.loc[[st for st in mstate], str(y)].astype(float).sum()) / 1000.0
    anch_y = sum(mstate[st][2019] * float(es.loc[st, str(y)]) / float(es.loc[st, "2019"]) for st in mstate) / 1000.0
    rows.append((y, round(twh, 1), round(pk, 1), round(anch_y, 1), round(seds_y, 1)))
df = pd.DataFrame(rows, columns=["year", "anchored_TWh", "peakGW", "target_anchored_TWh", "SEDS_sales_TWh"])
df.to_csv(f"{OUT}/annual_summary_seds.csv", index=False)
import shutil; shutil.copy(f"{HF}/meta.npz", f"{OUT}/meta.npz")
print(df.to_string(index=False))
print(f"\ngrowth 1980->2019 anchored: {df.anchored_TWh.iloc[-1]/df.anchored_TWh.iloc[0]:.3f}x (was 0.999x flat)")
print(f"WROTE {OUT}/ county_load_hourly.npy + subregion_load_hourly.npy + annual_summary_seds.csv + scale table")
