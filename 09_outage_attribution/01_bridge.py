"""Rebuild the outage-channel bridge that Figure 2d draws.

Only the stress indicator changed, because it is defined on the modeled net load, but the builder
script for the published json no longer exists, so the whole thing is rebuilt from the recorded
specification and first validated by reproducing the published numbers on the old panel.

  channel     OE-417 event text classified adequacy -> damage -> ambiguous, in that precedence
  cell        one subregion-day over the overlap years
  stress      the subregion's own p95 (p99) of daily-mean net-load anomaly over the full record
  RR          P(channel event | stress) / P(channel event | no stress)
  interval    moving-block bootstrap over consecutive dates, block 14 days, B = 2000

ENV: PANEL (parquet), TAG
"""
import json, os, re
import numpy as np, pandas as pd
PANEL = os.environ.get("PANEL", "/data/enso/r1_causal/panel_v2.parquet")
TAG = os.environ.get("TAG", "v2")
SPEC = json.load(open("/data/enso/r1_causal/r1_outage_bridge_v2.json"))
TAX = SPEC["taxonomy"]
BOILER = TAX["boilerplate_stripped_before_matching"]
RULES = [("ADEQUACY", TAX["rules_adequacy"]), ("DAMAGE", TAX["rules_damage"]),
         ("AMBIGUOUS", TAX["rules_ambiguous"])]
E = pd.read_csv("/data/equity_cost/oe417_archive_2000_2019.csv", dtype=str)
E["date"] = pd.to_datetime(E["date"], errors="coerce")
txt = (E["event_type"].fillna("") + " || " + E["alert_criteria"].fillna("")).str.lower()
for b in BOILER:
    txt = txt.str.replace(b, " ", regex=False)
def classify(t):
    for name, rules in RULES:
        for pat in rules.values():
            if re.search(pat, t):
                return name
    return "AMBIGUOUS"
E["channel"] = [classify(t) for t in txt]
print("events %d, classified %s" % (len(E), E.channel.value_counts().to_dict()), flush=True)
P = pd.read_parquet(PANEL); P["date"] = pd.to_datetime(P.date)
Y0, Y1 = 2002, 2019
P = P[(P.date.dt.year >= Y0) & (P.date.dt.year <= Y1)]
sub_all = sorted(P.subregion.unique())
thr = {}
full = pd.read_parquet(PANEL); full["date"] = pd.to_datetime(full.date)
for q, lab in [(0.95, "stress95"), (0.99, "stress99")]:
    thr[lab] = full.groupby("subregion").netload_anom_mean.quantile(q).to_dict()
cell = P[["subregion", "date", "netload_anom_mean"]].dropna().copy()
for lab in ("stress95", "stress99"):
    cell[lab] = cell.netload_anom_mean > cell.subregion.map(thr[lab])
ev = E.dropna(subset=["date"])
ev = ev[(ev.date.dt.year >= Y0) & (ev.date.dt.year <= Y1)]
rows = []
for _, r in ev.iterrows():
    for s in str(r["subregions"]).split("|"):
        s = s.strip()
        if s and s != "nan":
            rows.append((s, r["date"], r["channel"]))
M = pd.DataFrame(rows, columns=["subregion", "date", "channel"]).drop_duplicates()
M = M[M.subregion.isin(sub_all)]
print("event-cell rows %d over %d subregions" % (len(M), M.subregion.nunique()), flush=True)
key = ["subregion", "date"]
has = {c: set(map(tuple, M.loc[M.channel == c, key].values)) for c in ("ADEQUACY", "DAMAGE", "AMBIGUOUS")}
cell["k"] = list(zip(cell.subregion, cell.date))
out = {}
rng = np.random.default_rng(7)
dates = np.array(sorted(cell.date.unique()))
for lab in ("stress95", "stress99"):
    out[lab] = {}
    for ch in ("adequacy", "damage"):
        CH = ch.upper()
        amb_only = {k for k in has["AMBIGUOUS"] if k not in has["ADEQUACY"] and k not in has["DAMAGE"]}
        keep = ~cell.k.isin(amb_only)
        c = cell[keep].copy()
        c["ev"] = c.k.isin(has[CH])
        st = c[lab].values; evn = c["ev"].values
        p1 = evn[st].mean(); p0 = evn[~st].mean()
        rr = p1 / p0 if p0 > 0 else np.nan
        # moving-block bootstrap over consecutive calendar dates
        L, B = 14, 2000
        dloc = {d: i for i, d in enumerate(dates)}
        di = c.date.map(dloc).values
        nblk = int(np.ceil(len(dates) / L))
        boot = []
        for b in range(B):
            starts = rng.integers(0, max(len(dates) - L, 1), nblk)
            sel = np.concatenate([np.arange(s, min(s + L, len(dates))) for s in starts])
            m = np.isin(di, sel)
            s2, e2 = st[m], evn[m]
            if s2.sum() == 0 or (~s2).sum() == 0: continue
            q1, q0 = e2[s2].mean(), e2[~s2].mean()
            if q0 > 0: boot.append(q1 / q0)
        boot = np.array(boot)
        out[lab][ch] = dict(rr=float(rr), p_event_given_stress=float(p1),
                            p_event_given_nostress=float(p0), n_stress=int(st.sum()),
                            n_event_on_stress=int(evn[st].sum()), n_nostress=int((~st).sum()),
                            n_event_on_nostress=int(evn[~st].sum()), n_cells_in_sample=int(len(c)),
                            boot_ci_lo=float(np.percentile(boot, 2.5)),
                            boot_ci_hi=float(np.percentile(boot, 97.5)), B=int(len(boot)))
        print("  %-9s %-9s rr %.3f  p1 %.5f p0 %.5f  cells %d  stress %d  events %d/%d"
              % (lab, ch, rr, p1, p0, len(c), st.sum(), evn[st].sum(), evn[~st].sum()), flush=True)
json.dump(out, open("/data/enso/r1_causal/outage_bridge_%s.json" % TAG, "w"), indent=1)
print("wrote outage_bridge_%s.json" % TAG)
