"""#40 sectoral layer: decompose each county's REAL total load (8562 foundation) into
res/com/ind/transport via IPF (RAS matrix balancing).
  row margin  = real county annual total (county_load_hourly_realdist.npy, per year)
  col margin  = SEDS state-sector electricity share x that state's county-total (real EIA sector split)
  seed        = res/com/trans ∝ population ; ind ∝ max(eps, county_total - pop_expected(res+com+trans))
                (the 'excess load over population expectation' = the industrial-county signature)
IPF reconciles so BOTH county totals (8562) AND state sector splits (SEDS) hold exactly.
dsgrid EFS .dsg is used only as an independent cross-check + (later) enduse/hourly-shape source.
--diag  : sector split + top industrial counties, no write.  --build : write county_sector_annual.npy
"""
import sys, numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "--diag"
HFs = "/data/tell_pred/future/hist_full40_seds"; HF = "/data/tell_pred/future/hist_full40"
POP = "/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"
YRS = [2016, 2017, 2018, 2019]
SEC = ["res", "com", "ind", "trans"]; MSN = {"res": "ESRCP", "com": "ESCCP", "ind": "ESICP", "trans": "ESACP"}
FA = {"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
"13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
"25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
"36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN",
"48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"}

meta = np.load(f"{HF}/meta.npz", allow_pickle=True); NH = int(meta["NH"])
fips = np.array([str(f).zfill(5) for f in meta["fips"]]); nC = len(fips)
tidx = pd.date_range(str(meta["t0"]), periods=NH, freq="h")
stab = np.array([FA.get(f[:2], "??") for f in fips])
pop = pd.read_csv(POP, dtype={"FIPS": str}); pop["FIPS"] = pop["FIPS"].str.zfill(5)
popm = np.array([dict(zip(pop.FIPS, pop["2020"].astype(float))).get(f, 0.0) for f in fips])

# real county annual totals (MWh->TWh) from 8562-distribution foundation
A = np.load(f"{HFs}/county_load_hourly_realdist.npy", mmap_mode="r")
cty_tot = {y: np.asarray(A[:, tidx.year == y]).sum(1) / 1e6 for y in YRS}   # TWh per county

# SEDS state x sector electricity (million kWh -> TWh)
S = pd.read_csv("/data/loads_measured/seds_use_all_phy.csv")
def seds_state_sector(y):
    out = {}
    for sec, code in MSN.items():
        r = S[S.MSN == code].set_index("State")[str(y)]
        out[sec] = {st: float(r.get(st, 0.0)) / 1e3 for st in set(stab) if st != "??"}
    return out   # out[sec][state] TWh

def ipf(row_t, col_t, seed, iters=200, tol=1e-9):
    X = seed.astype(float).copy(); X[X <= 0] = 1e-9
    for _ in range(iters):
        rs = X.sum(1); X *= (row_t / np.where(rs > 0, rs, 1))[:, None]
        cs = X.sum(0); X *= (col_t / np.where(cs > 0, cs, 1))[None, :]
        if np.max(np.abs(X.sum(1) - row_t)) < tol * row_t.sum(): break
    return X

def build_year(y, verbose=False):
    ss = seds_state_sector(y); ct = cty_tot[y]
    out = np.zeros((nC, 4), float)
    for st in sorted(set(stab)):
        if st == "??": continue
        idx = np.where(stab == st)[0]
        F = ct[idx].sum()
        if F <= 0: continue
        sfrac = np.array([ss[s].get(st, 0.0) for s in SEC]); tot = sfrac.sum()
        if tot <= 0: sfrac = np.array([.38, .36, .26, .002])  # US fallback
        else: sfrac = sfrac / tot
        col_t = sfrac * F                                     # state sector margins (TWh), sum=F
        p = popm[idx]; p = p / p.sum() if p.sum() > 0 else np.ones(len(idx)) / len(idx)
        pop_exp = (col_t[0] + col_t[1] + col_t[3]) * p        # res+com+trans expected by pop
        ind_seed = np.maximum(ct[idx] - pop_exp, p * F * 0.01)  # excess-over-pop = industrial signature
        seed = np.column_stack([p, p, ind_seed, p])
        X = ipf(ct[idx], col_t, seed)
        out[idx] = X
    return out, ss

# ---- diagnostics ----
X19, ss19 = build_year(2019, verbose=True)
us = X19.sum(0); usf = us / us.sum()
print("=== county-sector decomposition (2019, IPF: rows=8562 county totals, cols=SEDS state-sector) ===")
print(f"US sector split TWh : res {us[0]:.0f} com {us[1]:.0f} ind {us[2]:.0f} trans {us[3]:.1f}  (sum {us.sum():.0f})")
print(f"US sector fraction  : res {usf[0]:.1%} com {usf[1]:.1%} ind {usf[2]:.1%} trans {usf[3]:.2%}")
# SEDS US check
seds_us = np.array([sum(ss19[s].values()) for s in SEC]); print(f"SEDS US (target)    : res {seds_us[0]:.0f} com {seds_us[1]:.0f} ind {seds_us[2]:.0f} trans {seds_us[3]:.1f}")
# industrial share per county, top offenders
indsh = X19[:, 2] / np.maximum(X19.sum(1), 1e-9)
o = np.argsort(-indsh)[:12]
print("\ntop-12 industrial-share counties (fips, state, ind%, totalTWh):")
for i in o:
    print(f"  {fips[i]} {stab[i]}  ind {indsh[i]:5.1%}  tot {X19[i].sum():.2f}  (r{X19[i,0]/X19[i].sum():.0%}/c{X19[i,1]/X19[i].sum():.0%}/i{indsh[i]:.0%})")
# state-margin closure check
err = []
for st in sorted(set(stab)):
    if st == "??": continue
    idx = stab == st; F = cty_tot[2019][idx].sum()
    if F <= 0: continue
    sfrac = np.array([ss19[s].get(st, 0.0) for s in SEC]); sfrac = sfrac / sfrac.sum() if sfrac.sum() > 0 else sfrac
    got = X19[idx].sum(0) / X19[idx].sum()
    err.append(np.abs(got - sfrac).sum())
print(f"\nstate sector-split closure (mean L1 err over states): {np.mean(err):.2e} (IPF should ~0)")
print(f"county-total closure: max|rowsum-real| = {np.max(np.abs(X19.sum(1)-cty_tot[2019])):.2e} TWh")

if STAGE == "--build":
    arr = np.zeros((len(YRS), nC, 4), "float32")
    for k, y in enumerate(YRS):
        Xy, _ = build_year(y); arr[k] = Xy
    np.savez(f"{HFs}/county_sector_annual.npz", years=np.array(YRS), fips=fips, states=stab,
             sectors=np.array(SEC), load_twh=arr)  # (nyear, 3108, 4)
    # mean fraction per county (2016-2019) for the future arm to apply differential growth
    frac = arr.mean(0); frac = frac / np.maximum(frac.sum(1, keepdims=True), 1e-9)
    np.save(f"{HFs}/county_sector_frac.npy", frac.astype("float32"))
    print(f"\nWROTE {HFs}/county_sector_annual.npz (load_twh {arr.shape}) + county_sector_frac.npy ({frac.shape})")
