"""#40 foundation: replace pop-share county disaggregation with REAL intra-BA structure from OEDI-8562.
Within-BA weight w[c,t]: 2016-2023 = real 8562 share (time-varying); else = 8562 climatological share
(month x hour-of-day). upgraded_county[c,t] = BA_anchored[b,t] x w[c,t]  (BA totals preserved exactly).
--diag: quantify pop-share error vs real (fast). --build: write full 1980-2019 upgraded county load."""
import sys, numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "--diag"
HFs = "/data/tell_pred/future/hist_full40_seds"; HF = "/data/tell_pred/future/hist_full40"
QS = "/data/tell_qs/tell_quickstarter_data/outputs"
POP = "/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"
OEDI = "/data/loads_measured/historic_load_hourly_2016_2023_county.h5"

meta = np.load(f"{HF}/meta.npz", allow_pickle=True); NH = int(meta["NH"])
fips = np.array([str(f).zfill(5) for f in meta["fips"]]); f2i = {f: i for i, f in enumerate(fips)}
tidx = pd.date_range(str(meta["t0"]), periods=NH, freq="h")

# county -> single (modal) BA
st = pd.read_csv(f"{QS}/ba_service_territory/ba_service_territory_2019.csv")
st["FIPS"] = st["County_FIPS"].astype(int).astype(str).str.zfill(5)
c2ba = st.groupby("FIPS")["BA_Code"].agg(lambda s: s.value_counts().index[0]).to_dict()
pop = pd.read_csv(POP, dtype={"FIPS": str}); pop["FIPS"] = pop["FIPS"].str.zfill(5)
popm = dict(zip(pop["FIPS"], pop["2020"].astype(float)))
ba_of = np.array([c2ba.get(f, "NONE") for f in fips])
bas = sorted(set(ba_of) - {"NONE"})

# 8562 real county hourly (2016-2023) -> align to our fips + UTC naive
oe = pd.read_hdf(OEDI); oe.index = pd.to_datetime(oe.index)
if oe.index.tz is not None: oe.index = oe.index.tz_convert("UTC").tz_localize(None)
oe = oe.rename(columns={c: str(c)[1:].zfill(5) for c in oe.columns})
common = [f for f in fips if f in oe.columns]
oe = oe[common]
print(f"8562 counties matched {len(common)}/{len(fips)}; span {oe.index.min()}..{oe.index.max()}")

# pop-share within BA vs REAL 8562-share within BA (annual mean)
oe_ann = oe.mean()                                   # mean MW per county (2016-2023)
rows = []
for b in bas:
    cs = [f for f in common if ba_of[f2i[f]] == b]
    if len(cs) < 2: continue
    ps = np.array([popm.get(f, 0) for f in cs]); ps = ps / ps.sum() if ps.sum() > 0 else ps
    rs = oe_ann[cs].values; rs = rs / rs.sum() if rs.sum() > 0 else rs
    for f, p, r in zip(cs, ps, rs):
        rows.append((b, f, p, r))
D = pd.DataFrame(rows, columns=["ba", "fips", "pop_share", "real_share"])
D = D[(D.pop_share > 0) & (D.real_share > 0)]
err = (D.real_share - D.pop_share)
print("\n=== POP-SHARE vs REAL (8562) county-within-BA share ===")
print(f"counties compared: {len(D)} in {D.ba.nunique()} BAs")
print(f"corr(pop_share, real_share): {np.corrcoef(D.pop_share, D.real_share)[0,1]:.3f}")
print(f"median |real-pop|/pop: {np.median(np.abs(err)/D.pop_share)*100:.0f}%   (how far pop-share is off, per county)")
print(f"counties where pop-share off by >2x: {(np.maximum(D.real_share/D.pop_share, D.pop_share/D.real_share)>2).mean()*100:.0f}%")
# worst offenders
D["ratio"] = D.real_share / D.pop_share
w = D.reindex(D.ratio.sort_values().index)
print("most UNDER-weighted by pop-share (real≫pop):", list(w.tail(3).fips), "ratios", [f"{x:.1f}" for x in w.tail(3).ratio])
print("most OVER-weighted by pop-share (real≪pop):", list(w.head(3).fips), "ratios", [f"{x:.2f}" for x in w.head(3).ratio])

if STAGE == "--diag":
    sys.exit(0)

# ---- BUILD: template (month x hod share) + upgraded county load 1980-2019 ----
print("\nbuilding template + upgraded county load...", flush=True)
oe["mo"] = oe.index.month; oe["hod"] = oe.index.hour
# climatological within-BA share by (month, hod): for each BA, share of each county
tmpl = {}   # (ba) -> DataFrame [month*hod x counties] normalized
ba_cols = {b: [f for f in common if ba_of[f2i[f]] == b] for b in bas}
grp = oe.groupby(["mo", "hod"]).mean(numeric_only=True)     # (12*24, ncounty) mean MW
for b in bas:
    cs = ba_cols[b]
    if not cs: continue
    sub = grp[cs]; tot = sub.sum(1).replace(0, np.nan)
    tmpl[b] = (sub.div(tot, axis=0)).fillna(0.0)            # rows sum to 1 over b's counties

anc = np.load(f"{HFs}/county_load_hourly.npy", mmap_mode="r")
out = np.lib.format.open_memmap(f"{HFs}/county_load_hourly_realdist.npy", mode="w+", dtype="float32", shape=(len(fips), NH))
mo = tidx.month.values; hod = tidx.hour.values

# A COUNTY OEDI-8562 DOES NOT NAME MUST NOT VANISH. The redistribution below only reaches counties
# present in that file, and the balancing-authority total was formed over the same subset, so any
# county outside it lost its load entirely rather than keeping the population share it came in with.
# Connecticut is the live case: OEDI-8562 keys the state by its 2022 planning regions, 09110 to
# 09190, while this model keys it by the legacy counties 09001 to 09015, so all eight were written
# as zeros and ISNE lost 22.4% of its load while the national total fell 0.72%. Every other
# balancing authority preserved its total to 1.5e-8, so the contract held everywhere but there.
# Counties outside OEDI now pass through on their incoming allocation, and the real shares
# redistribute only what is left, which preserves each balancing authority's total exactly.
_all_of_ba = {}
for _b in bas:
    _all_of_ba[_b] = [f for f in fips if ba_of[f2i[f]] == _b]
_kept = sorted({f for _b in bas for f in _all_of_ba[_b]} - set(common))
if _kept:
    print("counties not named by OEDI-8562, passed through on their incoming share: %d (%s)"
          % (len(_kept), ", ".join(sorted({f[:2] for f in _kept}))), flush=True)
    for _f in _kept:
        out[f2i[_f], :] = np.asarray(anc[f2i[_f], :], dtype="float32")

for b in bas:
    cs = ba_cols[b]; ci = [f2i[f] for f in cs]
    if not ci: continue
    _all = [f2i[f] for f in _all_of_ba[b]]
    _pass = [i for i in _all if i not in set(ci)]
    ba_total = np.asarray(anc[_all, :]).sum(0)              # (NH,) the WHOLE balancing authority
    ba_load = ba_total - (np.asarray(anc[_pass, :]).sum(0) if _pass else 0.0)
    T = tmpl[b]
    # weight per county per hour from (month,hod) template
    for k, f in enumerate(cs):
        wser = T[f].values.reshape(12, 24)                  # month(1-12)->row idx month-1
        w_t = wser[mo - 1, hod]
        out[f2i[f], :] = (ba_load * w_t).astype("float32")
out.flush()
# validate: subregion totals preserved? US annual preserved?
us_old = np.asarray(anc[:, tidx.year == 2019]).sum() / 1e6
us_new = np.asarray(out[:, tidx.year == 2019]).sum() / 1e6
print(f"US 2019 TWh: pop-share {us_old:.1f} vs realdist {us_new:.1f} (should ~match; BA totals preserved)")
# The contract is checked, not asserted in a comment: every balancing authority must keep its total.
_bad = []
for b in bas:
    _all = [f2i[f] for f in _all_of_ba[b]]
    if not _all: continue
    _yr = np.where(tidx.year == 2019)[0]                  # np.ix_ so the two index arrays do not broadcast
    _o = float(np.asarray(anc[np.ix_(_all, _yr)]).sum())
    _n = float(np.asarray(out[np.ix_(_all, _yr)]).sum())
    if _o > 0 and abs(_n / _o - 1.0) > 1e-6:
        _bad.append((b, _o / 1e6, _n / 1e6, _n / _o - 1.0))
if _bad:
    for b, o, n, r in sorted(_bad, key=lambda x: abs(x[3]), reverse=True)[:8]:
        print("   %-8s 2019 %8.1f -> %8.1f TWh  (%+.3f%%)" % (b, o, n, 100 * r), flush=True)
    raise SystemExit("%d balancing authorities do not preserve their 2019 total; the "
                     "redistribution dropped load instead of moving it" % len(_bad))
print("every balancing authority preserves its 2019 total to 1e-6")
print(f"WROTE {HFs}/county_load_hourly_realdist.npy")
