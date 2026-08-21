"""Stage E, future arm: 18-subregion hourly load, wind, solar and net load for all 32 scenarios.

Four climate realizations x two demand pathways x four policy variants. Capacity factors depend on
the climate alone and are computed once per climate on the union fleet; the pathway and the policy
select which units of that union are switched on and at what capacity. Load depends on the climate
and the pathway, through a state-by-year growth ratio applied to the same demand model.

Same guards as the historical arm: the stored capacity factors are already net of plant losses, so
no second haircut is applied, and hours where no plant reports are dropped rather than read as zero.
"""
import numpy as np, pandas as pd, os
from scipy import stats
from scipy.spatial import cKDTree
G = "/data/datasets/grid"; FUT = "/data/tell_pred/future"; GF = "/data/gcam_usa/processed"
CERF = "/data/cerf_out"; V5 = "/data/gen_targets/srgan3d_val/futgen_v5"
OUT = "/data/tell_pred/future/netload_ourchain"; os.makedirs(OUT, exist_ok=True)
CLIM = ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]; SSPS = ["ssp3", "ssp5"]
ARMS = [("nopolicy", "fleet_{c}_{s}.csv"), ("policy", "fleet_policy_{c}_{s}.csv"),
        ("ordonly", "fleet_policy_{c}_{s}_ordonly.csv"), ("obbba", "fleet_obbba_{c}_{s}.csv")]
FIPS2AB = {1:'AL',2:'AK',4:'AZ',5:'AR',6:'CA',8:'CO',9:'CT',10:'DE',11:'DC',12:'FL',13:'GA',15:'HI',16:'ID',17:'IL',18:'IN',19:'IA',20:'KS',21:'KY',22:'LA',23:'ME',24:'MD',25:'MA',26:'MI',27:'MN',28:'MS',29:'MO',30:'MT',31:'NE',32:'NV',33:'NH',34:'NJ',35:'NM',36:'NY',37:'NC',38:'ND',39:'OH',40:'OK',41:'OR',42:'PA',44:'RI',45:'SC',46:'SD',47:'TN',48:'TX',49:'UT',50:'VT',51:'VA',53:'WA',54:'WV',55:'WI',56:'WY'}
lat = np.load(f"{G}/coordinate.npz")["lat"]; lon = np.load(f"{G}/coordinate.npz")["lon"]
sm = np.load(f"{G}/subregion_mask.npz", allow_pickle=True); mask = sm["subregion_mask"]
id2 = dict((int(r[0]), str(r[1])) for r in sm["id_to_subregion"]); NS = 18
names = [id2[i] for i in range(1, NS + 1)]
NAME2ID = {n: i + 1 for i, n in enumerate(names)}
def cell_sub(la, lo):
    ila = np.clip(np.searchsorted(lat, la), 0, len(lat) - 1)
    ilo = np.clip(np.searchsorted(lon, lo), 0, len(lon) - 1)
    return mask[ila, ilo]
g = np.load("/data/tgw_hist/tgw_grid.npz"); XLAT = g["XLAT"].ravel(); XLONG = g["XLONG"].ravel()
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
cfips = np.array([str(x).zfill(5) for x in cm["fips"]]); pf = cm["pair_fips"]; pc = cm["pair_cell"]
csub_cell = cell_sub(XLAT[pc], XLONG[pc]); county_sub = np.zeros(len(cfips), int)
for i in range(len(cfips)):
    v = csub_cell[pf == i]; v = v[v > 0]
    county_sub[i] = int(stats.mode(v, keepdims=False).mode) if len(v) else 0
nz = np.argwhere(mask > 0); nzll = np.column_stack([lat[nz[:, 0]], lon[nz[:, 1]]])
for i in np.where(county_sub == 0)[0]:
    cc = pc[pf == i]
    if len(cc) == 0: continue
    j = np.argmin((nzll[:, 0] - XLAT[cc].mean()) ** 2 + (nzll[:, 1] - XLONG[cc].mean()) ** 2)
    county_sub[i] = int(mask[nz[j, 0], nz[j, 1]])
meta = np.load(f"{FUT}/{CLIM[0]}/meta.npz", allow_pickle=True)
lfips = np.array([str(x).zfill(5) for x in meta["fips"]])
f2s = {f: county_sub[i] for i, f in enumerate(cfips)}
lsub = np.array([f2s.get(f, 0) for f in lfips])
state_ab = np.array([FIPS2AB.get(int(f[:2]), "??") for f in lfips])
lidx = pd.date_range("2030-01-01", "2050-12-31 23:00", freq="h"); lyr = lidx.year.values
YEARS = list(range(2030, 2051))
print("counties %d, unassigned %d" % (len(lfips), int((lsub == 0).sum())), flush=True)

def growth(clim, ssp):
    gg = pd.read_csv(f"{GF}/growth_factor_{clim}_{ssp}.csv", index_col=0); gg.columns = gg.columns.astype(int)
    return pd.DataFrame({st: np.interp(YEARS, gg.columns.values.astype(float), gg.loc[st].values.astype(float))
                         for st in gg.index}, index=YEARS).T

def fleet_gen(fl, tech, tree, cfarr, cap_units=None, st_keys=None, since=2019):
    # since=2019 keeps the 2020 seed fleet: those units carry no retirement year and
    # operate through 2050, so the future system contains them (audit 2026-08-14)
    d = fl[(fl.tech == tech) & (fl.sited_year <= 2050) & (fl.sited_year > since)]
    d = d[(d.retirement_year.isna()) | (d.retirement_year > 2050)]
    d = d[np.isfinite(d.lon) & np.isfinite(d.lat)]
    if not len(d): return np.zeros((NS, cfarr.shape[1]), np.float32), 0, 0.0
    dist, j = tree.query(np.c_[d.lon.values, d.lat.values])
    assert dist.max() < 0.02, "a sited unit is %.3f deg from any unit of the union fleet" % dist.max()
    # the fleet carries its own subregion and it must be used: every offshore unit sits in the
    # ocean, outside the land mask, so a mask lookup silently assigns all 920 of them to nothing
    if "subregion" in d.columns and d.subregion.notna().all():
        sub = np.array([NAME2ID.get(str(x), 0) for x in d.subregion.values])
        miss = sub == 0
        if miss.any():
            sub[miss] = cell_sub(d.lat.values[miss], d.lon.values[miss])
    else:
        sub = cell_sub(d.lat.values, d.lon.values)
    assert (sub > 0).all(), "%d of %d units could not be placed" % ((sub == 0).sum(), len(sub))
    out = np.zeros((NS, cfarr.shape[1]), np.float32)
    capmw = d.capacity_mw.values.astype(float)
    for s in range(1, NS + 1):
        r = np.where(sub == s)[0]
        if len(r):
            out[s - 1] = np.nansum(np.asarray(cfarr[j[r]]) * capmw[r, None], axis=0)
    return out, len(d), float(capmw.sum() / 1e3)

rows = []
for clim in CLIM:
    W = np.load(f"{V5}/fut_wind_cf1h_{clim}.npz", allow_pickle=True)
    S = np.load(f"{V5}/fut_solar_cf1h_{clim}.npz", allow_pickle=True)
    wst = W["stamps"].astype(str); sst = S["stamps"].astype(str)
    wcf = np.asarray(W["cf"]); scf = np.asarray(S["cf"])
    wdead = np.isnan(wcf).all(0); sdead = np.isnan(scf).all(0)
    wtree = cKDTree(np.c_[np.asarray(W["lon"]), np.asarray(W["lat"])])
    stree = cKDTree(np.c_[np.asarray(S["lon"]), np.asarray(S["lat"])])
    lkey = lidx.strftime("%Y%m%d%H").values
    wpos = {k: i for i, k in enumerate(wst)}; spos = {k: i for i, k in enumerate(sst)}
    common = [(i, wpos[k], spos[k]) for i, k in enumerate(lkey)
              if k in wpos and k in spos and not wdead[wpos[k]] and not sdead[spos[k]]]
    li = np.array([c[0] for c in common]); wi = np.array([c[1] for c in common]); si = np.array([c[2] for c in common])
    print("%s: aligned %d h of %d (wind holes %d, solar holes %d)"
          % (clim, len(li), len(lidx), wdead.sum(), sdead.sum()), flush=True)
    # offshore wind is allocated to state mandates, not by economic siting, so it is one fleet per
    # climate and is identical across the policy variants. It was missing from the first pass.
    off = pd.read_csv(f"{CERF}/offshore_fleet_{clim}.csv")
    Og, no_, co = fleet_gen(off, "offshore_wind", wtree, wcf, since=2019)
    print("  offshore fleet: %d units, %.1f GW" % (no_, co), flush=True)
    county = np.load(f"{FUT}/{clim}/county_load_hourly.npy", mmap_mode="r")
    for ssp in SSPS:
        gr = growth(clim, ssp)
        fac = np.ones(len(lidx))
        LD = np.zeros((NS, len(lidx)), np.float32)
        for s in range(1, NS + 1):
            rr = np.where(lsub == s)[0]
            if not len(rr): continue
            for y in YEARS:
                cols = np.where(lyr == y)[0]
                mult = np.array([gr.loc[a, y] if a in gr.index else 1.0 for a in state_ab[rr]])
                LD[s - 1, cols] = (np.asarray(county[rr][:, cols]) * mult[:, None]).sum(0)
        for arm, pat in ARMS:
            fl = pd.read_csv(f"{CERF}/{pat.format(c=clim, s=ssp)}")
            Wg, nw, cw = fleet_gen(fl, "wind", wtree, wcf)
            Sg, ns_, cs = fleet_gen(fl, "solar", stree, scf)
            L = LD[:, li]; Wa = Wg[:, wi]; Sa = Sg[:, si]; Oa = Og[:, wi]
            if arm == "obbba":          # credit termination removes offshore wind entirely:
                Oa = np.zeros_like(Oa)  # the mandates do not survive the loss of the credit
                co_arm, no_arm = 0.0, 0
            else:
                co_arm, no_arm = co, no_
            NET = L - Wa - Sa - Oa
            tag = "%s_%s_%s" % (clim, ssp, arm)
            np.savez(f"{OUT}/netload_{tag}.npz", load=L, wind=Wa, solar=Sa, offshore=Oa, net=NET,
                     variant=arm, scenario="%s_%s" % (clim, ssp), climate=clim,
                     times=np.array([str(t) for t in lidx[li]], dtype=object),
                     subregions=np.array(names, dtype=object))
            rows.append(dict(climate=clim, ssp=ssp, arm=arm, wind_units=nw, wind_GW=cw,
                             solar_units=ns_, solar_GW=cs, offshore_units=no_arm, offshore_GW=co_arm,
                             load_GW=L.sum(0).mean() / 1e3,
                             vre_GW=(Wa + Sa + Oa).sum(0).mean() / 1e3,
                             netpeak_GW=NET.sum(0).max() / 1e3,
                             pen=100 * (Wa + Sa + Oa).sum() / L.sum()))
            print("  %-34s wind %5d %6.1f GW | offshore %6.1f GW | solar %5d %6.1f GW | load %5.0f | net peak %5.0f | pen %4.1f%%"
                  % (tag, nw, cw, co_arm, ns_, cs, rows[-1]["load_GW"], rows[-1]["netpeak_GW"], rows[-1]["pen"]), flush=True)
    del wcf, scf, W, S
pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
print("wrote %s/summary.csv and 32 scenario files" % OUT)
