"""#40: REAL observed 18-subregion net-load from EIA-930 (2016-2025). Demand + gen-by-source from the
two PUDL parquets; BA->subregion via build_subregion_hydro alloc (capacity-share for GEN), RTOs split
by MODEL demand-share for DEMAND. net = demand - solar - wind; net_hydro also subtracts hydro."""
import sys, numpy as np, pandas as pd
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import build_subregion_hydro as B
OPS = "/data/hydro/real/eia930_hourly_operations.parquet"
NG = "/data/hydro/real/eia930_hourly_netgen_by_source.parquet"
SUBS = B.SUBS; SUBIDX = B.SUBIDX
alloc = B.build_alloc()                                   # ba -> {subname: w}  (capacity-based; GEN)

# demand alloc: override the 3 RTO splits with MODEL subregion demand-share
anc = np.load("/data/tell_pred/future/hist_full40_seds/subregion_load_hourly.npy", mmap_mode="r")
meta = np.load("/data/tell_pred/future/hist_full40/meta.npz", allow_pickle=True)
tt = pd.date_range(str(meta["t0"]), periods=int(meta["NH"]), freq="h")
m1619 = (tt.year >= 2016) & (tt.year <= 2019)
submean = np.asarray(anc[:, np.where(m1619)[0]]).mean(1)   # (18,) mean MW per subregion
dem_alloc = {ba: dict(w) for ba, w in alloc.items()}
# FIX 2026-07-13: submean/anc (hist_full40_seds) is in FMAP Subregion_Code order, which SWAPS MISO_Central/
# North vs build_subregion_hydro SUBS order -> reading submean[SUBIDX[name]] mis-assigned the MISO N/C demand
# split. Remap submean BY NAME using the anc (FMAP-code) order.
FMAP_ORDER = ['CAISO','ERCOT','FRCC','ISONE','MISO_Central','MISO_North','MISO_South','NYISO',
  'NorthernGrid_East','NorthernGrid_South','NorthernGrid_West','PJM_East','PJM_West','SERTP',
  'SPP_North','SPP_South','WestConnect_North','WestConnect_South']
submean_by_name = {FMAP_ORDER[i]: submean[i] for i in range(18)}
for rto, subs in B.SPLIT.items():
    sh = np.array([submean_by_name[s] for s in subs]); sh = sh / sh.sum()
    dem_alloc[rto] = {s: float(sh[i]) for i, s in enumerate(subs)}

# ---- demand (adjusted->imputed->reported), pivot BA x time ----
d = pd.read_parquet(OPS, columns=["balancing_authority_code_eia", "datetime_utc",
                                  "demand_adjusted_mwh", "demand_imputed_eia_mwh", "demand_reported_mwh"])
d["v"] = d["demand_adjusted_mwh"].fillna(d["demand_imputed_eia_mwh"]).fillna(d["demand_reported_mwh"])
d.loc[d["v"].abs() > 3e5, "v"] = np.nan
demP = d.pivot_table(index="datetime_utc", columns="balancing_authority_code_eia", values="v")

# ---- gen by source ----
g = pd.read_parquet(NG, columns=["balancing_authority_code_eia", "datetime_utc",
                                 "generation_energy_source", "net_generation_adjusted_mwh", "net_generation_reported_mwh"])
g["v"] = g["net_generation_adjusted_mwh"].fillna(g["net_generation_reported_mwh"])
g.loc[g["v"].abs() > 3e5, "v"] = np.nan
def srcpiv(src):
    gs = g[g["generation_energy_source"].isin(src)].groupby(["datetime_utc", "balancing_authority_code_eia"], as_index=False)["v"].sum()
    return gs.pivot(index="datetime_utc", columns="balancing_authority_code_eia", values="v")
solarP = srcpiv(["solar", "solar_wo_integrated_battery_storage", "solar_w_integrated_battery_storage"])
windP = srcpiv(["wind", "wind_wo_integrated_battery_storage", "wind_w_integrated_battery_storage"])
hydroP = srcpiv(["hydro", "hydro_excluding_pumped_storage"])

# common hourly index 2016-2025
idx = pd.date_range("2016-01-01", "2025-12-31 23:00", freq="h")
def agg(P, al):
    P = P.reindex(idx); out = np.zeros((len(idx), 18), "float32")
    for ba in P.columns:
        if ba in al:
            col = P[ba].values.astype("float32")
            for s, w in al[ba].items():
                out[:, SUBIDX[s]] += np.nan_to_num(col) * w
    return out
demand = agg(demP, dem_alloc); solar = agg(solarP, alloc); wind = agg(windP, alloc); hydro = agg(hydroP, alloc)
netload = demand - solar - wind; net_hydro = netload - hydro
np.savez("/data/loads_measured/real_subregion_2016_2025.npz",
         times=idx.astype(str).values, subs=np.array(SUBS), demand=demand, solar=solar, wind=wind,
         hydro=hydro, netload=netload, net_hydro=net_hydro)

# ---- validation: real US demand annual (TWh) vs SEDS ----
yr = idx.year.values; us = demand.sum(1)
seds = pd.read_csv("/data/loads_measured/seds_use_all_phy.csv"); es = seds[seds.MSN == "ESTCP"].set_index("State")
st48 = [s for s in es.index if s not in ("US", "AK", "HI", "X3", "X5")]
print(f"{'year':>4} {'real_demand_TWh':>15} {'SEDS_TWh':>9} {'solar_TWh':>9} {'wind_TWh':>9}")
for y in range(2016, 2025):
    m = yr == y
    print(f"{y:>4} {us[m].sum()/1e6:15.1f} {float(es.loc[st48,str(y)].sum())/1e6:9.1f} {solar[m].sum()/1e6:9.1f} {wind[m].sum()/1e6:9.1f}")
print(f"\nsolar growth 2016->2024: {solar[yr==2024].sum()/max(solar[yr==2016].sum(),1):.1f}x ; wind: {wind[yr==2024].sum()/max(wind[yr==2016].sum(),1):.2f}x")
print("WROTE /data/loads_measured/real_subregion_2016_2025.npz (demand/solar/wind/hydro/netload/net_hydro, 18 sub)")
