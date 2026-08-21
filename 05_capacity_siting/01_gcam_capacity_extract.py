"""Extract future GENERATION-CAPACITY by technology from the 8 IM3 GCAM-USA BaseX databases.

Query : "Electricity generation by technology (inc solar roofs)"  (units: EJ/yr generation)
Method: GCAM has NO installed-capacity query -> extract GENERATION (EJ), convert EJ->TWh,
        then GW = TWh*1000 / (8760 * CF) using per-technology annual capacity factors.
State->subregion: population-weighted crosswalk (county pop 2020, SSP-matched) using the
        county FIPS->subregion mapping (/tmp/fips_to_subregion_mapping.csv).
Output: /data/gcam_usa/gcam_capacity_by_subregion.csv
        columns: scenario, year, subregion, tech, capacity_GW
"""
import os, sys, subprocess, gcamreader, pandas as pd, numpy as np

MI      = "/data/tellenv/lib/python3.10/site-packages/gcamreader/ModelInterface"
GDIR    = "/data/gcam_usa"
DBROOT  = f"{GDIR}/db_extracted"
FIPSMAP = "/tmp/fips_to_subregion_mapping.csv"
SSP_POP = {"ssp3": "/data/tell_data/sample_forcing_data/sample_population_projections/ssp3_county_population.csv",
           "ssp5": "/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"}
OUTCSV  = f"{GDIR}/gcam_capacity_by_subregion.csv"
QTITLE  = "Electricity generation by technology (inc solar roofs)"
EJ2TWh  = 1e18/3.6e15                       # 1 EJ = 277.778 TWh
YEARS   = [2020,2025,2030,2035,2040,2045,2050]
SCENARIOS = [f"{r}{t}_{s}" for r in ("rcp45","rcp85") for t in ("cooler","hotter") for s in ("ssp3","ssp5")]
STATES  = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
              "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
# postal -> 2-digit FIPS state code
POSTAL2FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22",
    "ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32",
    "NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42",
    "RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
    "WI":"55","WY":"56"}
FIPS2POSTAL = {v:k for k,v in POSTAL2FIPS.items()}

# --- per-detailed-technology -> (clean category, annual capacity factor) -----
def classify(t):
    tl = t.lower()
    if "offshore" in tl:            return ("wind_offshore", 0.45)
    if tl.startswith("wind"):       return ("wind",          0.35)   # incl wind_storage
    if tl.startswith("pv"):         return ("solar",         0.24)   # utility PV (incl PV_storage)
    if tl.startswith("csp"):        return ("solar",         0.30)   # concentrating solar -> solar
    if tl == "rooftop_pv":          return ("rooftop_solar", 0.15)   # (0 at US-state level)
    if tl.startswith("gen_ii") or tl.startswith("gen_iii"): return ("nuclear", 0.90)
    if tl == "hydro":               return ("hydro",         0.40)
    if "hydrogen" in tl:            return ("other",         0.50)
    if "geothermal" in tl:          return ("geothermal",    0.80)
    if tl.startswith("biomass"):    return ("biomass",       0.60)
    if tl.startswith("coal"):       return ("coal",          0.55)
    if tl.startswith("gas"):        return ("gas",           0.45)   # mixed CC/CT (rough)
    if tl.startswith("refined liquids"): return ("oil",      0.10)
    return ("other", 0.50)

# --- build state -> {subregion: weight} crosswalk from county population ------
def build_crosswalk(ssp):
    fm = pd.read_csv(FIPSMAP, dtype={"FIPS": str}); fm["FIPS"] = fm["FIPS"].str.zfill(5)
    pop = pd.read_csv(SSP_POP[ssp]); pop["FIPS"] = pop["FIPS"].astype(str).str.zfill(5)
    pop = pop[["FIPS", "2020"]].rename(columns={"2020": "pop"})
    m = fm.merge(pop, on="FIPS", how="left")
    m["pop"] = m["pop"].fillna(0.0)
    # counties with no pop match -> give tiny weight so they still count (rare)
    m.loc[m["pop"] <= 0, "pop"] = 1.0
    m["STFIPS"] = m["FIPS"].str[:2]
    m["POSTAL"] = m["STFIPS"].map(FIPS2POSTAL)
    g = m.groupby(["POSTAL", "Subregion"], as_index=False)["pop"].sum()
    tot = g.groupby("POSTAL")["pop"].transform("sum")
    g["weight"] = g["pop"] / tot
    return g[["POSTAL", "Subregion", "weight"]]

def ensure_extracted(scen):
    d = f"{DBROOT}/database_{scen}"
    if os.path.isdir(d) and any(f.endswith(".basex") for f in os.listdir(d)):
        return "already"
    z = f"{GDIR}/database_{scen}.zip"
    subprocess.run(["unzip", "-q", "-o", z, "-d", DBROOT], check=True)
    return "unzipped"

def extract_scen(scen, xwalk):
    conn = gcamreader.LocalDBConn(DBROOT, f"database_{scen}", suppress_gabble=True)
    q = [x for x in gcamreader.parse_batch_query(MI+"/Main_queries.xml") if x.title == QTITLE][0]
    df = conn.runQuery(q)
    us = df[df.region.isin(STATES)].copy()
    cat_cf = us.technology.map(classify)
    us["cat"] = cat_cf.str[0]; us["cf"] = cat_cf.str[1]
    us = us[us.Year.isin(YEARS)]
    us["GW"] = (us["value"] * EJ2TWh * 1000.0) / (8760.0 * us["cf"])   # value(EJ)->TWh->GW
    # state x category x year GW
    sg = us.groupby(["region", "cat", "Year"], as_index=False)["GW"].sum()
    # split to subregions by population weight
    merged = sg.merge(xwalk, left_on="region", right_on="POSTAL", how="inner")
    merged["capGW"] = merged["GW"] * merged["weight"]
    out = merged.groupby(["Subregion", "cat", "Year"], as_index=False)["capGW"].sum()
    out.insert(0, "scenario", scen)
    out = out.rename(columns={"Subregion": "subregion", "cat": "tech", "Year": "year", "capGW": "capacity_GW"})
    return out[["scenario", "year", "subregion", "tech", "capacity_GW"]]

def main():
    xwalks = {s: build_crosswalk(s) for s in ("ssp3", "ssp5")}
    parts = []
    for scen in SCENARIOS:
        st = ensure_extracted(scen)
        ssp = "ssp3" if scen.endswith("ssp3") else "ssp5"
        out = extract_scen(scen, xwalks[ssp])
        parts.append(out)
        w = out[out.tech == "wind"].capacity_GW.sum()
        s = out[out.tech == "solar"].capacity_GW.sum()
        print(f"[{scen}] {st:9s} rows={len(out):5d}  sum-over-all-yrs wind={w:7.0f} solar={s:7.0f} GW", flush=True)
    allout = pd.concat(parts, ignore_index=True).sort_values(["scenario","year","subregion","tech"])
    allout["capacity_GW"] = allout["capacity_GW"].round(4)
    allout.to_csv(OUTCSV, index=False)
    print(f"\nWROTE {OUTCSV}  rows={len(allout)}  scenarios={allout.scenario.nunique()} "
          f"subregions={allout.subregion.nunique()} techs={sorted(allout.tech.unique())}")

if __name__ == "__main__":
    main()
