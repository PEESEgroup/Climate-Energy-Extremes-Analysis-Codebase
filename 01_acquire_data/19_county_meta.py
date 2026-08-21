"""county_meta.csv crosswalk: fips, state, state_name, county_name, population, subregion.
Universe = 3108 TGW/GODEEEP counties (county_sub) UNION any FIPS seen in EAGLE-I outages (incl territories).
population = SSP5 county 2020 (the table the TELL load model uses; ~2020 baseline).
subregion = 18 GODEEEP subregions via majority-vote county->subregion (precomputed in netload npz)."""
import numpy as np, pandas as pd
A = "/data/equity_cost/analysis"
NL = "/data/tell_pred/future/hist_full40/subregion_netload_1980_2019.npz"
POP = "/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"
BAST = "/data/tell_qs/tell_quickstarter_data/outputs/ba_service_territory/ba_service_territory_2019.csv"
NAMES = f"{A}/eaglei_fips_names.csv"

ST = {
"01":("AL","Alabama"),"02":("AK","Alaska"),"04":("AZ","Arizona"),"05":("AR","Arkansas"),
"06":("CA","California"),"08":("CO","Colorado"),"09":("CT","Connecticut"),"10":("DE","Delaware"),
"11":("DC","District of Columbia"),"12":("FL","Florida"),"13":("GA","Georgia"),"15":("HI","Hawaii"),
"16":("ID","Idaho"),"17":("IL","Illinois"),"18":("IN","Indiana"),"19":("IA","Iowa"),"20":("KS","Kansas"),
"21":("KY","Kentucky"),"22":("LA","Louisiana"),"23":("ME","Maine"),"24":("MD","Maryland"),
"25":("MA","Massachusetts"),"26":("MI","Michigan"),"27":("MN","Minnesota"),"28":("MS","Mississippi"),
"29":("MO","Missouri"),"30":("MT","Montana"),"31":("NE","Nebraska"),"32":("NV","Nevada"),
"33":("NH","New Hampshire"),"34":("NJ","New Jersey"),"35":("NM","New Mexico"),"36":("NY","New York"),
"37":("NC","North Carolina"),"38":("ND","North Dakota"),"39":("OH","Ohio"),"40":("OK","Oklahoma"),
"41":("OR","Oregon"),"42":("PA","Pennsylvania"),"44":("RI","Rhode Island"),"45":("SC","South Carolina"),
"46":("SD","South Dakota"),"47":("TN","Tennessee"),"48":("TX","Texas"),"49":("UT","Utah"),
"50":("VT","Vermont"),"51":("VA","Virginia"),"53":("WA","Washington"),"54":("WV","West Virginia"),
"55":("WI","Wisconsin"),"56":("WY","Wyoming"),"60":("AS","American Samoa"),"66":("GU","Guam"),
"69":("MP","Northern Mariana Islands"),"72":("PR","Puerto Rico"),"78":("VI","U.S. Virgin Islands"),
}

# subregion map
z = np.load(NL, allow_pickle=True)
cf = np.array([str(x).zfill(5) for x in z["county_fips"]])
cs = np.array(z["county_sub"]).astype(int)
subs = [str(s) for s in z["subregions"]]
fips2sub = {f: (subs[s - 1] if 1 <= s <= len(subs) else "") for f, s in zip(cf, cs)}

# population (SSP5 2020 baseline)
pop = pd.read_csv(POP, dtype={"FIPS": str}); pop["FIPS"] = pop["FIPS"].str.zfill(5)
fips2pop = dict(zip(pop["FIPS"], pd.to_numeric(pop["2020"], errors="coerce")))

# county names: primary EAGLE-I, fallback ba_service_territory
names = {}
try:
    nm = pd.read_csv(NAMES, dtype=str)
    for r in nm.itertuples(index=False):
        names[str(r.fips).zfill(5)] = str(r.county)
except Exception as e:
    print("names load warn:", e)
ba = pd.read_csv(BAST)
ba["FIPS"] = ba["County_FIPS"].astype(int).astype(str).str.zfill(5)
ba_name = dict(zip(ba["FIPS"], ba["County_Name"].astype(str)))

# universe
universe = sorted(set(cf) | set(names.keys()) | set(fips2pop.keys()))
rows = []
for f in universe:
    ab, nm_state = ST.get(f[:2], ("??", "Unknown"))
    cty = names.get(f) or ba_name.get(f) or ""
    rows.append((f, ab, nm_state, cty, fips2pop.get(f, np.nan), fips2sub.get(f, "")))
meta = pd.DataFrame(rows, columns=["fips", "state", "state_name", "county_name", "population", "subregion"])
meta = meta.sort_values("fips").reset_index(drop=True)
meta.to_csv(f"{A}/county_meta.csv", index=False)
print("WROTE", f"{A}/county_meta.csv")
print("n rows:", len(meta))
print("n with subregion:", int((meta.subregion != "").sum()),
      "| n with population:", int(meta.population.notna().sum()),
      "| n with county_name:", int((meta.county_name != "").sum()))
print("subregion counts:")
print(meta[meta.subregion != ""].subregion.value_counts().to_string())
print("US pop (sum, SSP5-2020):", f"{meta.population.sum():,.0f}")
print(meta.head(4).to_string(index=False))
print("territories (state in AS/GU/MP/PR/VI):")
print(meta[meta.state.isin(["AS","GU","MP","PR","VI"])].groupby("state").size().to_string())
