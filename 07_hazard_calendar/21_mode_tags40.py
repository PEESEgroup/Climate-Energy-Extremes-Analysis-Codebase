"""Build monthly + seasonal circulation-mode index table for 1997-2019: ONI, PNA, NAO, AO (observed NOAA).
Blocking + storm-tracks need Z500 (flagged separately). Saves /data/enso/mode_tags_monthly_1980.csv."""
import numpy as np, pandas as pd
def parse_oni(p):
    rows=[]
    for ln in open(p):
        q=ln.split()
        if len(q)==13 and q[0].isdigit():
            y=int(q[0])
            for m in range(12): rows.append((y,m+1,float(q[m+1])))
    return pd.DataFrame(rows,columns=["year","month","ONI"])
def parse_ym(p,name):
    rows=[]
    for ln in open(p):
        q=ln.split()
        if len(q)>=3 and q[0].isdigit() and q[1].isdigit():
            rows.append((int(q[0]),int(q[1]),float(q[2])))
    return pd.DataFrame(rows,columns=["year","month",name])
d=parse_oni("/data/enso/oni.data")
for f,n in [("pna","PNA"),("nao","NAO"),("ao","AO")]:
    d=d.merge(parse_ym(f"/data/enso/{f}.txt",n),on=["year","month"],how="left")
d=d[(d.year>=1980)&(d.year<=2019)].reset_index(drop=True)
d.to_csv("/data/enso/mode_tags_monthly_1980.csv",index=False)
print("monthly modes 1997-2019:", d.shape, "| cols:", list(d.columns))
print(d.head(3).to_string(index=False))
print("\ncoverage (non-null):", {c:int(d[c].notna().sum()) for c in ["ONI","PNA","NAO","AO"]})
# quick cross-check: does ONI El Nino winter tend to +PNA? (known teleconnection)
w=d[d.month.isin([12,1,2])]
en=w[w.ONI>=0.5]; ln=w[w.ONI<=-0.5]
print(f"\nDJF PNA: El Nino mean {en.PNA.mean():.2f} vs La Nina mean {ln.PNA.mean():.2f} (expect EN>LN, +PNA in El Nino)")
print(f"DJF NAO: El Nino mean {en.NAO.mean():.2f} vs La Nina mean {ln.NAO.mean():.2f}")
