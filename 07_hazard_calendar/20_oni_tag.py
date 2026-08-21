import numpy as np, pandas as pd
rows=[]
for ln in open("/data/enso/oni.data"):
    p=ln.split()
    if len(p)==13 and p[0].isdigit():
        y=int(p[0])
        if 1997<=y<=2019:
            for m in range(12): rows.append((y,m+1,float(p[m+1])))
df=pd.DataFrame(rows,columns=["year","month","oni"])
def phase(o): return "ElNino" if o>=0.5 else ("LaNina" if o<=-0.5 else "neutral")
df["phase"]=df.oni.apply(phase)
# DJF-centered winter (the ENSO peak season) label per winter year (Dec of y-1..Feb of y)
win=[]
for y in range(1998,2020):
    o=df[((df.year==y-1)&(df.month==12))|((df.year==y)&(df.month.isin([1,2])))].oni.mean()
    win.append((y,round(o,2),phase(o)))
w=pd.DataFrame(win,columns=["winter","DJF_ONI","phase"])
print("=== 1997-2019 winter(DJF) ENSO phase ===")
print(w.to_string(index=False))
print("\n=== event inventory (DJF |ONI|>=0.5) ===")
strong=w[w.DJF_ONI.abs()>=1.0]
print("El Nino winters:", list(w[w.phase=='ElNino'].winter))
print("La Nina winters:", list(w[w.phase=='LaNina'].winter))
print("neutral winters:", list(w[w.phase=='neutral'].winter))
print("STRONG (|DJF ONI|>=1.0):", list(zip(strong.winter,strong.DJF_ONI)))
print(f"\nmonthly hours by phase (of 23 yr): {df.phase.value_counts().to_dict()}")
w.to_csv("/data/enso/enso_winter_tags.csv",index=False); df.to_csv("/data/enso/oni_monthly_1997_2019.csv",index=False)
