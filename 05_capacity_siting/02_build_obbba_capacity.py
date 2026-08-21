"""
Build the OBBBA capacity pathway from the IRA-era GCAM canonical-8 baseline.

Mechanism (STEP 2): OBBBA (H.R.1, signed 2025-07) terminates the tech-neutral PTC(45Y)/ITC(48E)
for wind & solar (begin-construction ~12mo of enactment or placed-in-service by 2027-12-31) + FEOC.
Net effect = fewer NEW wind/solar additions AFTER the ~2027 credit cliff. We scale the POSITIVE new
additions (level[Y]-level[Y-5]) in each post-cliff period by a national retention factor; pre-2027
pipeline is locked; net retirements (delta<0) unchanged.

=== EVIDENCE-BASED RETENTION (2026-07-23 literature synthesis; replaces earlier hand-set 0.80 solar) ===
Retention = fraction of IRA additions that survive. RED = 1 - retention.
  WIND  retention 0.47 (RED 0.53)  -- KEPT. Anchored to MIT CEEPR "Glass Half Full" (Bermel 2026):
        ~47% of IRA onshore-wind CAPACITY preserved 2025-35 post-OBBBA vs IRA. Bracketed by
        REPEAT 0.33 (8.5/24.5 GW/yr) low and MIT gen-preservation 0.52 / EIA-AEO2026 / WoodMac-ACP high.
        Mechanism: removing the ~$27.5-30/MWh 45Y PTC raises wind LCOE +37% (mid) and the PTC is worth
        more per invested $ (CBO 36% vs 30% ITC); wind also loses 45X + longer lead-times vs the 2027 cliff.
  SOLAR retention 0.72 (RED 0.28)  -- REVISED DOWN from 0.80. MIT's 0.80 is a 2025-35 AVERAGE floor that
        includes the pre-cliff rush; REPEAT system-optimization puts solar steady-state at 0.52-0.66 and
        Rhodium's combined 57-62% cut implies solar cannot stay as insulated as 0.80 once wind+solar+storage
        substitution is co-optimized. Central 0.72; range 0.55-0.83 (0.80 retained as HIGH case).
  FEOC embedded in the anchors (not a separate multiplier); qualitatively squeezes solar/battery more,
        partially offsetting wind's larger credit-value loss (supports wind-high / solar-low).

Period ramp (fraction of the full cut applied to each 5-yr period's additions) -- SHARPER than before:
  the statutory cliff is a hard end-2027 in-service deadline; the safe-harbored pipeline drains by ~2030-32
  (BNEF: installs -41% after 2027). So push the 2030 period (2026-30 additions, mostly post-cliff) to 0.70
  (was 0.50) and reach full cut by 2035 (nearest GCAM 5-yr step to the spec's 2032 convergence point).
  ->2025: 0.0 (locked)   ->2030: 0.70   ->2035..2050: 1.0 (full)

Region modifier -- WEST FEDERAL-LAND extra haircut: EIA implemented OBBBA Sec 50302/50303 by ELIMINATING
  federal lands from the wind & solar supply curves, and Interior permitting puts ~44 GW of solar at risk;
  both concentrate in the federal-land-heavy interior West. Apply an ADDITIONAL 0.075 reduction (ramped) on
  post-2027 wind AND solar positive additions in NorthernGrid_* and WestConnect_* subregions. Offshore wind
  left UNCHANGED (state OSW mandates persist; sited to mandates by our offshore module, not by CERF economics).
  All non-VRE unchanged.
"""
import pandas as pd, numpy as np
IRA="/data/gcam_usa/gcam_capacity_by_subregion.csv"
OUT="/data/gcam_usa/gcam_capacity_obbba.csv"
YEARS=[2020,2025,2030,2035,2040,2045,2050]
RED={"wind":0.53,"solar":0.28}                     # 1 - retention;  wind 0.47 / solar 0.72
RAMP={2020:0.0,2025:0.0,2030:0.70,2035:1.0,2040:1.0,2045:1.0,2050:1.0}
WEST_FED={"NorthernGrid_East","NorthernGrid_South","NorthernGrid_West",
          "WestConnect_North","WestConnect_South"}  # federal-land-heavy interior West
WEST_EXTRA=0.075                                    # additional reduction on post-2027 VRE additions there

g=pd.read_csv(IRA)
keep=g[~g.tech.isin(["wind","solar"])].copy()            # unchanged techs (incl wind_offshore)
recs=[]
for scen in sorted(g.scenario.unique()):
  for tech in ["wind","solar"]:
    sub=g[(g.scenario==scen)&(g.tech==tech)]
    for subreg in sorted(sub.subregion.unique()):
      west = subreg in WEST_FED
      s=sub[sub.subregion==subreg].set_index("year")["capacity_GW"].reindex(YEARS)
      vals={}
      vals[2020]=float(s[2020]) if not pd.isna(s[2020]) else 0.0
      vals[2025]=float(s[2025]) if not pd.isna(s[2025]) else vals[2020]
      for Y in [2030,2035,2040,2045,2050]:
        Y0=Y-5
        iY =float(s[Y])  if not pd.isna(s[Y])  else vals[Y0]
        iY0=float(s[Y0]) if not pd.isna(s[Y0]) else iY
        delta=iY-iY0
        if delta>0:
          nat=1.0-RED[tech]*RAMP[Y]                 # national credit-loss retention
          wst=(1.0-WEST_EXTRA*RAMP[Y]) if west else 1.0   # extra federal-land haircut (ramped)
          delta_o=delta*nat*wst
        else:
          delta_o=delta                             # retirements unchanged
        vals[Y]=vals[Y0]+delta_o
      for Y in YEARS:
        recs.append(dict(scenario=scen,year=Y,subregion=subreg,tech=tech,capacity_GW=round(vals[Y],4)))
obbba=pd.concat([keep,pd.DataFrame(recs)],ignore_index=True)
obbba=obbba.sort_values(["scenario","year","subregion","tech"]).reset_index(drop=True)
obbba.to_csv(OUT,index=False)
print("wrote",OUT,"rows",len(obbba))

# ---- sanity: national wind/solar 2050 IRA vs OBBBA ----
def natl(df):
    d=df[df.tech.isin(["wind","solar"])]
    return d.groupby(["scenario","year","tech"]).capacity_GW.sum().unstack("tech")
A=natl(g); B=natl(obbba)
print("\n=== 2050 national wind/solar  IRA -> OBBBA (GW) and % cut ===")
tw=ts=bw=bs=0.0
for scen in sorted(g.scenario.unique()):
    aw=A.loc[(scen,2050),"wind"]; b_w=B.loc[(scen,2050),"wind"]
    as_=A.loc[(scen,2050),"solar"]; b_s=B.loc[(scen,2050),"solar"]
    tw+=aw; ts+=as_; bw+=b_w; bs+=b_s
    print(f"{scen:22s} wind {aw:6.1f}->{b_w:6.1f} ({100*(1-b_w/aw):4.1f}% cut)   solar {as_:6.1f}->{b_s:6.1f} ({100*(1-b_s/as_):4.1f}% cut)")
n=len(g.scenario.unique())
print(f"\nMEAN/8 2050: wind {tw/n:6.1f}->{bw/n:6.1f} ({100*(1-bw/tw):4.1f}% cut)   solar {ts/n:6.1f}->{bs/n:6.1f} ({100*(1-bs/ts):4.1f}% cut)")
