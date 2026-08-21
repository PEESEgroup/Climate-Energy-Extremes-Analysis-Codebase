"""
The intensity-margin projection, rebuilt on the unified PPML panel.

WHAT CHANGED. The dose object is no longer the matched-cohort binned profile. It is the WHOLE-EVENT
effect of one exposed county-day from `/data/equity_cost/analysis/attrib/attrib.json`: impact
(days 0 to +1) plus restore (+2 to +6) plus tail (+7 to +14), by wind band.

THE OBJECT PROJECTED. Predicted tropical-cyclone outage on an exposed county-day is proportional to
the county's customer count times exp(f(w)) - 1, where f is that whole-event log effect. A
county-day no storm reaches contributes zero rather than one, which is the one arithmetic change
from the earlier version.

THE SHARE. If tropical-cyclone outage scales by R and nothing else moves, a baseline share B of
national customer-hours becomes B*R / (1 + B*(R - 1)). The earlier version used B*R, which ignores
that the denominator grows with the numerator.

MONOTONICITY. The headline uses the profile made monotone by weighted pool-adjacent violators.
That is the physical restriction that more wind cannot mean less damage. The unconstrained profile
is reported beside it. On the 19 August panel fit the restriction does not bind: the 64 to 83 kt
band sits 0.057 log points ABOVE the 50 to 64 kt band, on a standard error of 0.400. The estimated
profile is therefore already increasing, and the constrained and unconstrained projections agree to
every printed digit. The constraint is kept because an earlier fit inverted those two bands, and an
inversion makes county-days that move up a band LOSE outage.

UNCERTAINTY. The nine tropical-cyclone coefficients that enter are drawn together from the two-way
clustered covariance of the panel fit, so the correlation between impact, restore and tail is
carried. B is held at its point estimate.
"""
import json
import numpy as np, pandas as pd

SR = "/data/scratch_r5"
AT = "/data/equity_cost/analysis/attrib"
DZ = json.load(open(f"{AT}/dose_for_fig6.json"))
# The baseline is the share of national outage that would not have happened without tropical
# cyclones, which is the number the results section states. attrib.json's per-hazard entry is a
# proportional split of the joint total and is a different object, 22.64% against 25.33%.
AP = json.load(open(f"{AT}/attrib_post.json"))
B = AP["individual_share"]["tc"]
KT = np.array([b["mid_kt"] for b in DZ["bands"]])
LAB = [b["band"] for b in DZ["bands"]]
CO = np.array([b["total"] for b in DZ["bands"]])
print("baseline tropical-cyclone share of national outage %.2f%%" % (100 * B))

Z = np.load(f"{AT}/attrib_vcov.npz", allow_pickle=True)
NM = list(Z["names"]); VC = Z["vcov"]; BE = Z["beta"]
IDX = [NM.index(t) for b in LAB for t in ("tc|impact|" + b, "tc|restore|" + b)] + [NM.index("tc|tail")]
Vs = VC[np.ix_(IDX, IDX)]; mu = BE[IDX]
A = np.zeros((4, len(IDX)))
for i in range(4):
    A[i, 2 * i] = A[i, 2 * i + 1] = A[i, -1] = 1.0
assert np.allclose(A @ mu, CO, atol=1e-6)
VB = A @ Vs @ A.T; SEB = np.sqrt(np.diag(VB))
L = np.linalg.cholesky(Vs + 1e-12 * np.eye(len(IDX)))
print("band totals: " + "  ".join("%s %.3f+-%.3f (x%.0f)" % (l, c, s, np.exp(c))
                                  for l, c, s in zip(LAB, CO, SEB)))
for i in range(3):
    for j in range(i + 1, 4):
        d = CO[j] - CO[i]; sd = np.sqrt(VB[i, i] + VB[j, j] - 2 * VB[i, j])
        print("   %-16s vs %-16s %+7.3f (se %.3f, z %+5.2f)" % (LAB[j], LAB[i], d, sd, d / sd))
W_ISO = 1.0 / SEB ** 2


def isotonic(t, w=W_ISO):
    val = list(np.asarray(t, float)); wt = list(np.asarray(w, float)); siz = [1] * len(val)
    i = 0
    while i < len(val) - 1:
        if val[i] <= val[i + 1] + 1e-15:
            i += 1; continue
        v_ = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / (wt[i] + wt[i + 1])
        val[i:i + 2] = [v_]; wt[i:i + 2] = [wt[i] + wt[i + 1]]; siz[i:i + 2] = [siz[i] + siz[i + 1]]
        i = max(i - 1, 0)
    return np.array([v_ for v_, n_ in zip(val, siz) for _ in range(n_)])


CO_I = isotonic(CO)
print("monotone profile: " + "  ".join("%.3f (x%.0f)" % (c, np.exp(c)) for c in CO_I))


def dose(w, co):
    out = np.interp(np.asarray(w, float), KT, co, left=co[0], right=co[-1])
    return np.where(np.asarray(w, float) < 34.0, 0.0, out)


F = pd.read_parquet(f"{SR}/tc_flags.parquet")
F["date"] = pd.to_datetime(F.date)
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0][["fips", "denom"]]
F = F.merge(E, on="fips", how="left")
miss = F.denom.isna().mean(); F = F.dropna(subset=["denom"])
print("county-days %s  (%.1f%% dropped for no customer count)" % (format(len(F), ","), 100 * miss))
H = F[F.scen == "historical"][["fips", "date", "wind_kt", "denom"]]
SCEN = sorted(s for s in F.scen.unique() if s != "historical")
# The decomposition used to put every county-day in its HISTORICAL band, which sends the two halves
# of one symmetric process to opposite ends of the waterfall: 1,203 county-days drop below 34 kt and
# leave the 34-50 kt bar as a large negative, while the 1,504 that cross the other way arrive as a
# large positive. Exposure gained and exposure lost are now their own two components, and the bands
# hold only county-days a storm reaches in BOTH periods, where the change really is intensity.
BANDS = [(34, 50, "34-50 kt"), (50, 64, "50-64 kt"), (64, 83, "64-83 kt"), (83, 999, "83+ kt")]


def share(r):
    return 100 * B * r / (1 + B * (r - 1))


OUT = {"baseline_share_pct": 100 * B, "labels": LAB,
       "profile": {"kt": KT.tolist(), "coef": CO_I.tolist(), "coef_raw": CO.tolist(),
                   "se": SEB.tolist()},
       "note": "whole-event PPML dose, monotone by weighted PAVA; share renormalized"}
rng = np.random.default_rng(0)
BND = {}
print("\n%-12s %9s %9s %9s %10s %10s %10s" % ("scenario", "d_kt", "extensive", "R", "dose pp",
                                              "raw pp", "binary pp"))
for sc in SCEN:
    Ff = F[F.scen == sc][["fips", "date", "wind_kt", "denom"]]
    M = H.merge(Ff, on=["fips", "date"], how="outer", suffixes=("_h", "_f"))
    M["denom"] = M.denom_h.fillna(M.denom_f)
    wh = M.wind_kt_h.fillna(0.0).values; wf = M.wind_kt_f.fillna(0.0).values; cu = M.denom.values

    def ratio(co):
        return float((cu * np.expm1(dose(wf, co))).sum() / (cu * np.expm1(dose(wh, co))).sum())
    R, R_raw = ratio(CO_I), ratio(CO)
    dr = np.array([ratio(isotonic(A @ (mu + L @ rng.standard_normal(len(IDX)))))
                   for _ in range(2000)])
    lo, hi = np.percentile(dr, [2.5, 97.5])
    Rb = float((cu * (wf >= 34)).sum() / (cu * (wh >= 34)).sum())
    ext = 100 * (Ff.shape[0] / H.shape[0] - 1)
    OUT[sc] = dict(ratio=R, ratio_raw=R_raw, ratio_lo=float(lo), ratio_hi=float(hi),
                   dose_pp=share(R) - 100 * B, dose_pp_raw=share(R_raw) - 100 * B,
                   dose_pp_lo=share(lo) - 100 * B, dose_pp_hi=share(hi) - 100 * B,
                   binary_ratio=Rb, binary_pp=share(Rb) - 100 * B, extensive_pct=float(ext),
                   dwind_kt=float(np.nanmean(M.wind_kt_f) - np.nanmean(M.wind_kt_h)),
                   new_share_pct=share(R))
    o = OUT[sc]
    print("%-12s %+9.2f %+8.1f%% %9.4f %+10.2f %+10.2f %+10.2f  [%+.2f, %+.2f]"
          % (sc, o["dwind_kt"], ext, R, o["dose_pp"], o["dose_pp_raw"], o["binary_pp"],
             o["dose_pp_lo"], o["dose_pp_hi"]))
    gain = cu * (np.expm1(dose(wf, CO_I)) - np.expm1(dose(wh, CO_I)))
    tot = gain.sum(); PP = share(R) - 100 * B
    both = (wh >= 34) & (wf >= 34)
    parts = [("newly exposed", (wh < 34) & (wf >= 34)), ("exposure lost", (wh >= 34) & (wf < 34))]
    parts += [(nm, both & (wh >= lo_) & (wh < hi_)) for lo_, hi_, nm in BANDS]
    BND[sc] = {nm: dict(share_pct=float(100 * gain[m_].sum() / tot), n=int(m_.sum()),
                        pp=float(PP * gain[m_].sum() / tot),
                        dwind_kt=float(np.mean(wf[m_] - wh[m_])) if m_.any() else 0.0)
               for nm, m_ in parts}
    BND[sc]["already exposed"] = dict(share_pct=float(100 * gain[both].sum() / tot),
                                      n=int(both.sum()), pp=float(PP * gain[both].sum() / tot),
                                      dwind_kt=float(np.mean(wf[both] - wh[both])))
OUT["bands"] = BND
OUT["intensification_share_pct"] = {sc: float(BND[sc]["already exposed"]["share_pct"])
                                    for sc in SCEN}
print("\ncontribution of each component (pp, and % of the total gain):")
for nm in ["newly exposed", "exposure lost"] + [b[2] for b in BANDS] + ["already exposed"]:
    print("   %-16s %s" % (nm, "  ".join("%+6.2fpp (%+6.1f%%)" % (BND[sc][nm]["pp"],
                                                                  BND[sc][nm]["share_pct"])
                                         for sc in SCEN)))
json.dump(OUT, open(f"{SR}/dose_projection_ppml.json", "w"), indent=1, default=float)
print("\nwrote %s/dose_projection_ppml.json" % SR)
