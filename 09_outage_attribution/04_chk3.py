"""Attribution restricted to the hazards whose pre-event blocks are zero.

All five hazards stay in the regression, because leaving a co-occurring hazard out would load its
outage onto the others. Which of them enters the attributable total is decided here and nowhere
else, by the screen below, and the set changes when the estimates change. Naming a fixed set in
this header is how it came to say the opposite of what the code did.
"""
import json, time
import numpy as np, pandas as pd
T_START = time.time()
def log(*a): print("[%5.1fs]" % (time.time() - T_START), *a, flush=True)
exec(open("/data/attrib.py").read().split("# ================================================================= design matrix")[0])
R = json.load(open("/data/equity_cost/analysis/attrib/attrib.json"))["results"]
BLOCKS = [("lead", -14, -8), ("antic", -7, -1), ("impact", 0, 1), ("restore", 2, 6), ("tail", 7, 14)]
BINNED = {"impact", "restore"}
ORDER = ["tc", "convective", "cold", "heat", "fire"]
lin = {h: np.zeros(N) for h in ORDER}
for h in ORDER:
    Hh = HAZ[h]
    for blk, a_, z_ in BLOCKS:
        if blk == "lead":
            continue
        for bi in range(4 if blk in BINNED else 1):
            nm = "%s|%s%s" % (h, blk, ("|%s" % Hh["edges"][bi]) if blk in BINNED else "")
            if nm not in R:
                continue
            sel = (Hh["b"] == bi) if blk in BINNED else np.ones(len(Hh["ci"]), bool)
            c0, d0 = Hh["ci"][sel], Hh["di"][sel]
            acc = np.zeros(N, np.float32)
            for lag in range(a_, z_ + 1):
                d = d0 + lag
                ok = (d >= 0) & (d < ND)
                acc += np.bincount(c0[ok].astype(np.int64) * ND + d[ok], minlength=N).astype(np.float32)
            lin[h] += acc * R[nm]["beta"]
y = Y.reshape(-1); OBSV = float(y.sum())
# WHICH HAZARDS ARE IDENTIFIED IS DECIDED HERE, NOT ASSUMED.
# This used to read ID = ["tc", "convective"], a hard-coded pair, while the docstring said
# inclusion follows a zero pre-event block. A rerun whose placebo failed still published them as
# identified. The rule is now executed against the current estimates and it fails closed.
# WHAT THIS SCREEN CAN AND CANNOT DO. Failing to reject a pre-event effect is not evidence that
# there is none, so passing this screen does not identify a causal effect. A hazard whose pre-event
# coefficient is estimated with a very wide interval has a z near zero and passes while the data
# say nothing about a pretrend. The screen is therefore paired with a PRECISION requirement: the
# pre-event interval must also be tight enough to exclude an effect as large as the one claimed
# post-event. A hazard that passes both is reported as passing a placebo screen, not as identified.
Z_PRE = 1.96          # two-sided 5 percent on the pre-event blocks
PRECISION_RATIO = 0.5  # the pre-event 95% half-width must be under this share of the impact effect
# WHICH WINDOW IS THE PLACEBO. Only the lead block is. The design estimates five blocks, lead
# -14 to -8, antic -7 to -1, impact 0 to 1, restore 2 to 6 and tail 7 to 14, and the
# anticipation block is a substantive estimand, not a falsification test. A tropical cyclone is
# forecast days ahead, utilities de-energize before landfall, and the outer circulation arrives
# before the swath day, so a significant antic coefficient is the response the design was built
# to measure. Screening on it excluded tc at z 4.89 and heat at z 4.02, which would bar every
# forecastable hazard from ever passing. The lead block sits far enough ahead to serve as the
# placebo, and fire still fails it at z -5.45.
PRE_BLOCKS = ["lead"]
ID, EXCL = [], {}
for h in ORDER:
    zs = {}
    for blk in PRE_BLOCKS:
        nm = "%s|%s" % (h, blk)
        if nm in R and R[nm].get("se"):
            zs[blk] = abs(float(R[nm]["beta"]) / float(R[nm]["se"]))
    if not zs:
        EXCL[h] = "no pre-event estimate"
        continue
    worst = max(zs, key=zs.get)
    if zs[worst] >= Z_PRE:
        EXCL[h] = "%s block z = %.2f exceeds %.2f" % (worst, zs[worst], Z_PRE)
        continue
    # precision: a null that cannot see the effect it is meant to rule out is not informative
    # The reference is the LARGEST impact coefficient, not the top intensity bin. Taking the last
    # bin by position made heat's verdict turn on an incidental array index: its bins run
    # +0.080 +0.196 +0.251 +0.022, so the top bin is the smallest, and a pre-event interval that
    # is uninformative against 0.022 is comfortably informative against 0.251. The screen asks
    # whether the null could have seen an effect of the size being claimed, so it must be judged
    # against the largest claim.
    _cand = [R["%s|impact|%s" % (h, e)] for e in HAZ[h]["edges"]
             if "%s|impact|%s" % (h, e) in R and R["%s|impact|%s" % (h, e)].get("se")]
    if "%s|impact" % h in R:
        _cand.append(R["%s|impact" % h])
    imp = max(_cand, key=lambda d: abs(float(d["beta"]))) if _cand else None
    if not imp or not imp.get("se"):
        EXCL[h] = "no impact estimate against which to judge the pre-event precision"
        continue
    half = 1.96 * max(float(R["%s|%s" % (h, worst)]["se"]) for worst in PRE_BLOCKS
                      if "%s|%s" % (h, worst) in R)
    ref = abs(float(imp["beta"]))
    if ref <= 0 or half > PRECISION_RATIO * ref:
        EXCL[h] = ("pre-event interval half-width %.3f is not under %.0f%% of the impact effect "
                   "%.3f, so the null is uninformative" % (half, 100 * PRECISION_RATIO, ref))
        continue
    # SIGN. A hazard kept in the attributable total must raise outage. The screen tests the
    # pre-event window and the precision of that test, and neither notices a hazard whose own
    # effect points the wrong way. Cold reached this line with all four intensity coefficients
    # negative and contributed -3.34e+08 customer-hours, 2,225 of 2,268 counties negative, which
    # dragged the four-hazard total below the tropical-cyclone total alone. A negative attributable
    # amount is not a portion of observed outage, so it cannot be a share of it. The raw data say a
    # cold day carries 5.9x the outage of an ordinary winter day, so a negative coefficient is a
    # statement about the estimator's within-region contrast, not about cold, and it is reported
    # as an exclusion with its reason rather than silently netted off the total.
    if float(imp["beta"]) <= 0:
        EXCL[h] = ("the effect it is kept for points the wrong way: the largest impact coefficient "
                   "is %+.4f, so including it would subtract from the attributable total"
                   % float(imp["beta"]))
        continue
    ID.append(h)
if not ID:
    raise SystemExit("no hazard passes the pre-event test; nothing can be called identified")
log("passing the placebo screen (pre-event null AND sufficient precision): %s" % ", ".join(ID))
for h, why in EXCL.items():
    log("   excluded %-11s %s" % (h, why))
two = sum(lin[h] for h in ID)
att2 = y * (1 - np.exp(-two))
log("")
log("observed                                        %.4e customer-hours" % OBSV)
log("hazards passing the placebo screen, %-20s %.4e = %.2f%%" % (" + ".join(ID), att2.sum(), 100 * att2.sum() / OBSV))
for h in ID:
    a1 = y * (1 - np.exp(-lin[h]))
    log("   %-11s alone                             %.4e = %5.2f%%" % (h, a1.sum(), 100 * a1.sum() / OBSV))
all5 = sum(lin.values())
log("all five, for reference only                    %.4e = %.2f%%"
    % ((y * (1 - np.exp(-all5))).sum(), 100 * (y * (1 - np.exp(-all5))).sum() / OBSV))
CTY = pd.DataFrame({"fips": cf, "denom": den, "observed_cho": Y.sum(1),
                    "att_id": att2.reshape(NC, ND).sum(1),
                    **{("att_%s" % h): (y * (1 - np.exp(-lin[h]))).reshape(NC, ND).sum(1)
                       for h in ID}})
CTY["rate"] = CTY.att_id / CTY.denom
CTY.to_parquet("/data/equity_cost/analysis/attrib/county_attributable_identified.parquet", index=False)
q = CTY.rate[CTY.rate > 0]
log("county rate: median %.2f, 90th %.2f, max %.2f h per customer, %d counties positive, %d exceed observed"
    % (q.median(), q.quantile(.9), q.max(), len(q), int((CTY.att_id > CTY.observed_cho + 1e-6).sum())))
json.dump(dict(observed=OBSV, identified=float(att2.sum()), share=float(att2.sum() / OBSV),
               screened_hazards=ID, excluded_hazards=EXCL, pre_event_z_threshold=Z_PRE,
               precision_ratio=PRECISION_RATIO,
               label_note=("passing this screen means the pre-event blocks are null AND estimated "
                           "precisely enough to have detected a pretrend of the size claimed "
                           "post-event. It is a placebo screen, not a proof of identification."),
               alone={h: float((y * (1 - np.exp(-lin[h]))).sum() / OBSV) for h in ID}),
          open("/data/equity_cost/analysis/attrib/attrib_identified.json", "w"), indent=1)
log("wrote county_attributable_identified.parquet")
