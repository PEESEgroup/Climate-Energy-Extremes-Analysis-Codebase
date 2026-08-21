"""Post-estimation for the unified attribution: individual removals, physical checks, and the
independent cross-check against the within-storm dose response.

Three things the main run does not do.

1. INDIVIDUAL REMOVAL. The main run reports the joint counterfactual, all five hazards absent, and
   splits it across hazards in proportion to each one's contribution to the linear index. Under a
   multiplicative model the individual removals do not sum to the joint removal, and pretending
   they do would be a presentational choice, not a result. Both are computed and both are reported.

2. MONOTONICITY. A stronger wind cannot break less. If the tropical-cyclone coefficients are not
   increasing in the wind category, the specification is wrong and must be said to be wrong.

3. CROSS-CHECK. The wind gradient here is estimated on the national panel against every other
   county in the country that day. The dose response in Figure 3e is estimated inside each storm
   against the unexposed counties of that same storm. They share no fixed effects and almost no
   identifying variation, so agreement between them is real evidence and disagreement is a warning.
"""
import json, time
import numpy as np, pandas as pd

T_START = time.time()
def log(*a):
    print("[%6.1fs]" % (time.time() - T_START), *a, flush=True)

A = "/data/equity_cost/analysis/attrib"
J = json.load(open("%s/attrib.json" % A))
R = J["results"]
names = list(R.keys())
beta = np.array([R[n]["beta"] for n in names])
se = np.array([R[n]["se"] for n in names])
ORDER = ["tc", "convective", "cold", "heat", "fire"]

# ---------------------------------------------------------------- 2. monotonicity and the leads
log("PHYSICAL CHECKS")
for h in ORDER:
    bins = [n for n in names if n.startswith(h + "|impact|")]
    b = [R[n]["beta"] for n in bins]
    mono = all(b[i] <= b[i + 1] + 1e-9 for i in range(len(b) - 1))
    log("  %-11s impact by intensity %s   monotone %s"
        % (h, "  ".join("%+.3f" % x for x in b), "yes" if mono else "NO"))
    ld = R.get(h + "|lead")
    if ld:
        log("  %-11s lead -14 to -8  %+.4f (se %.4f, z %+.2f)  %s"
            % (h, ld["beta"], ld["se"], ld["z"],
               "placebo clean" if abs(ld["z"]) < 1.96 else "PLACEBO FAILS"))

# ---------------------------------------------------------------- 1. individual removals
# Rebuilding the design costs half a minute and avoids carrying a 1.4 GB array between scripts.
ATTRIB_ECHO = False        # read by attrib.py, suppresses its observed-total line on replay
exec(open("/data/attrib.py").read()
     .split("# ================================================================= design matrix")[0])
BLOCKS = [("lead", -14, -8), ("antic", -7, -1), ("impact", 0, 1), ("restore", 2, 6), ("tail", 7, 14)]
BINNED = {"impact", "restore"}
cols = {}
for h in ORDER:
    Hh = HAZ[h]
    for blk, a_, z_ in BLOCKS:
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
            cols[nm] = acc
log("design rebuilt, %d columns" % len(cols))

y = Y.reshape(-1)
OBSV = float(y.sum())
lin = {h: np.zeros(N) for h in ORDER}
for nm, v in cols.items():
    h, blk = nm.split("|")[0], nm.split("|")[1]
    if blk == "lead":
        continue                                    # the placebo block is a test, not an exposure
    lin[h] += v * R[nm]["beta"]
tot = sum(lin.values())
joint = float((y * (1 - np.exp(-tot))).sum())
log("")
log("ATTRIBUTION, %s customer-hours observed" % format(int(OBSV), ","))
log("  joint removal, all five hazards absent      %.4e = %5.2f%%" % (joint, 100 * joint / OBSV))
ind = {}
for h in ORDER:
    ind[h] = float((y * (1 - np.exp(-lin[h]))).sum())
    log("  removing %-11s alone                  %+.4e = %+5.2f%%" % (h, ind[h], 100 * ind[h] / OBSV))
log("  sum of the individual removals              %.4e = %5.2f%%   (it need not equal the joint)"
    % (sum(ind.values()), 100 * sum(ind.values()) / OBSV))

# county table on the individual removals, which is what the map should carry
CTY = pd.DataFrame({"fips": cf, "denom": den, "observed_cho": Y.sum(1)})
for h in ORDER:
    CTY["att_" + h] = (y * (1 - np.exp(-lin[h]))).reshape(NC, ND).sum(1)
CTY["att_joint"] = (y * (1 - np.exp(-tot))).reshape(NC, ND).sum(1)
CTY["rate_joint"] = CTY.att_joint / CTY.denom
CTY.to_parquet("%s/county_attributable_ppml.parquet" % A, index=False)
q = CTY.rate_joint[CTY.rate_joint > 0]
log("  county rate, hours per customer: median %.2f, 90th %.2f, max %.2f over %d counties with a positive value"
    % (q.median(), q.quantile(.9), q.max(), len(q)))

# ---------------------------------------------------------------- 3. the independent cross-check
DR = json.load(open("/data/equity_cost/analysis/did/did_results_v2.json"))["dose_response"]["binned"]
log("")
log("CROSS-CHECK, tropical cyclone wind gradient, two designs that share no identifying variation")
log("  %-18s %14s %22s" % ("wind", "national panel", "within storm (Fig. 3e)"))
PAIR = [("34 to 50 kt", "34-40"), ("34 to 50 kt", "40-50"), ("50 to 64 kt", "50-64"),
        ("64 to 83 kt", "64-83"), ("83 kt and above", "83+")]
for lab, k in PAIR:
    n = "tc|impact|" + lab
    if n in R:
        log("  %-18s %+8.3f (%.3f) %14.3f (%.3f)"
            % (k, R[n]["beta"], R[n]["se"], DR[k]["coef"], DR[k]["se"]))
json.dump(dict(joint=joint, individual=ind, observed=OBSV,
               joint_share=joint / OBSV, individual_share={h: ind[h] / OBSV for h in ORDER}),
          open("%s/attrib_post.json" % A, "w"), indent=1)
log("wrote %s/attrib_post.json" % A)
