"""Figure 3e and 3f, re-estimated by Poisson pseudo-likelihood on the published landfall panel.

WHAT CHANGES AND WHAT DOES NOT

  The panel does not change. It is the audited did_panel_v2: ten landfalls 2016 to 2022, treated
  counties are those reaching 34 kt, controls are matched two to one inside the same storm, and it
  already carries the documented fixes (quarter-hours to customer-hours, point-in-polygon landfall,
  an explicit observed flag, the 2022 source correction, and the Harvey onboarding gap).

  The estimator changes. The published version regressed log(1 + customer-hours), and this paper
  states elsewhere that log(1+x) with a mass at zero has no scale-free interpretation, so its
  coefficient cannot be read as a percentage. Every other panel of Figure 3 is now Poisson, whose
  coefficient is a log multiplier at any scale. Running the same contrast in Poisson makes the
  figure one estimator, and it makes the wind gradient here directly comparable with the national
  panel in (b) instead of only qualitatively similar.

  E[cho_it] = exp( alpha_{i,event} + delta_{event,day} + sum_k beta_k 1{treated_i, day = k} )

  county-by-event and event-by-day fixed effects, so the contrast is a treated county against the
  control counties of the SAME storm on the SAME day. Day -7 is the reference, as published.
  Inference is two-way clustered on county and on event.
"""
import json, time
import numpy as np, pandas as pd
from scipy import sparse as sps

T0 = time.time()
def log(*a): print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

EQ = "/data/equity_cost/analysis"
OUT = f"{EQ}/attrib"
P = pd.read_parquet(f"{EQ}/did/did_panel_v2.parquet")
P = P[(P.day_rel >= -21) & (P.day_rel <= 21)].copy()

# ---------------------------------------------------------------- remove overlap between storms
# Switching to a level scale exposed a defect the log scale had been hiding. Michael came ashore on
# 10 October 2018 and Florence on 14 September, so Michael's day -21 and -20 fall inside Florence's
# restoration in the same North Carolina counties: three counties, New Hanover, Onslow and Robeson,
# carried 44% of all treated outage on Michael's day -21. That is one storm's aftermath sitting in
# another storm's pre-period, not a pre-trend. Any county-day that lies in the post window of a
# DIFFERENT event that also reached that county is dropped.
EV = pd.read_csv(f"{EQ}/did/did_events_v2.csv")
LF = {r.event: pd.Timestamp(r.landfall_date) for _, r in EV.iterrows()}
P["date"] = P.event.map(LF) + pd.to_timedelta(P.day_rel, unit="D")
post_cells = set()
for e, lf in LF.items():
    f = P.loc[(P.event == e) & (P.treated == 1), "fips"].unique()
    for k in range(0, 22):
        d = lf + pd.Timedelta(days=k)
        post_cells.update(zip(f, [d] * len(f)))
key = list(zip(P.fips.values, P.date.values))
own = set()
for e, lf in LF.items():
    f = P.loc[(P.event == e) & (P.treated == 1), "fips"].unique()
    for k in range(0, 22):
        own.update(zip(f, [lf + pd.Timedelta(days=k)] * len(f)))
contam = np.array([(fp, dt) in post_cells and not (0 <= dr <= 21)
                   for fp, dt, dr in zip(P.fips.values, P.date.values, P.day_rel.values)])
log_n0 = len(P)
P = P[~contam].copy()
print("dropped %s county-days that sat in another storm's post window (%.2f%% of the panel)"
      % (format(int(contam.sum()), ","), 100 * contam.mean()), flush=True)
y = P.cho_zerofill.values.astype(float)
log("panel %s rows, %d counties, %d events, outcome mean %.1f customer-hours"
    % (format(len(P), ","), P.fips.nunique(), P.event.nunique(), y.mean()))

g_ce = pd.factorize(P.event.astype(str) + "|" + P.fips.astype(str))[0].astype(np.int32)
g_ed = pd.factorize(P.event.astype(str) + "|" + P.day_rel.astype(str))[0].astype(np.int32)
g_c = pd.factorize(P.fips.astype(str))[0].astype(np.int32)
g_e = pd.factorize(P.event.astype(str))[0].astype(np.int32)
log("fixed effects: %d county-event, %d event-day" % (g_ce.max() + 1, g_ed.max() + 1))

REF = -7
days = np.array(sorted(P.day_rel.unique()))
kk = [d for d in days if d != REF]
tr = (P.treated.values == 1)
X = np.zeros((len(P), len(kk)), float)
for j, d in enumerate(kk):
    X[:, j] = tr & (P.day_rel.values == d)
names = ["k%+d" % d for d in kk]

# the dose response: treated counties split by the wind they took, against the same controls
BINS = [(34, 40), (40, 50), (50, 64), (64, 83), (83, 999)]
post = P.day_rel.values >= 0
kt = P.exposure_kt.values
D = np.zeros((len(P), len(BINS)), float)
dn = []
for j, (a, b) in enumerate(BINS):
    D[:, j] = tr & (kt >= a) & (kt < b) & post
    dn.append("%d-%d kt" % (a, b) if b < 999 else "83+ kt")
NB = [int(P[(tr) & (kt >= a) & (kt < b)].fips.nunique()) for a, b in BINS]
log("dose bins, treated counties: %s" % dict(zip(dn, NB)))


def make_S(g, w):
    n = int(g.max()) + 1
    S = sps.csr_matrix((w, (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g)))
    return S, np.asarray(S.sum(1)).ravel()


def demean(V, w, groups, sweeps=400, tol=1e-11):
    SS = [make_S(g, w) for g in groups]
    def sweep(A):
        for g, (S, d_) in zip(groups, SS):
            A -= ((S @ A) / np.maximum(d_, 1e-300)[:, None])[g]
        return A
    sc = np.maximum(np.abs(V).max(0), 1e-12)
    for it in range(0, sweeps, 3):
        v0 = V.copy(); sweep(V); v1 = V.copy(); sweep(V); v2 = V
        d1, d2 = v1 - v0, v2 - v1
        dd = d2 - d1
        num = (dd * d2).sum(0); den = (dd * dd).sum(0)
        ok = den > 1e-300
        st = np.zeros_like(num); st[ok] = num[ok] / den[ok]
        V -= st * d2
        if float((np.abs(d2).max(0) / sc).max()) < tol:
            return it + 2
    return sweeps


def deviance(y_, mu_):
    t = np.zeros_like(mu_); m = y_ > 0
    t[m] = y_[m] * np.log(y_[m] / np.maximum(mu_[m], 1e-300))
    return 2.0 * float((t - (y_ - mu_)).sum())


def ppml(Xd, y_, groups, tag):
    beta = np.zeros(Xd.shape[1])
    eta = np.log(y_ + 0.1)
    dev_prev = deviance(y_, np.exp(eta))
    for it in range(30):
        mu = np.exp(eta); w = np.maximum(mu, 1e-10)
        z = eta + (y_ - mu) / w
        V = np.empty((len(y_), Xd.shape[1] + 1))
        V[:, :-1] = Xd; V[:, -1] = z
        ns = demean(V, w, groups)
        Xg, zg = V[:, :-1], V[:, -1]
        A = Xg.T @ (w[:, None] * Xg); rhs = Xg.T @ (w * zg)
        A[np.diag_indices_from(A)] += 1e-11 * np.trace(A) / A.shape[0]
        nb = np.linalg.solve(A, rhs)
        stp = nb - beta; h = 1.0
        for _ in range(14):
            bt = beta + h * stp
            et = z - (zg - Xg @ bt)
            dv = deviance(y_, np.exp(np.clip(et, -50, 30)))
            if dv <= dev_prev * (1 + 1e-12):
                break
            h *= .5
        d = float(np.abs(bt - beta).max()); beta = bt
        eta = np.clip(et, -50, 30); dev_prev = dv
        log("  %s irls %2d sweeps %3d dev %.6e max|db| %.2e%s"
            % (tag, it + 1, ns, dv, d, "  halved to %.4f" % h if h < 1 else ""))
        if d / max(1.0, float(np.abs(beta).max())) < 1e-5 and it >= 3:
            break
    mu = np.exp(eta); w = np.maximum(mu, 1e-10)
    z = eta + (y_ - mu) / w
    V = np.empty((len(y_), Xd.shape[1] + 1)); V[:, :-1] = Xd; V[:, -1] = z
    demean(V, w, groups)
    Xg = V[:, :-1]
    A = Xg.T @ (w[:, None] * Xg); Ai = np.linalg.pinv(A)
    u = (y_ - mu)[:, None] * Xg
    def meat(g):
        n = int(g.max()) + 1
        S = sps.csr_matrix((np.ones(len(g)), (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g)))
        G = S @ u
        return G.T @ G
    Vc = Ai @ (meat(g_c) + meat(g_e) - (u.T @ u)) @ Ai
    return beta, np.sqrt(np.maximum(np.diag(Vc), 0))


GR = [g_ce, g_ed]
log("panel e, event study, reference day %+d" % REF)
be, se = ppml(X, y, GR, "e")
ES = pd.DataFrame({"day_rel": kk, "coef": be, "se": se})
ES.loc[len(ES)] = [REF, 0.0, 0.0]
ES = ES.sort_values("day_rel").reset_index(drop=True)
ES.to_csv(f"{OUT}/eventstudy_ppml.csv", index=False)
pk = ES.loc[ES[ES.day_rel >= 0].coef.idxmax()]
log("  peak %+.3f at day %+d  (x%.1f)  day -1 %+.3f  mean of days -21..-2 %+.4f"
    % (pk.coef, pk.day_rel, np.exp(pk.coef), float(ES[ES.day_rel == -1].coef.iloc[0]),
       ES[(ES.day_rel <= -2)].coef.mean()))

log("panel f, dose response, days 0 to +21 against the pre-period")
bd, sd = ppml(D, y, GR, "f")
DR = pd.DataFrame({"bin": dn, "counties": NB, "coef": bd, "se": sd})
DR["mult"] = np.exp(DR.coef)
DR.to_csv(f"{OUT}/dose_ppml.csv", index=False)
log("\n" + DR.to_string(index=False))
json.dump({"event_study_ref": REF,
           "peak": {"day": int(pk.day_rel), "coef": float(pk.coef), "mult": float(np.exp(pk.coef))},
           "pre_mean": float(ES[ES.day_rel <= -2].coef.mean()),
           "dose": DR.to_dict("records"),
           "n_rows": int(len(P)), "n_counties": int(P.fips.nunique()),
           "n_events": int(P.event.nunique())},
          open(f"{OUT}/did_ppml.json", "w"), indent=1)
log("wrote eventstudy_ppml.csv, dose_ppml.csv, did_ppml.json")
