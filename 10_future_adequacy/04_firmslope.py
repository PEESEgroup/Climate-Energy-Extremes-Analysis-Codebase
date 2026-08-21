"""How much firm capacity does a GW of VRE actually buy?

The policy dumbbells and the violins were both pictures of the same thing, the peak net load by
policy variant. This is the question underneath them, which neither answered: the four variants
differ mostly in how much VRE gets built, so regressing the firm requirement on the VRE fleet across
the 32 realisations - with demand held by an SSP fixed effect, because ssp5 has both more VRE and
more load - gives the capacity credit directly.
"""
import numpy as np, pandas as pd

S = pd.read_csv("/data/cerf_out/r4_netload/firm_vs_vre.csv")
for xk in ["vre_peak_gw", "vre_mean_gw"]:
    X = np.c_[np.ones(len(S)), S[xk].values, (S.ssp == "ssp5").astype(float).values]
    b, *_ = np.linalg.lstsq(X, S.firm_gw.values, rcond=None)
    r = S.firm_gw.values - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (r @ r) / (len(S) - 3)))
    print("firm = %+.3f x %s  (s.e. %.3f, t %.1f)   ssp5 %+.1f GW   R2 %.3f"
          % (b[1], xk, se[1], b[1] / se[1], b[2],
             1 - (r @ r) / ((S.firm_gw - S.firm_gw.mean()) ** 2).sum()))
print("\nso 1 GW of VRE peak output displaces %.2f GW of firm capacity: a capacity credit of %.0f%%"
      % (-np.linalg.lstsq(np.c_[np.ones(len(S)), S.vre_peak_gw,
                                (S.ssp == "ssp5").astype(float)],
                          S.firm_gw, rcond=None)[0][1],
         -100 * np.linalg.lstsq(np.c_[np.ones(len(S)), S.vre_peak_gw,
                                      (S.ssp == "ssp5").astype(float)],
                                S.firm_gw, rcond=None)[0][1]))
print("\nby ssp, the OBBBA-IRA contrast:")
for ssp in ["ssp3", "ssp5"]:
    d = S[S.ssp == ssp].groupby("vlab")[["vre_peak_gw", "firm_gw"]].mean()
    dv = d.loc["OBBBA", "vre_peak_gw"] - d.loc["IRA", "vre_peak_gw"]
    df_ = d.loc["OBBBA", "firm_gw"] - d.loc["IRA", "firm_gw"]
    print("  %s: %+.0f GW VRE -> %+.0f GW firm  (%.2f GW firm per GW VRE)"
          % (ssp, dv, df_, df_ / dv))
print("\nrealisations %d ; VRE peak %.0f-%.0f GW ; firm %.0f-%.0f GW"
      % (len(S), S.vre_peak_gw.min(), S.vre_peak_gw.max(), S.firm_gw.min(), S.firm_gw.max()))
