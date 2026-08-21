"""The one place the historical net-load product is named.

WHY THIS FILE EXISTS. 03_stageE.py used to write into hist_full40 and was changed to write the
SEDS-anchored product into hist_full40_seds. Six consumers kept their own hardcoded copy of the
old directory and silently read a net load whose subregion mean is 25.2 GW against the rebuilt
21.0 GW, so a rebuild of the load never reached Figure 1 or Figure 5. Naming the product once
removes the class of failure, and the loader below refuses a product that is not annually
anchored, so a stale file raises instead of being plotted.
"""
import numpy as np

LOAD_DIR = "/data/tell_pred/future/hist_full40_seds"
NETLOAD_NPZ = LOAD_DIR + "/subregion_netload_ourchain_1980_2019.npz"
SUPERSEDED_DIR = "/data/tell_pred/future/hist_full40"   # unanchored, kept only for provenance
MIN_GROWTH = 1.5      # 1980 to 2019 national total; the anchored product is 1.81, the flat one 1.00


def netload(path=NETLOAD_NPZ, check=True):
    """Load the subregion net-load product and refuse it if it is not annually anchored."""
    z = np.load(path, allow_pickle=True)
    if check:
        t = np.asarray(z["times"])
        yr = np.array([int(str(x)[:4]) for x in t])
        L = np.asarray(z["load"], dtype=np.float64).sum(0)
        a80, a19 = L[yr == 1980].sum(), L[yr == 2019].sum()
        if a80 <= 0 or a19 / a80 < MIN_GROWTH:
            raise ValueError(
                "%s is not annually anchored: 1980 total %.4e, 2019 total %.4e, ratio %.3f is under "
                "%.2f. The unanchored product under %s is superseded and must not be read."
                % (path, a80, a19, (a19 / a80) if a80 else float("nan"), MIN_GROWTH, SUPERSEDED_DIR))
    return z


# ---------------------------------------------------------------- the three load products
# FIXED ECONOMY. The economy is frozen at 2019 and only the weather varies, which is what a
# fixed-fleet counterfactual needs: Figure 1 asks what weather does to a system, so letting demand
# grow 1.82x across the record would mix the demand trend into the answer. Built by dividing the
# anchored product by its own SEDS state-year factor, so the annual total returns to the model's
# own value for that year and still moves with the weather (sd 0.82%), while the real county shares
# survive. FIGURE 1 ONLY.
NETLOAD_FIXEDECON = LOAD_DIR + "/subregion_netload_ourchain_1980_2019_fixedecon.npz"
COUNTY_FIXEDECON = LOAD_DIR + "/county_load_hourly_fixedecon.npy"
# ANCHORED. The real economic path, 1980 to 2019 growth of 1.82. Everything that is compared with
# an observation or projected forward takes this one.
COUNTY_ANCHORED = LOAD_DIR + "/county_load_hourly_realdist.npy"
MAX_FIXED_GROWTH = 1.10   # the fixed-economy product must NOT grow; measured 1.028


def netload_fixedecon(path=NETLOAD_FIXEDECON):
    """The fixed-economy net load, refused if it carries an economic trend."""
    z = np.load(path, allow_pickle=True)
    t = np.asarray(z["times"])
    yr = np.array([int(str(x)[:4]) for x in t])
    L = np.asarray(z["load"], dtype=np.float64).sum(0)
    a80, a19 = L[yr == 1980].sum(), L[yr == 2019].sum()
    r = a19 / a80 if a80 else float("nan")
    if not (1.0 / MAX_FIXED_GROWTH <= r <= MAX_FIXED_GROWTH):
        raise ValueError(
            "%s carries an economic trend: 1980/2019 ratio %.3f is outside %.2f to %.2f. Figure 1 "
            "needs the fixed-economy product; the anchored one is at %s."
            % (path, r, 1.0 / MAX_FIXED_GROWTH, MAX_FIXED_GROWTH, NETLOAD_NPZ))
    return z
