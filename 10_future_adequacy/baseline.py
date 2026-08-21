"""The historical reference period every Figure 5 comparison is made against.

WHY THIS FILE EXISTS. The canonical historical product named in `paths.NETLOAD_NPZ` is annually
anchored, so the national total grows 1.82x between 1980 and 2019 and the annual peak net load
climbs from 336 GW in 1980 to 642 GW in 2019. An average taken over all forty years is therefore
not a system that ever operated. The mean annual maximum over the whole record is 535.06 GW, which
is 110 GW under the modern peak and 18% under the 648.4 GW that `cerf_out/R4_NETLOAD_RESULTS.md`
validates against EIA-930 (651.5 GW observed, agreement within 0.5%).

Every future arm in Figure 5 covers 2030 to 2050, so the reference has to be the system as it now
stands rather than its forty-year average. The last ten years of the record give 645.55 GW, within
1.1% of the validated observation and within 1.3% of the 649.88 GW the figure carried while it was
still reading the superseded flat-economy product under `hist_full40`. Restoring that level is the
point: the migration to the anchored product was never meant to move the baseline, and it moved it
by 22%.

THE WINDOW APPLIES TO THE COUNT AS WELL AS TO THE THRESHOLD. A threshold set on 2010 to 2019 and
then counted over 1980 to 2019 would spend three decades physically unable to reach a level the
economy had not yet grown into, so the historical rate would be diluted roughly fourfold. Both the
threshold and the historical rate use the window returned here.

THE FUTURE ARMS ARE NOT WINDOWED. They are 2030 to 2050 throughout and are compared whole.
"""
import numpy as np

# Ten years, not thirty. A climate normal is thirty because the weather is what is being averaged;
# here the economy moves as well, and over thirty years of this record it moves by a factor of 1.7.
N_BASE_YEARS = 10


def base_years(years):
    """The reference years: the last N_BASE_YEARS present in `years`, ascending."""
    u = np.unique(np.asarray(years).astype(int))
    if len(u) < N_BASE_YEARS:
        raise ValueError("the historical record carries %d years, fewer than the %d the reference "
                         "period needs" % (len(u), N_BASE_YEARS))
    return u[-N_BASE_YEARS:]


def base_mask(years):
    """Boolean mask selecting the reference period out of an hourly `years` vector."""
    y = np.asarray(years).astype(int)
    return np.isin(y, base_years(y))


def base_tail_hours(n_hours, n_years):
    """Index of the first reference-period hour in a record of `n_hours` over `n_years`.

    For the county arrays, which carry no time vector, only a uniform hour count per year. The
    division is checked rather than assumed, because a record that is not uniform would silently
    take the wrong block."""
    if n_hours % n_years:
        raise ValueError("%d hours do not divide into %d whole years, so the reference period "
                         "cannot be taken as a tail block" % (n_hours, n_years))
    return n_hours - (n_hours // n_years) * N_BASE_YEARS
