"""The wind dose-response Figure 6 applies, taken from the panel that produces Figure 3a and 3b.

Figure 6 pushes a projected shift in the county wind distribution through an observed response, so
it needs the WHOLE event, not the landfall day: a county-day at a given wind carries an impact on
days 0 to 1, a restoration on days 2 to 6 and a tail on days 7 to 14. The total log effect per
exposed county-day is the sum of those blocks, with the two banded blocks taken at that band.
"""
import json
import numpy as np
A = json.load(open("/data/equity_cost/analysis/attrib/attrib.json"))
R = A["results"]
BANDS = ["34 to 50 kt", "50 to 64 kt", "64 to 83 kt", "83 kt and above"]
MIDS = [42.0, 57.0, 73.5, 95.0]
tail = R["tc|tail"]
out = []
print("%-18s %8s %8s %8s %9s %10s" % ("band", "impact", "restore", "tail", "total", "multiplier"))
for b, m in zip(BANDS, MIDS):
    i_, r_ = R["tc|impact|" + b], R["tc|restore|" + b]
    tot = i_["beta"] + r_["beta"] + tail["beta"]
    se = float(np.sqrt(i_["se"] ** 2 + r_["se"] ** 2 + tail["se"] ** 2))   # independence assumed
    print("%-18s %+8.3f %+8.3f %+8.3f %+9.3f %10.1f" % (b, i_["beta"], r_["beta"], tail["beta"],
                                                        tot, np.exp(tot)))
    out.append(dict(band=b, mid_kt=m, impact=i_["beta"], restore=r_["beta"], tail=tail["beta"],
                    total=tot, total_se_independent=se, multiplier=float(np.exp(tot))))
print()
print("the object Figure 6 needs: total log effect per exposed county-day, by wind band")
print("   held flat above 83 kt, since no band is estimated above it")
json.dump(dict(note="total event effect per exposed county-day, days 0 to +14, by wind band, from "
                    "the county-by-calendar-month and day Poisson panel",
               bands=out,
               se_note="the standard error assumes the three blocks are independent, which they "
                       "are not; it is an approximation and is not quoted in the paper"),
          open("/data/equity_cost/analysis/attrib/dose_for_fig6.json", "w"), indent=1)
print("wrote /data/equity_cost/analysis/attrib/dose_for_fig6.json")
