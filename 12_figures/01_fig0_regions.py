"""FIGURE 0 — the 18 subregions and the plain-English name each one carries in the text.

The planning-region codes (SERTP, PJM_East, NorthernGrid_South) are opaque to a general reader, so
the Results text names every subregion in plain English. This is the key that makes that possible,
and it is the single source of truth: no other figure or paragraph may invent a different name.

The names are DERIVED, not remembered. `/data/aliases.py` reads the county-to-subregion mapping the
net-load panel is actually built on and counts the states each subregion covers; `subregion_states.csv`
holds the result and every alias below is that county-weighted state list in ordinary words. Note
that `county_sub` in the npz is 1-BASED; reading it as 0-based silently shifts every subregion by
one place and makes Florida look like Texas.

Drawn at A4 width so it sits with the rest of the set. The author annotates over this.
"""
import sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
from figstyle import *                                    # noqa
import numpy as np, pandas as pd
from matplotlib.lines import Line2D

ALIAS = {
    "CAISO": "California", "ERCOT": "Texas", "FRCC": "Florida",
    "ISONE": "New England", "NYISO": "New York", "PJM_East": "Mid-Atlantic",
    "PJM_West": "Chicago area", "MISO_Central": "Central Midwest",
    "MISO_North": "Upper Midwest", "MISO_South": "Lower Mississippi",
    "SERTP": "Southeast", "SPP_North": "Northern Plains", "SPP_South": "Southern Plains",
    "WestConnect_North": "Central Rockies", "WestConnect_South": "Southwest",
    "NorthernGrid_East": "Northern Rockies", "NorthernGrid_South": "Great Basin",
    "NorthernGrid_West": "Pacific Northwest"}
# label offsets in degrees for the crowded northeast, where a centroid label will not fit
NUDGE = {"NYISO": (8.6, 3.6), "ISONE": (9.0, 0.6), "PJM_East": (8.2, -4.8),
         "PJM_West": (-0.6, 4.4), "MISO_Central": (0.0, -1.6), "MISO_North": (-3.4, 0.2),
         "NorthernGrid_South": (-1.4, -0.8), "WestConnect_South": (0.6, -1.0)}

zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]
name2id = {str(n): int(i) for i, n in zm["id_to_subregion"]}
zc = np.load("/data/datasets/grid/coordinate.npz")
lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
rows = np.where((mask > 0).any(1))[0]; cols = np.where((mask > 0).any(0))[0]
r0, r1_, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
subm = mask[r0:r1_, c0:c1]
ext = [lon[c0], lon[c1 - 1], lat[r0], lat[r1_ - 1]]
LON, LAT = np.meshgrid(lon[c0:c1], lat[r0:r1_])
ASP = 1.0 / np.cos(np.deg2rad(np.mean(ext[2:])))

W_MM = 183.0
MAP_W = 179.0
MAP_H = MAP_W * (ext[3] - ext[2]) * ASP / (ext[1] - ext[0])
H_MM = MAP_H + 4.0
fig = new_fig(W_MM, H_MM)

def R(x, y, w, h):
    return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]

ax = fig.add_axes(R(2.0, 2.0, MAP_W, MAP_H))
# a repeating pale palette: colour separates neighbours, it carries no quantity
PAL = ["#DCE9F5", "#F6E3DC", "#E2EFE0", "#F1E6F2", "#FAF0D8", "#E0EDEF"]
img = np.full(subm.shape, np.nan)
order = sorted(name2id, key=lambda n: name2id[n])
for k, nm in enumerate(order):
    img[subm == name2id[nm]] = k
cmap = plt.matplotlib.colors.ListedColormap([PAL[k % len(PAL)] for k in range(len(order))])
ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=len(order) - 1,
          interpolation="nearest", aspect="auto")
ax.contour(np.where(subm > 0, subm, np.nan), levels=np.arange(.5, subm.max() + 1),
           colors="white", linewidths=.6, extent=ext, origin="lower")

for nm in order:
    m = subm == name2id[nm]
    if not m.sum():
        continue
    cx, cy = LON[m].mean(), LAT[m].mean()
    dx, dy = NUDGE.get(nm, (0.0, 0.0))
    tx, ty = cx + dx, cy + dy
    if (dx, dy) != (0.0, 0.0):
        ax.plot([cx, tx], [cy, ty], color="#9A9A9A", lw=.4, zorder=3)
        ax.scatter([cx], [cy], s=3.0, color="#5A5A5A", zorder=4)
    ax.text(tx, ty + 0.30, ALIAS[nm], fontsize=FS_LABEL, ha="center", va="bottom",
            color="#1A1A1A", zorder=5)
    ax.text(tx, ty - 0.30, nm.replace("_", " "), fontsize=FS_VAL, ha="center", va="top",
            color="#8A8A8A", zorder=5)
bare(ax)
ax.set_xlim(ext[0], ext[1] + 10.0)          # room for the three northeastern leaders
ax.set_ylim(ext[2], ext[3])

save(fig, "fig0_regions", tight=False)
print("page %.0f x %.0f mm, %d subregions labelled" % (W_MM, H_MM, len(order)))
pd.DataFrame([dict(code=k, name=v) for k, v in ALIAS.items()]).to_csv(
    "/data/figs/out/region_names.csv", index=False)
print("wrote /data/figs/out/region_names.csv — the single source of truth for these names")
