"""County (FIPS) -> TGW 12km grid (299x424) membership, for county-mean weather.
Primary: point-in-polygon (grid-cell center within county). Fallback: counties with 0 cells get their
single nearest grid cell (small counties at 12km). CONUS only (drop AK/HI/territories).
Output /data/loads_measured/county_mask_tgw.npz: fips[str], pair_fips[int], pair_cell[int flat idx], H,W, counts.
"""
import numpy as np, geopandas as gpd
g = np.load("/data/tgw_hist/tgw_grid.npz")
XLAT = g['XLAT'].astype('float64'); XLONG = g['XLONG'].astype('float64'); H, W = XLAT.shape
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XLONG.ravel(), XLAT.ravel()), crs="EPSG:4326")
cty = gpd.read_file("/data/loads_measured/county_shp/tl_2018_us_county.shp")
cty['STATEFP'] = cty['STATEFP'].astype(int)
cty = cty[(cty['STATEFP'] <= 56) & (~cty['STATEFP'].isin([2, 15]))][['GEOID', 'geometry']].to_crs("EPSG:4326")
cty = cty.reset_index(drop=True)
fips = cty['GEOID'].tolist(); nC = len(fips); fidx = {f: i for i, f in enumerate(fips)}
# primary point-in-polygon
j = gpd.sjoin(pts, cty, how='left', predicate='within')
gid = j['GEOID'].groupby(level=0).first().reindex(range(H * W)).values
pf, pc = [], []
for cell, f in enumerate(gid):
    if isinstance(f, str): pf.append(fidx[f]); pc.append(cell)
pf = np.array(pf, np.int32); pc = np.array(pc, np.int32)
assigned = set(pf.tolist())
missing = [i for i in range(nC) if i not in assigned]
# fallback: nearest grid cell to county centroid
if missing:
    cen = cty.geometry.centroid
    glon = XLONG.ravel(); glat = XLAT.ravel()
    ef, ec = [], []
    for i in missing:
        d = (glon - cen.x.iloc[i])**2 + (glat - cen.y.iloc[i])**2
        ec.append(int(d.argmin())); ef.append(i)
    pf = np.concatenate([pf, np.array(ef, np.int32)]); pc = np.concatenate([pc, np.array(ec, np.int32)])
counts = np.bincount(pf, minlength=nC)
np.savez("/data/loads_measured/county_mask_tgw.npz", fips=np.array(fips), pair_fips=pf, pair_cell=pc,
         H=H, W=W, counts=counts)
print(f"CONUS counties: {nC} | via point-in-poly: {nC - len(missing)} | via nearest-fallback: {len(missing)}")
print(f"cells assigned (poly): {len(assigned)} unique counties; total memberships {len(pf)}")
print(f"cells/county min/median/max: {counts.min()}/{int(np.median(counts))}/{counts.max()}")
print(f"grid cells inside CONUS counties: {len(set(pc.tolist()))}/{H*W}")
