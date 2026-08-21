"""Regrid TGW 12km (curvilinear 299x424) -> CONUS404 LR grid (regular latlon 89x211) by conservative
area-average (each LR ~31km cell = mean of the TGW ~12km points falling in it). This is the physical
'blur' the SRGAN expects (Step 1 of TGW_DOWNSCALING_WORKFLOW.md). Regrid indices are grid-only -> computed
once, reused for all times/scenarios. PSFC restored hPa->Pa via the 'scale' key here.
Usage: python regrid_tgw.py --test <one_npz>   |   python regrid_tgw.py --scenario rcp45cooler (mass regrid)
"""
import numpy as np, glob, os, argparse, torch, torch.nn.functional as F

def pool1d(a,n): return F.adaptive_avg_pool1d(torch.tensor(np.asarray(a,dtype='float64'))[None,None],n)[0,0].numpy()
def cell_edges(c):                                   # cell-center array -> len+1 edges (assumes monotone increasing)
    m=0.5*(c[1:]+c[:-1]); return np.concatenate([[2*c[0]-m[0]],m,[2*c[-1]-m[-1]]])

# --- target LR grid (CONUS404 regular latlon coarsened to 89x211) ---
co=np.load("/data/datasets/grid/coordinate.npz")
LATLR=pool1d(co['lat'],89); LONLR=pool1d(co['lon'],211)          # cell centers, ascending
LATE=cell_edges(LATLR); LONE=cell_edges(LONLR)                    # 90, 212 edges
NLAT,NLON=89,211

def build_index(gridpath):
    g=np.load(gridpath); XLAT=g['XLAT'].astype('float64').ravel(); XLONG=g['XLONG'].astype('float64').ravel()
    bi_lat=np.digitize(XLAT,LATE)-1; bi_lon=np.digitize(XLONG,LONE)-1
    valid=(bi_lat>=0)&(bi_lat<NLAT)&(bi_lon>=0)&(bi_lon<NLON)
    cell=(bi_lat*NLON+bi_lon)[valid]
    counts=np.bincount(cell,minlength=NLAT*NLON).astype('float64')
    return valid, cell, counts

def regrid_stack(arr, valid, cell, counts):          # arr (n,7,299,424) -> (n,7,89,211) area-avg
    n,c,H,W=arr.shape; out=np.full((n,c,NLAT*NLON),np.nan,dtype='float64')
    nz=counts>0
    for i in range(n):
        for ch in range(c):
            f=arr[i,ch].ravel()[valid]
            s=np.bincount(cell,weights=f,minlength=NLAT*NLON)
            out[i,ch,nz]=s[nz]/counts[nz]
    return out.reshape(n,c,NLAT,NLON)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--test'); ap.add_argument('--scenario'); a=ap.parse_args()
    if a.test:
        scen=a.test.split('/')[-2]; gp=f"/data/tgw_extract/{scen}/tgw_grid.npz"
        valid,cell,counts=build_index(gp)
        print(f"target LR {NLAT}x{NLON} | TGW pts in-domain: {valid.sum()}/{valid.size} | LR cells with >=1 pt: {(counts>0).sum()}/{NLAT*NLON} | min/median pts/cell: {counts[counts>0].min():.0f}/{np.median(counts[counts>0]):.0f}")
        z=np.load(a.test); arr=z['data'][:1].astype('float32'); sc=z['scale']
        arr=arr*sc[None,:,None,None]                 # restore physical units (PSFC hPa->Pa etc via scale)
        # PSFC we keep hPa for readability here; undo:
        out=regrid_stack(arr, valid, cell, counts)
        vn=['u10','v10','q2','psfc','t2','long','short']
        print("regridded one timestep — per-var [min, mean, max], nan_frac:")
        for ch,name in enumerate(vn):
            o=out[0,ch]; nanf=np.isnan(o).mean()
            print(f"  {name:6s} [{np.nanmin(o):9.2f}, {np.nanmean(o):9.2f}, {np.nanmax(o):9.2f}]  nan={nanf:.3f}")
        # sanity on T2: south (low lat rows) warmer than north
        t2=out[0,4]; south=np.nanmean(t2[:20]); north=np.nanmean(t2[-20:])
        print(f"T2 south(24-30N) {south:.1f}K vs north(44-49N) {north:.1f}K  -> {'OK south warmer' if south>north else 'CHECK'}")
        np.save('/tmp/regrid_test_out.npy', out[0])
        print("saved /tmp/regrid_test_out.npy")
