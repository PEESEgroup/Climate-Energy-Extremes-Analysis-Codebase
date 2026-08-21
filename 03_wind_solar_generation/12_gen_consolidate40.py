import os, numpy as np, pandas as pd
G="/data/datasets/gen/tgw-gen-historical"; OUT="/data/gen_targets"; os.makedirs(OUT, exist_ok=True)
Y0,Y1=1980,2019
def consolidate(tech):
    cfg=pd.read_csv(f"{G}/eia_{tech}_configs.csv", dtype={'plant_code_unique':str})
    cfg=cfg.drop_duplicates('plant_code_unique').set_index('plant_code_unique')
    frames=[]
    for y in range(Y0,Y1+1):
        df=pd.read_csv(f"{G}/{tech}/{tech}_gen_cf_{y}.csv", index_col=0)
        df.index=pd.to_datetime(df.index, utc=True)
        df.columns=df.columns.astype(str)
        frames.append(df.astype(np.float16))
        print(f"{tech} {y} {df.shape}", flush=True)
    full=pd.concat(frames)
    plants=[c for c in full.columns if c in cfg.index]
    full=full[plants]
    nan=int(np.isnan(full.to_numpy()).sum())
    cf=np.nan_to_num(full.to_numpy(dtype=np.float16))
    times=full.index.strftime('%Y%m%d%H').to_numpy()
    keep=[c for c in ['lat','lon','system_capacity','ba','nerc_region','state','county'] if c in cfg.columns]
    meta=cfg.loc[plants, keep].reset_index()
    np.savez(f"{OUT}/{tech}_cf_{Y0}_{Y1}.npz", cf=cf, times=times, plants=np.array(plants))
    meta.to_csv(f"{OUT}/{tech}_meta.csv", index=False)
    print(f"{tech}: cf{cf.shape} plants={len(plants)} nan_filled={nan} -> {OUT}", flush=True)
for t in ['solar','wind']:
    consolidate(t)
print("ALL DONE", flush=True)
