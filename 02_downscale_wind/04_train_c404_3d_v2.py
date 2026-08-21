"""Physics-fused 3D-WIND SRGAN training on CONUS404 hub-height winds (fp16 cache) — v2 FORK of train_c404_3d.py.
Single wind GROUP: G(in=21 [14 wind + 7 pred-context], out=14 [u,v @ 7 levels]) on the fine 625x1475 grid.
Baseline fixes (a)-(d) unchanged; adds a DEFENSE-IN-DEPTH non-finite guard (skip poisoned batch).

v2 adds THREE physics-oriented, STATIONARY, ablatable wind losses (all OFF by default -> reproduces baseline):
  (i)  --speed_w   : wind-SPEED + POWER-DENSITY (speed^3) moment matching. De-normalizes (u,v) to physical
                     m/s via norm_stats, computes speed=sqrt(u^2+v^2) per level, penalizes RELATIVE |mean/var|
                     mismatch of speed and speed^3 SR-vs-HR (power ~ speed^3; Charbonnier alone misses moments).
  (ii) --psd_w     : radially-averaged log-PSD matching on the ~8-28 km band (torch.fft.rfft2 + Hann window,
                     radial rings), on per-level windspeed, driving band PSD ratio -> 1 (fights high-k deficit).
  (iii)--vshear_w  : VERTICAL-PROFILE prior across the 7 heights [10,40,80,110,140,160,200] m — smoothness of
                     d(speed)/dz (roughness of the shear profile) + soft surface-layer monotonicity (no-crossing).
  --hr_dx_km (4.0) / --psd_band ("8,28") parameterize the PSD band; --psd_nring (12) radial rings.
All three are added additively to the G loss in BOTH phases. With weights 0 the extra block is skipped entirely
so v2 is bit-identical to the baseline recipe.
  Smoke (GPU):  /opt/pytorch/bin/python 04_train_c404_3d_v2.py --smoke --speed_w 0.1 --psd_w 0.05 --vshear_w 1.0
  Full run:     see the launch command in the handoff (single GPU busy with baseline -> do NOT launch yet)."""
import os, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from model.srgan import SRGAN_g, SRGAN_d

OUT_CH=14; IN_CH=21; D_LR=1e-5; ADV_W=5e-3
CYC=[(i,i) for i in range(14)]                    # every HR wind ch <-> its LR wind ch (LR idx 0..13)
NLV=7
HEIGHTS=[10,40,80,110,140,160,200]                # 7 output levels (m AGL); ch 2h=u@h, ch 2h+1=v@h

def m_mse(p,t,M):                                 # kept for val/PSNR reporting
    n=p.shape[0]*p.shape[1]; return ((p-t)**2*M).sum()/(M.sum()*n+1e-8)
def m_charb(p,t,M,eps=1e-3):                      # (b) Charbonnier content loss (robust L1)
    n=p.shape[0]*p.shape[1]; return (torch.sqrt((p-t)**2+eps*eps)*M).sum()/(M.sum()*n+1e-8)
def m_grad(p,t,M):
    n=p.shape[0]*p.shape[1]; Mx=M[...,1:]; My=M[:,:,1:,:]
    return (((p[...,1:]-p[...,:-1])-(t[...,1:]-t[...,:-1])).abs()*Mx).sum()/(Mx.sum()*n+1e-8) \
         + (((p[:,:,1:,:]-p[:,:,:-1,:])-(t[:,:,1:,:]-t[:,:,:-1,:])).abs()*My).sum()/(My.sum()*n+1e-8)
def m_tv(x,M):
    n=x.shape[0]*x.shape[1]; Mx=M[...,1:]; My=M[:,:,1:,:]
    return ((x[...,1:]-x[...,:-1]).abs()*Mx).sum()/(Mx.sum()*n+1e-8) \
         + ((x[:,:,1:,:]-x[:,:,:-1,:]).abs()*My).sum()/(My.sum()*n+1e-8)
def d_ragan_w(dr,df,w):
    s=w.sum()+1e-8; rm=((dr*w).sum()/s).detach(); fm=((df*w).sum()/s).detach()
    return 0.5*(((dr-fm-1)**2*w).sum()/s + ((df-rm+1)**2*w).sum()/s)
def g_ragan_w(dr,df,w):
    s=w.sum()+1e-8; rm=(dr*w).sum()/s; fm=(df*w).sum()/s
    return 0.5*(((dr-fm+1)**2*w).sum()/s + ((df-rm-1)**2*w).sum()/s)
def d_hinge_w(dr,df,w):
    s=w.sum()+1e-8; return (F.relu(1.-dr)*w).sum()/s + (F.relu(1.+df)*w).sum()/s
def g_hinge_w(df,w):
    s=w.sum()+1e-8; return -(df*w).sum()/s
def phys_range_3d(fake, hm, hs):                  # (d) windspeed<=60 m/s penalty at every one of the 7 levels
    raw=fake*hs+hm; p=fake.new_zeros(())
    for h in range(NLV):
        spd=torch.sqrt(raw[:,2*h]**2+raw[:,2*h+1]**2+1e-6); p=p+F.relu(spd-60).mean()
    return p/NLV
def cyc_loss(fake, lr_full, hm, hs, lm, ls, Mc):  # (a) de-aliased coarsen via adaptive_avg_pool2d
    fc=F.adaptive_avg_pool2d(fake, lr_full.shape[-2:])
    tot=fake.new_zeros(()); k=0
    Mc_c=F.adaptive_avg_pool2d(Mc, fc.shape[-2:])
    for hg,lf in CYC:
        raw_lr=lr_full[:,lf]*ls[lf]+lm[lf]; tgt=(raw_lr-hm[0,hg,0,0])/hs[0,hg,0,0]
        tot=tot+(((fc[:,hg]-tgt).abs()*Mc_c[0,0]).sum()/(Mc_c.sum()*fc.shape[0]+1e-8)); k+=1
    return tot/max(k,1)

# ============================== v2 wind-specific physics losses (STATIONARY; no per-year/per-BA fits) ==============================
def _speeds(z, hm, hs):                            # z-space (B,14,H,W) -> physical per-level speed (B,7,H,W) m/s
    raw=z*hs+hm
    return torch.stack([torch.sqrt(raw[:,2*h]**2+raw[:,2*h+1]**2+1e-6) for h in range(NLV)],1)
def _mmask(x, m):                                  # x:(B,...,H,W), m:(H,W) -> masked mean over H,W then over other dims (scalar)
    s=m.sum().clamp_min(1.0); return ((x*m).sum(dim=(-1,-2))/s).mean()
def speedpow_loss(ff, hr, hm, hs, M):              # (i) speed + power-density (v^3) RELATIVE moment matching
    m=M[0,0]; s=m.sum().clamp_min(1.0); eps=1e-6
    sp_sr=_speeds(ff,hm,hs); sp_hr=_speeds(hr,hm,hs)
    tot=ff.new_zeros(())
    for h in range(NLV):
        for xs,xt in ((sp_sr[:,h],sp_hr[:,h]),(sp_sr[:,h]**3,sp_hr[:,h]**3)):   # speed and speed^3
            ms=((xs*m).sum(dim=(-1,-2))/s); mt=((xt*m).sum(dim=(-1,-2))/s)      # per-sample masked mean
            vs=(((xs-ms[:,None,None])**2)*m).sum(dim=(-1,-2))/s
            vt=(((xt-mt[:,None,None])**2)*m).sum(dim=(-1,-2))/s
            ms,mt,vs,vt=ms.mean(),mt.mean(),vs.mean(),vt.mean()
            tot=tot+(ms-mt).abs()/(mt.abs()+eps)+(vs-vt).abs()/(vt.abs()+eps)
    return tot/(NLV*4)                             # 7 levels x {speed,power} x {mean,var}
def build_psd(H,W,dx_km,band_lo,band_hi,nring,dev):# precompute Hann window + radial ring masks for the km band
    ky=torch.fft.fftfreq(H,device=dev); kx=torch.fft.rfftfreq(W,device=dev)     # cycles/pixel
    KY,KX=torch.meshgrid(ky,kx,indexing='ij'); kr=torch.sqrt(KY**2+KX**2)
    lam=torch.where(kr>0, dx_km/kr, torch.full_like(kr,1e9))                    # wavelength in km
    band=(lam>=band_lo)&(lam<=band_hi)
    rings=[]
    if band.any():
        lo,hi=kr[band].min(),kr[band].max(); edges=torch.linspace(float(lo),float(hi),nring+1,device=dev)
        for r in range(nring):
            rm=band&(kr>=edges[r])&(kr<=(edges[r+1] if r==nring-1 else edges[r+1]))
            if rm.any(): rings.append(rm)
    wy=torch.hann_window(H,periodic=False,device=dev); wx=torch.hann_window(W,periodic=False,device=dev)
    win=(wy[:,None]*wx[None,:])
    return win, rings
def psd_loss(ff, hr, hm, hs, win, rings):          # (ii) radially-averaged log-PSD matching on the band -> ratio 1
    if not rings: return ff.new_zeros(())
    sp_sr=_speeds(ff,hm,hs); sp_hr=_speeds(hr,hm,hs); tot=ff.new_zeros(()); nr=0
    for h in range(NLV):
        a=sp_sr[:,h]; b=sp_hr[:,h]
        a=a-a.mean(dim=(-1,-2),keepdim=True); b=b-b.mean(dim=(-1,-2),keepdim=True)  # remove DC before window
        Pa=torch.fft.rfft2(a*win).abs()**2; Pb=torch.fft.rfft2(b*win).abs()**2      # (B,H,Wf)
        for rm in rings:
            pa=Pa[:,rm].mean(); pb=Pb[:,rm].mean()
            tot=tot+(torch.log(pa+1e-8)-torch.log(pb+1e-8)).abs(); nr+=1
    return tot/max(nr,1)
def vshear_loss(ff, hm, hs, M, dz):                # (iii) SR-only vertical-profile prior (smoothness + surface monotonicity)
    m=M[0,0]; spd=_speeds(ff,hm,hs)                # (B,7,H,W) physical m/s
    shear=(spd[:,1:]-spd[:,:-1])/dz                # (B,6,H,W) d(speed)/dz, non-uniform heights
    rough=(shear[:,1:]-shear[:,:-1]).abs()         # (B,5,H,W) roughness of shear profile (curvature of speed(z))
    mono=F.relu(-shear[:, :2])                     # surface layer 10->40->80 m should not decrease with height
    return _mmask(rough,m)+_mmask(mono,m)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs_init',type=int,default=4); ap.add_argument('--epochs_adv',type=int,default=10)
    ap.add_argument('--bs',type=int,default=3); ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--lam_cyc',type=float,default=5e-2); ap.add_argument('--lam_phys',type=float,default=1e-2)
    ap.add_argument('--lam_grad',type=float,default=0.1)
    # ---- v2 wind-specific physics losses (OFF by default -> reproduces baseline) ----
    ap.add_argument('--speed_w',type=float,default=0.0,help='(i) speed + power-density(v^3) relative moment loss')
    ap.add_argument('--psd_w',type=float,default=0.0,help='(ii) radial log-PSD band-matching loss')
    ap.add_argument('--vshear_w',type=float,default=0.0,help='(iii) vertical-profile smoothness + surface monotonicity prior')
    ap.add_argument('--hr_dx_km',type=float,default=4.0,help='fine-grid spacing (km) for PSD wavelength mapping')
    ap.add_argument('--psd_band',default='8,28',help='PSD wavelength band lo,hi in km')
    ap.add_argument('--psd_nring',type=int,default=12,help='number of radial rings across the PSD band')
    ap.add_argument('--no_mask',action='store_true'); ap.add_argument('--no_amp',action='store_true')
    ap.add_argument('--no_static',action='store_true',help='disable baked terrain conditioning')
    ap.add_argument('--flip',action='store_true',help='random horizontal-flip aug (forces --no_static)')
    ap.add_argument('--crop',type=int,default=0,help='random LR crop N -> HR 7N (forces --no_static); 0=full frame')
    ap.add_argument('--out',default='/data/ckpt/g3d'); ap.add_argument('--val_batches',type=int,default=0)
    ap.add_argument('--patience',type=int,default=5); ap.add_argument('--min_delta',type=float,default=1e-4)
    ap.add_argument('--max_iters',type=int,default=0); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--adv_warmup',type=int,default=1000)
    ap.add_argument('--advloss',default='hinge',choices=['hinge','ragan'])
    ap.add_argument('--snap_adv',action='store_true')
    a=ap.parse_args(); n_ep=a.epochs_init+a.epochs_adv
    if a.flip or a.crop: a.no_static=True                        # aug needs terrain off (buffers can't flip/crop)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp=(not a.no_amp) and dev.type=='cuda'
    import contextlib
    def ac(): return torch.autocast('cuda',dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    crop_size=(7*a.crop,7*a.crop) if a.crop else (625,1475)
    # static terrain(z)+land-sea conditioning (same machinery as surface trainer)
    if a.no_static:
        _static_lr=None; _static_hr=None
        print("[static] terrain conditioning OFF (augmentation mode)",flush=True)
    else:
        _S=np.load("/data/datasets/grid/static_terrain_landmask.npz")
        _hgt=_S['hgt'].astype('float32'); _lmask=_S['landmask'].astype('float32'); _lake=_S['lakemask'].astype('float32')
        _island=((_lmask>0.5)&(_lake<0.5)).astype('float32'); _hgtz=(_hgt-_hgt.mean())/(_hgt.std()+1e-6)
        _static_hr=np.stack([_hgtz,_island*2-1],0)
        _static_lr=F.adaptive_avg_pool2d(torch.from_numpy(_static_hr)[None],(89,211))[0].numpy()
        print(f"[static] terrain+landmask hr{_static_hr.shape} lr{_static_lr.shape}",flush=True)
    G=SRGAN_g(in_channels=IN_CH,out_channels=OUT_CH,crop_size=crop_size,
              static_lr=_static_lr,static_hr=_static_hr,dyn_lr_ch=0,dyn_hr_ch=0).to(dev)
    D=SRGAN_d(in_channels=OUT_CH).to(dev)
    def raw(m): return getattr(m,'_orig_mod',m)
    nparam=sum(p.numel() for p in G.parameters()); ndparam=sum(p.numel() for p in D.parameters())
    print(f"[params] G={nparam:,} ({nparam/1e6:.2f}M)  D={ndparam:,} ({ndparam/1e6:.2f}M)",flush=True)
    og=torch.optim.Adam(G.parameters(),lr=2e-4,betas=(0.9,0.999)); od=torch.optim.Adam(D.parameters(),lr=D_LR,betas=(0.9,0.999))
    gsch=torch.optim.lr_scheduler.CosineAnnealingLR(og,T_max=n_ep,eta_min=1e-6)
    dsch=torch.optim.lr_scheduler.CosineAnnealingLR(od,T_max=max(a.epochs_adv,1),eta_min=1e-6)
    T="/data/datasets/train"; st=np.load(f"{T}/norm_stats_3d.npz")
    hm=torch.tensor(st['hr_mean'],dtype=torch.float32,device=dev).view(1,OUT_CH,1,1)
    hs=torch.tensor(st['hr_std'],dtype=torch.float32,device=dev).view(1,OUT_CH,1,1)
    lm=torch.tensor(st['lr_mean'],dtype=torch.float32,device=dev); ls=torch.tensor(st['lr_std'],dtype=torch.float32,device=dev)
    masked = (not a.no_mask) and (a.crop==0)                     # ROI mask only aligns in full-frame mode
    rm=np.load("/data/datasets/grid/land_roi_mask.npz")
    M=torch.tensor(rm['roi'].astype('float32'),device=dev)[None,None]
    Mc=torch.tensor(rm['roi_coarse'].astype('float32'),device=dev)[None,None]
    # ---- v2: precompute PSD band (window+rings) + vertical dz; only used when the corresponding weight>0 ----
    v2_on = (a.speed_w>0) or (a.psd_w>0) or (a.vshear_w>0)
    _bl,_bh=[float(x) for x in a.psd_band.split(',')]
    _Hf,_Wf=crop_size
    win_psd,rings_psd=build_psd(_Hf,_Wf,a.hr_dx_km,_bl,_bh,a.psd_nring,dev)
    dz_v=torch.tensor([HEIGHTS[i+1]-HEIGHTS[i] for i in range(NLV-1)],dtype=torch.float32,device=dev).view(1,NLV-1,1,1)
    if v2_on: print(f"[v2] speed_w{a.speed_w} psd_w{a.psd_w} vshear_w{a.vshear_w} | PSD band {_bl:.0f}-{_bh:.0f}km dx{a.hr_dx_km}km rings{len(rings_psd)}",flush=True)

    def g_losses(fake, hr, lrf, Mu, adv_on, adv_scale=1.0):
        ff=fake.float()
        pix=m_charb(ff,hr,Mu); gr=m_grad(ff,hr,Mu)
        cyc=cyc_loss(ff,lrf,hm,hs,lm,ls,Mc) if a.crop==0 else ff.new_zeros(())   # cyc uses full-grid ROI
        ph_=phys_range_3d(ff,hm,hs)
        if adv_on:
            if a.advloss=='hinge':
                with ac(): dfk=D(fake)
                w=F.adaptive_avg_pool2d(Mu,dfk.shape[-2:]); adv=g_hinge_w(dfk.float(),w)
            else:
                with ac(): dro=D(hr.detach()); dfk=D(fake)
                w=F.adaptive_avg_pool2d(Mu,dro.shape[-2:]); adv=g_ragan_w(dro.float(),dfk.float(),w)
            base=pix+a.lam_grad*gr+adv_scale*ADV_W*adv                            # (c) grad kept in adv phase
        else:
            adv=ff.new_zeros(()); base=pix+a.lam_grad*gr
        tot=base+a.lam_cyc*cyc+a.lam_phys*ph_
        # ---- v2 additive wind-specific physics terms (both phases); skipped entirely when all weights 0 ----
        spd_l=ff.new_zeros(()); psd_l=ff.new_zeros(()); vsh_l=ff.new_zeros(())
        if v2_on:
            if a.speed_w>0:  spd_l=speedpow_loss(ff,hr,hm,hs,Mu)
            if a.psd_w>0:    psd_l=psd_loss(ff,hr,hm,hs,win_psd,rings_psd)
            if a.vshear_w>0: vsh_l=vshear_loss(ff,hm,hs,Mu,dz_v)
            tot=tot+a.speed_w*spd_l+a.psd_w*psd_l+a.vshear_w*vsh_l
        return tot, pix,gr,cyc,ph_,adv,spd_l,psd_l,vsh_l
    def d_step(fkd,hr,Mu):
        with ac(): dro=D(hr); dfk=D(fkd)
        w=F.adaptive_avg_pool2d(Mu,dro.shape[-2:]); ld=d_hinge_w(dro.float(),dfk.float(),w) if a.advloss=='hinge' else d_ragan_w(dro.float(),dfk.float(),w)
        if not torch.isfinite(ld): od.zero_grad(set_to_none=True); return   # non-finite guard: skip poisoned D step
        od.zero_grad(); ld.backward(); nn.utils.clip_grad_norm_(D.parameters(),1.0); od.step()

    if a.smoke:
        print(f"[smoke] dev={dev} out={OUT_CH} in={IN_CH} masked={masked} amp={amp} crop_size={crop_size}")
        B=2; H,W=crop_size
        lrf=torch.randn(B,IN_CH, (a.crop or 89),(a.crop or 211),device=dev); hr=torch.randn(B,OUT_CH,H,W,device=dev)
        Mu = M if masked else torch.ones(1,1,H,W,device=dev)
        for ph,adv_on in [("content",False),("adversarial",True)]:
            with ac(): fake=G(lrf)
            if adv_on: d_step(fake.detach(),hr,Mu)
            gl,pix,gr,cyc,ph_,adv,spd_l,psd_l,vsh_l=g_losses(fake,hr,lrf,Mu,adv_on)
            assert torch.isfinite(gl), f"non-finite G loss in smoke [{ph}]"
            og.zero_grad(); gl.backward(); nn.utils.clip_grad_norm_(G.parameters(),1.0); og.step()
            print(f"  [{ph}] fake{tuple(fake.shape)} pix{pix.item():.4f} grad{gr.item():.4f} cyc{float(cyc):.4f} phys{ph_.item():.4f} adv{float(adv):.4f}"
                  f" | v2: speedpow{float(spd_l):.5f} psd{float(psd_l):.5f} vshear{float(vsh_l):.5f} | g{gl.item():.4f} OK")
        print(f"[smoke] PASS  G={nparam/1e6:.2f}M params  backward OK  (v2 terms finite)"); return

    import sys, json, csv, math; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    from srgan_dataset_3d import C404SRGAN3D
    os.makedirs(a.out,exist_ok=True)
    cropN=a.crop or None
    tr=DataLoader(C404SRGAN3D('train',flip=a.flip,crop=cropN),batch_size=a.bs,shuffle=True,num_workers=a.workers,
                  pin_memory=True,drop_last=True,persistent_workers=a.workers>0,prefetch_factor=4 if a.workers>0 else None)
    va=DataLoader(C404SRGAN3D('val'),batch_size=a.bs,shuffle=False,num_workers=2,pin_memory=True)
    te=DataLoader(C404SRGAN3D('test'),batch_size=a.bs,shuffle=False,num_workers=2,pin_memory=True)
    def eval_mse(loader,nb=0):
        G.eval(); s=0.; n=0
        with torch.no_grad():
            for j,b in enumerate(loader):
                if nb and j>=nb: break
                lrf=b['lr'].to(dev,non_blocking=True); hr=b['hr'].to(dev,non_blocking=True)
                Mu = M if masked else torch.ones(1,1,hr.shape[2],hr.shape[3],device=dev)
                with ac(): fake=G(lrf)
                s+=m_mse(fake.float(),hr,Mu).item(); n+=1
        G.train(); return s/max(n,1)
    json.dump({**vars(a),'device':str(dev),'out_ch':OUT_CH,'in_ch':IN_CH,'masked':masked,'crop_size':crop_size,
               'fixes':'de- aliased coarsen(adaptive_avg_pool2d) + Charbonnier content + grad-in-adv + per-level phys cap',
               'n_train':len(tr.dataset),'n_val':len(va.dataset),'n_test':len(te.dataset)},
              open(f"{a.out}/config.json",'w'),indent=2,default=str)
    mcsv=f"{a.out}/metrics.csv"
    if not os.path.exists(mcsv):
        csv.writer(open(mcsv,'w')).writerow(['epoch','phase','sec','lr_g','tr_pix','tr_grad','tr_cyc','tr_adv','val_mse','val_psnr','best','since_best'])
    start_ep=0; best=1e9; since=0; ck=f"{a.out}/last.pt"
    if os.path.exists(ck):
        c=torch.load(ck,map_location=dev); raw(G).load_state_dict(c['G']); raw(D).load_state_dict(c['D'])
        og.load_state_dict(c['og']); od.load_state_dict(c['od']); gsch.load_state_dict(c['gsch']); dsch.load_state_dict(c['dsch'])
        start_ep=c['epoch']+1; best=c['best']; since=c.get('since',0); print(f"resumed ep{start_ep} best{best:.4f}",flush=True)
    print(f"dev {dev} out {OUT_CH} in {IN_CH} masked {masked} amp {amp} | train {len(tr)} val {len(va)} test {len(te)} batches | patience {a.patience}",flush=True)
    stopped=False; adv_it=0
    for ep in range(start_ep,n_ep):
        adv_on=ep>=a.epochs_init; G.train(); D.train(); t0=time.time()
        sp_=sg=sc=sa=0.0; ni=0
        for it,b in enumerate(tr):
            if a.max_iters and it>=a.max_iters: break
            lrf=b['lr'].to(dev,non_blocking=True); hr=b['hr'].to(dev,non_blocking=True)
            Mu = M if masked else torch.ones(1,1,hr.shape[2],hr.shape[3],device=dev)
            adv_scale = min(1.0, adv_it/max(a.adv_warmup,1)) if adv_on else 0.0
            with ac(): fake=G(lrf)
            if adv_on: d_step(fake.detach(),hr,Mu); adv_it+=1
            gl,pix,gr,cyc,ph_,adv,spd_l,psd_l,vsh_l=g_losses(fake,hr,lrf,Mu,adv_on,adv_scale)
            if not torch.isfinite(gl):                                    # non-finite guard (both phases): skip poisoned batch
                og.zero_grad(set_to_none=True); print(f"ep{ep} it{it} NONFINITE G-loss -> batch skipped",flush=True); continue
            og.zero_grad(); gl.backward(); nn.utils.clip_grad_norm_(G.parameters(),1.0); og.step()
            sp_+=pix.item(); sg+=gr.item(); sc+=float(cyc); sa+=float(adv); ni+=1
            if it%100==0: print(f"ep{ep}({'adv' if adv_on else 'init'}) it{it}/{len(tr)} pix{pix.item():.4f} grad{gr.item():.4f} cyc{float(cyc):.4f} adv{float(adv):.4f} aw{adv_scale:.2f}"
                                + (f" | spd{float(spd_l):.4f} psd{float(psd_l):.4f} vsh{float(vsh_l):.4f}" if v2_on else ""),flush=True)
        gsch.step()
        if adv_on: dsch.step()
        vmse=eval_mse(va,a.val_batches); vpsnr=-10*math.log10(max(vmse,1e-9)); sec=time.time()-t0
        improved = vmse < best - a.min_delta
        if improved: best=vmse; torch.save(raw(G).state_dict(),f"{a.out}/G_best.pth")
        if ep==a.epochs_init: since=0
        if adv_on: since = 0 if improved else since+1
        torch.save({'epoch':ep,'G':raw(G).state_dict(),'D':raw(D).state_dict(),'og':og.state_dict(),'od':od.state_dict(),
                    'gsch':gsch.state_dict(),'dsch':dsch.state_dict(),'best':best,'since':since},ck)
        if a.snap_adv and adv_on: torch.save(raw(G).state_dict(),f"{a.out}/G_advep{ep+1}.pth")
        csv.writer(open(mcsv,'a')).writerow([ep+1,'adv' if adv_on else 'init',round(sec,1),f"{og.param_groups[0]['lr']:.2e}",
                    round(sp_/ni,5),round(sg/ni,5),round(sc/ni,5),round(sa/ni,5),round(vmse,5),round(vpsnr,3),round(best,5),since])
        print(f"== ep{ep+1}/{n_ep} {sec/60:.1f}min val_mse{vmse:.5f} psnr{vpsnr:.2f} best{best:.5f} since{since}{' *' if improved else ''}",flush=True)
        if adv_on and since>=a.patience:
            print(f"[early stop] no val improvement for {a.patience} adv epochs (best {best:.5f})",flush=True); stopped=True; break
    if os.path.exists(f"{a.out}/G_best.pth"): raw(G).load_state_dict(torch.load(f"{a.out}/G_best.pth",map_location=dev))
    tmse=eval_mse(te); tpsnr=-10*math.log10(max(tmse,1e-9))
    json.dump({'best_val_mse':best,'test_mse':tmse,'test_psnr':tpsnr,'stopped_early':stopped,'epochs_run':ep+1},
              open(f"{a.out}/summary.json",'w'),indent=2)
    print(f"[FINAL] best_val_mse{best:.5f} | TEST mse{tmse:.5f} psnr{tpsnr:.2f} (held-out yr 2015)",flush=True)

if __name__=="__main__": main()
