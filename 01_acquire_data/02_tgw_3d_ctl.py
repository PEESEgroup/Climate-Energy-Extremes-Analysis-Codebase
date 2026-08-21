"""Disk-guarded, HIGH-CONCURRENCY Globus->extract controller for the TGW 3-hourly 3-D wind stream.
Adapted from 06_tgw_pr_ctl.py for the BIG (9.48 GB) three_hourly files: small batches but MANY concurrent
transfer tasks ("more agents"), a few memory-heavy extract shards. Pairs with 03_process_tgw_3d.py (pwat-patched).
Idempotent + PWAT-AWARE: a target file is (re)fetched unless its npz exists AND already has the 'pwat' key
(stale pwat-less npz from before the patch are deleted so they get regenerated with pwat).
Supports --kstride for the strided schedule-C sample.

  python 02_tgw_3d_ctl.py --scenario historical --src-path /historical_1980_2019/three_hourly \
      --dropdir /data/tgw_3d_drop --outdir /data/tgw_3d --tmin 1980 --tmax 2019 \
      --kstride 10 --shards 3 --batch 6 --max-pending 72 --min-free-gb 800
"""
import os, re, glob, time, subprocess, argparse, signal
import numpy as np

SRC_EP  = "9392aa20-799e-4630-acb6-7e6a4ed5b795"   # tgw-wrf-conus (source)
DST_EP  = os.environ.get("GLOBUS_DST_EP", "")    # your own Globus endpoint UUID
GLOBUS  = "/home/ubuntu/venv/bin/globus"
PYEXE   = "/data/tellenv/bin/python"
EXTRACT = "/home/ubuntu/code/03_process_tgw_3d.py"
DATE_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})_")

def log(m): print(f"[3dctl {time.strftime('%H:%M:%S')}] {m}", flush=True)
def free_gb(path): s = os.statvfs(path); return s.f_bavail * s.f_frsize / 1e9

def globus_ls(src_path):
    out = subprocess.run([GLOBUS, "ls", f"{SRC_EP}:{src_path}/"], capture_output=True, text=True, timeout=180)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip().endswith(".nc")]

def want(fn, tmin, tmax):
    m = DATE_RE.search(fn)
    if not m: return False
    y, mo = int(m.group(1)), int(m.group(2))
    return (tmin <= y <= tmax) or (y == tmin - 1 and mo == 12)     # +Dec spillover week of year before tmin

def outpath(outdir, fn): return os.path.join(outdir, fn.replace('.nc', '.npz'))
def has_pwat(npz):
    try:
        z = np.load(npz); ok = ('pwat' in z.files and 'uv' in z.files); z.close(); return ok
    except Exception: return False

def submit_batch(items, dropdir, scenario, label):
    lines = [f"{SRC_BASE}/{it} {os.path.join(dropdir, it)}" for it in items]   # absolute src + absolute dst
    bf = f"/tmp/3d_batch_{scenario}_{label}.txt"
    with open(bf, "w") as fh: fh.write("\n".join(lines) + "\n")
    r = subprocess.run([GLOBUS, "transfer", SRC_EP, DST_EP, "--batch", bf,
                        "--label", f"3d-{scenario}-{label}", "--sync-level", "exists"],
                       capture_output=True, text=True, timeout=120)
    tid = ""
    for ln in (r.stdout + r.stderr).splitlines():
        if "Task ID" in ln: tid = ln.split(":")[-1].strip()
    if not tid: log(f"WARN submit noid: {r.stdout.strip()} {r.stderr.strip()}")
    return tid

def main():
    global SRC_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--src-path", required=True)
    ap.add_argument("--dropdir", required=True);  ap.add_argument("--outdir", required=True)
    ap.add_argument("--tmin", type=int, required=True); ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--kstride", type=int, default=1)
    ap.add_argument("--shards", type=int, default=3); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--min-free-gb", type=float, default=800.); ap.add_argument("--max-pending", type=int, default=72)
    ap.add_argument("--poll", type=int, default=20)
    a = ap.parse_args()
    SRC_BASE = a.src_path.rstrip("/")
    os.makedirs(a.dropdir, exist_ok=True); os.makedirs(a.outdir, exist_ok=True); os.makedirs("/data/logs", exist_ok=True)

    log(f"listing {a.src_path} ...")
    allf   = sorted(globus_ls(a.src_path))
    inrng  = [f for f in allf if want(f, a.tmin, a.tmax)]
    strided = inrng[::a.kstride] if a.kstride > 1 else inrng
    # purge stale pwat-less npz so they regenerate AND don't falsely count as done
    purged = 0
    for fn in strided:
        op = outpath(a.outdir, fn)
        if os.path.exists(op) and not has_pwat(op):
            try: os.remove(op); purged += 1
            except Exception: pass
    todo = [fn for fn in strided if not os.path.exists(outpath(a.outdir, fn))]
    n_expect = len(strided)
    log(f"{len(allf)} src, {len(inrng)} in {a.tmin}-{a.tmax}, kstride={a.kstride} -> {n_expect} target; "
        f"purged {purged} pwat-less; {len(todo)} to fetch")

    procs = []
    for i in range(a.shards):
        lf = open(f"/data/logs/3d_{a.scenario}_shard{i}.log", "a")
        p = subprocess.Popen([PYEXE, "-u", EXTRACT, "--indir", a.dropdir, "--outdir", a.outdir,
                              "--watch", "--delete", "--poll", "30", "--shard", f"{i}/{a.shards}"],
                             stdout=lf, stderr=subprocess.STDOUT)
        procs.append((p, lf))
    log(f"launched {a.shards} extract shards")

    def n_out(): return sum(1 for fn in strided if os.path.exists(outpath(a.outdir, fn)))  # cheap: new npz all have pwat

    i = 0; bn = 0
    while i < len(todo):
        pend = len(glob.glob(f"{a.dropdir}/*.nc")); fg = free_gb(a.dropdir)
        if fg < a.min_free_gb or pend > a.max_pending:
            log(f"throttle: free={fg:.0f}G pending={pend} sent={i}/{len(todo)} out={n_out()}/{n_expect}")
            time.sleep(a.poll); continue
        batch = todo[i:i+a.batch]; bn += 1
        tid = submit_batch(batch, a.dropdir, a.scenario, f"b{bn:04d}")
        i += len(batch)
        log(f"submit b{bn} ({len(batch)}f {i}/{len(todo)}) tid={tid[:12]} free={fg:.0f}G pending={pend} out={n_out()}/{n_expect}")
        time.sleep(1.5)

    log("all batches submitted; draining...")
    stall = 0; last = -1
    while True:
        outs = n_out(); pend = len(glob.glob(f"{a.dropdir}/*.nc"))
        if outs >= n_expect and pend == 0: log(f"DONE {a.scenario}: {outs}/{n_expect}"); break
        if outs == last: stall += 1
        else: stall = 0; last = outs
        if stall % 10 == 1: log(f"drain: {outs}/{n_expect} pending_raw={pend} free={free_gb(a.dropdir):.0f}G")
        if stall > 240: log(f"STALL >80min at {outs}/{n_expect} pend={pend}; exit drain"); break
        time.sleep(a.poll)

    for p, lf in procs: p.send_signal(signal.SIGINT)
    time.sleep(3)
    for p, lf in procs:
        try: p.terminate()
        except Exception: pass
        lf.close()
    log(f"{a.scenario} controller exit ({n_out()}/{n_expect})")

if __name__ == "__main__":
    main()
