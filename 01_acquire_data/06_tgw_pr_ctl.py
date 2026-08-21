"""Disk-guarded, batched Globus->extract controller for ONE TGW precip/snow stream.
Streams weekly .nc from a Globus source dir into a local drop dir in BATCHES, throttling on free-disk and
pending-file caps so raw never piles up and the disk never fills; N parallel 08_process_tgw_pr.py shards extract+delete raw.
Idempotent: source files whose output npz already exists are skipped (cheap restart). Runs one stream to
completion (all transferred + drained), then exits. An orchestrator calls it per scenario, sequentially.

  python 06_tgw_pr_ctl.py --scenario historical \
      --src-path /historical_1980_2019/hourly --dropdir /data/tgw_raw/historical \
      --outdir /data/tgw_precip/historical --y0 1980 --y1 2019 --tmin 1980 --tmax 2019 \
      --shards 12 --batch 60
"""
import os, re, sys, glob, time, subprocess, argparse, signal

SRC_EP = "9392aa20-799e-4630-acb6-7e6a4ed5b795"   # tgw-wrf-conus (source)
DST_EP = os.environ.get("GLOBUS_DST_EP", "")    # your own Globus endpoint UUID
GLOBUS = "/home/ubuntu/venv/bin/globus"
PYEXE  = "/data/tellenv/bin/python"
EXTRACT = "/home/ubuntu/code/08_process_tgw_pr.py"
DATE_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})_")

def log(m): print(f"[ctl {time.strftime('%H:%M:%S')}] {m}", flush=True)

def free_gb(path):
    s = os.statvfs(path); return s.f_bavail * s.f_frsize / 1e9

def globus_ls(src_path):
    out = subprocess.run([GLOBUS, "ls", f"{SRC_EP}:{src_path}/"], capture_output=True, text=True, timeout=120)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip().endswith(".nc")]

def want(fn, tmin, tmax):
    m = DATE_RE.search(fn)
    if not m: return False
    y, mo = int(m.group(1)), int(m.group(2))
    return (tmin <= y <= tmax) or (y == tmin - 1 and mo == 12)   # +Dec spillover week of the year before tmin

def submit_batch(items, dropdir, scenario, label):
    """items: list of basenames. Writes a batch file (ABSOLUTE src/dst paths) and submits one globus task."""
    lines = [f"{SRC_BASE}/{it} {os.path.join(dropdir, it)}" for it in items]   # absolute src + absolute dst
    bf = f"/tmp/pr_batch_{scenario}_{label}.txt"
    with open(bf, "w") as fh: fh.write("\n".join(lines) + "\n")
    r = subprocess.run([GLOBUS, "transfer", SRC_EP, DST_EP,
                        "--batch", bf, "--label", f"pr-{scenario}-{label}",
                        "--sync-level", "exists"], capture_output=True, text=True, timeout=120)
    tid = ""
    for ln in (r.stdout + r.stderr).splitlines():
        if "Task ID" in ln: tid = ln.split(":")[-1].strip()
    if not tid: log(f"WARN submit noid: {r.stdout.strip()} {r.stderr.strip()}")
    return tid

def main():
    global SRC_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--src-path", required=True)
    ap.add_argument("--dropdir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--y0", type=int, required=True); ap.add_argument("--y1", type=int, required=True)
    ap.add_argument("--tmin", type=int, required=True); ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--shards", type=int, default=12); ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--min-free-gb", type=float, default=1500.); ap.add_argument("--max-pending", type=int, default=250)
    ap.add_argument("--poll", type=int, default=20)
    a = ap.parse_args()
    SRC_BASE = a.src_path.rstrip("/")
    os.makedirs(a.dropdir, exist_ok=True); os.makedirs(a.outdir, exist_ok=True)

    # 1) enumerate + filter source, drop already-extracted (idempotent restart)
    log(f"listing {a.src_path} ...")
    allf = globus_ls(a.src_path)
    todo = []
    for fn in sorted(allf):
        if not want(fn, a.tmin, a.tmax): continue
        outp = os.path.join(a.outdir, f"tgw_pr_{a.scenario}_{fn.replace('.nc','')}.npz")
        if os.path.exists(outp): continue           # already extracted
        todo.append(fn)
    log(f"{len(allf)} source .nc -> {len(todo)} to fetch (scenario={a.scenario}, years {a.tmin}-{a.tmax}, out={a.outdir})")
    n_expect = len([f for f in allf if want(f, a.tmin, a.tmax)])

    # 2) launch extractor shards (daemons)
    procs = []
    for i in range(a.shards):
        lf = open(f"/data/logs/pr_{a.scenario}_shard{i}.log", "a")
        p = subprocess.Popen([PYEXE, EXTRACT, "--scenario", a.scenario, "--indir", a.dropdir,
                              "--outdir", a.outdir, "--y0", str(a.y0), "--y1", str(a.y1),
                              "--watch", "--delete", "--poll", "30", "--shard", f"{i}/{a.shards}"],
                             stdout=lf, stderr=subprocess.STDOUT)
        procs.append((p, lf))
    log(f"launched {a.shards} extractor shards")

    # 3) batched submit with disk + pending guard
    i = 0; bn = 0
    while i < len(todo):
        pend = len(glob.glob(f"{a.dropdir}/*.nc")); fg = free_gb(a.dropdir)
        outs = len(glob.glob(f"{a.outdir}/tgw_pr_{a.scenario}_*.npz"))
        inflight = i - outs         # submitted-ahead minus produced (raw on disk OR in transit) -> true cap
        if fg < a.min_free_gb or pend > a.max_pending or inflight > a.max_pending:
            log(f"throttle: free={fg:.0f}G pending={pend} inflight={inflight} outputs={outs}/{n_expect} (waiting)")
            time.sleep(a.poll); continue
        batch = todo[i:i+a.batch]; bn += 1
        tid = submit_batch(batch, a.dropdir, a.scenario, f"b{bn:04d}")
        i += len(batch)
        log(f"submitted batch {bn} ({len(batch)} files, {i}/{len(todo)}) tid={tid[:12]} free={fg:.0f}G pending={pend} outputs={outs}/{n_expect}")
        time.sleep(2)

    # 4) drain: wait until all expected outputs present (or no progress for a long while)
    log("all batches submitted; draining...")
    stall = 0; last = -1
    while True:
        outs = len(glob.glob(f"{a.outdir}/tgw_pr_{a.scenario}_*.npz")); pend = len(glob.glob(f"{a.dropdir}/*.nc"))
        if outs >= n_expect and pend == 0:
            log(f"DONE stream {a.scenario}: outputs={outs}/{n_expect}"); break
        if outs == last: stall += 1
        else: stall = 0; last = outs
        if stall % 10 == 1: log(f"draining: outputs={outs}/{n_expect} pending_raw={pend} free={free_gb(a.dropdir):.0f}G")
        if stall > 180:      # ~1h no progress -> give up gracefully (leftover files logged)
            log(f"STALL >1h at outputs={outs}/{n_expect} pending={pend}; exiting drain"); break
        time.sleep(a.poll)

    # 5) stop shards
    for p, lf in procs:
        p.send_signal(signal.SIGINT)
    time.sleep(3)
    for p, lf in procs:
        try: p.terminate()
        except Exception: pass
        lf.close()
    log(f"stream {a.scenario} controller exit (outputs={len(glob.glob(f'{a.outdir}/tgw_pr_{a.scenario}_*.npz'))}/{n_expect})")

if __name__ == "__main__":
    main()
