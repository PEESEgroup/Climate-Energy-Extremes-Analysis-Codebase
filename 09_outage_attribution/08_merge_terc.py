"""Merge the per-shard trait files 07_terc.py writes when it is run as k/n.

The six traits are independent fits on one shared panel. Running them in one process took six
hours; two processes at three traits each halve that, and the box has room for two because one fit
peaks near 44 GB. Each shard writes tercile_gaps_v2_s<k>.json and this joins them into the single
file every consumer already reads. It refuses to write a partial merge.
"""
import json, glob, sys

OUT = "/data/equity_cost/analysis/attrib/tercile_gaps_v2.json"
EXPECT = ["median age", "poverty", "minority share", "income", "rurality", "undergrounding"]
files = sorted(glob.glob(OUT.replace(".json", "_s*.json")))
if not files:
    raise SystemExit("no shard files next to %s" % OUT)
M = {}
for f in files:
    d = json.load(open(f))
    dup = set(M) & set(d)
    if dup:
        raise SystemExit("shards overlap on %s; k/n was not a partition" % sorted(dup))
    M.update(d)
    print("  %-58s %d traits" % (f, len(d)))
missing = [t for t in EXPECT if not any(t == k or k.startswith(t) for k in M)]
if missing:
    raise SystemExit("merged %d traits but %s are absent; do not use a partial merge" % (len(M), missing))
json.dump(M, open(OUT, "w"), indent=1)
print("wrote %s with %d traits" % (OUT, len(M)))
