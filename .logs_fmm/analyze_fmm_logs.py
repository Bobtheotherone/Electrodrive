import os
import json
import math
import pathlib
import collections

def _read_text_auto(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def analyze_log(label: str, env_var: str) -> None:
    path_str = os.environ.get(env_var, "")
    if not path_str:
        print(f"[log-analyze] {label}: env var {env_var} not set, skipping.")
        return

    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"[log-analyze] {label}: file {path} does not exist, skipping.")
        return

    print(f"\n===== {label}: {path} =====")
    counts = collections.Counter()
    numeric_stats = collections.defaultdict(lambda: collections.defaultdict(list))

    total_lines = 0
    json_lines = 0

    text = _read_text_auto(path)

    for line in text.splitlines():
        total_lines += 1
        line = line.strip()
        if not line:
            continue

        try:
            rec = json.loads(line)
            json_lines += 1
        except Exception:
            if total_lines <= 5:
                print(f"[non-json] {line[:120]}")
            continue

        event = (
            rec.get("event")
            or rec.get("msg")
            or rec.get("message")
            or "<unknown>"
        )
        counts[event] += 1

        for k, v in rec.items():
            if isinstance(v, (int, float)):
                numeric_stats[event][k].append(float(v))

    print(
        f"[summary] total_lines={total_lines}, "
        f"json_lines={json_lines}, "
        f"unique_events={len(counts)}"
    )

    if not counts:
        print("[summary] No structured JSON events found in this log.")
        return

    print("\n[top events by count]")
    for ev, c in counts.most_common(15):
        print(f"  {ev:30s}  count={c}")

    interesting = [
        e for e in counts.keys()
        if any(key in e.lower() for key in ("fmm", "p2p", "bem", "error", "residual"))
    ]
    if not interesting:
        interesting = list(counts.keys())[:10]

    print("\n[numeric ranges for selected events]")
    for ev in interesting[:10]:
        stats = numeric_stats.get(ev)
        if not stats:
            continue
        print(f"  [{ev}]")
        for k, vals in stats.items():
            if not vals:
                continue
            vmin = min(vals); vmax = max(vals)
            if math.isfinite(vmin) and math.isfinite(vmax):
                print(f"    {k:20s}  min={vmin: .3e}  max={vmax: .3e}")
            else:
                print(f"    {k:20s}  min={vmin}  max={vmax}")
    print()

if __name__ == "__main__":
    analyze_log("sanity_suite", "FMM_SANITY_LOG")
    analyze_log("pytest_fmm_tests", "FMM_TESTS_LOG")
