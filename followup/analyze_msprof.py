"""Export allowlisted steady-state task/API timings from actual msprof output.

Usage: python analyze_msprof.py mindstudio_profiler_output app.stdout output
Excludes identifiers, original event arguments, metadata, warm-up and teardown.
"""
import collections
import csv
import datetime
from decimal import Decimal
import hashlib
import json
import pathlib
import sys

source, log, output = map(pathlib.Path, sys.argv[1:])
phases = []
for line in log.read_text().splitlines():
    try:
        item = json.loads(line)
    except ValueError:
        continue
    if item.get("test") == "profile_phase":
        phases.append(item)


def timestamp(item):
    text = item["utc"]
    whole = int(datetime.datetime.fromisoformat(text).timestamp())
    fraction = text.split(".", 1)[1].split("+", 1)[0] if "." in text else "0"
    return Decimal(whole) * 1_000_000 + Decimal("0." + fraction) * 1_000_000


begin, finish = timestamp(phases[0]), timestamp(phases[-1])
elapsed_us = float(finish - begin)
events, rows, groups = [], [], collections.defaultdict(list)
fields = ["OP Type", "Task Type", "Task Start Time(us)", "Task Duration(us)",
          "Input Shapes", "Input Data Types", "Output Shapes", "Output Data Types",
          "aicore_time(us)", "aic_mac_ratio", "aic_mte2_ratio", "aiv_time(us)",
          "aiv_vec_ratio", "aiv_mte2_ratio", "cube_utilization(%)"]
intervals = []
for row in csv.DictReader(next(source.glob("op_summary*.csv")).open()):
    start, duration = Decimal(row["Task Start Time(us)"].strip()), Decimal(row["Task Duration(us)"])
    if start + duration <= begin or start >= finish:
        continue
    end = min(finish, start + duration)
    start = max(start, begin)
    start, end = float(start - begin), float(end - begin)
    groups[row["OP Type"]].append(end - start)
    intervals.append((start, end))
    record = {key: row[key].strip() for key in fields}
    record["Task Start Time(us)"] = start
    record["Task Duration(us)"] = end - start
    rows.append(record)
    events.append({"name": row["Op Name"], "cat": "device-task", "ph": "X",
                   "pid": 1, "tid": 1, "ts": start, "dur": end - start,
                   "args": {"opType": row["OP Type"], "taskType": row["Task Type"]}})
api = collections.defaultdict(list)
trace_path = next(source.glob("msprof*.json"))
for row in json.loads(trace_path.read_text()):
    if row.get("ph") != "X" or not row.get("name", "").startswith("AscendCL@"):
        continue
    start, duration = Decimal(str(row["ts"])), Decimal(str(row["dur"]))
    if start + duration <= begin or start >= finish:
        continue
    end = min(finish, start + duration)
    start = max(start, begin)
    start, end = float(start - begin), float(end - begin)
    name = row["name"].split("@", 1)[1]
    api[name].append(end - start)
    events.append({"name": name, "cat": "ACL API", "ph": "X", "pid": 1,
                   "tid": 2, "ts": start, "dur": end - start, "args": {}})
for phase in phases:
    events.append({"name": phase["phase"] + (f"_{phase['step']}" if "step" in phase else ""),
                   "cat": "phase", "ph": "i", "s": "g", "pid": 1, "tid": 3,
                   "ts": float(timestamp(phase) - begin), "args": {}})
for tid, name in [(1, "NPU device tasks"), (2, "ACL API (nested calls)"), (3, "Phases")]:
    events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
                   "args": {"name": name}})
busy = 0
if intervals:
    left, right = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= right:
            right = max(right, end)
        else:
            busy += right - left
            left, right = start, end
    busy += right - left
total = sum(sum(values) for values in groups.values())
summary = {
    "device": "Ascend 910B4", "cann": "8.3.RC1", "tool": "MindStudio Profiler (msprof)",
    "model": "SmolLM-135M.Q8_0.gguf",
    "workload": "one warm cached 32-token prefill plus eight forced decode steps",
    "intervalMs": elapsed_us / 1000,
    "deviceTaskSumMs": total / 1000, "deviceTaskBusyUnionMs": busy / 1000,
    "deviceTaskBusyFraction": busy / elapsed_us,
    "ops": [{"name": name, "calls": len(values), "totalMs": sum(values) / 1000,
             "meanUs": sum(values) / len(values), "sharePercent": sum(values) / total * 100}
            for name, values in sorted(groups.items(), key=lambda item: -sum(item[1]))],
    "aclApi": [{"name": name, "calls": len(values), "totalMs": sum(values) / 1000}
               for name, values in sorted(api.items(), key=lambda item: -sum(item[1]))],
    "phases": phases,
    "rawTraceSha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    "limitations": ["ACL API durations include nested calls and device waits; not additive.",
                    "Device-task busy fraction excludes standalone copy activity and is not AICore utilization.",
                    "One profiled run; profiler overhead and logging are included in wall time.",
                    "Counter groups are per-task PipeUtilization; no HBM bandwidth roofline measured."]
}
output.mkdir(parents=True, exist_ok=True)
(output / "msprof-steady-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output / "ascend-steady-trace.json").write_text(json.dumps({"traceEvents": events,
                                                          "displayTimeUnit": "ms"}))
with (output / "msprof-op-steady.csv").open("w") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({key: summary[key] for key in ["intervalMs", "deviceTaskSumMs",
                                               "deviceTaskBusyUnionMs", "deviceTaskBusyFraction"]}))
print("ops", json.dumps(summary["ops"]))
print("ACL", json.dumps(summary["aclApi"][:15]))
