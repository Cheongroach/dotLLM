"""Export only CUDA timing records from an Nsight Systems SQLite database.

Usage: python analyze_nsys.py profile.sqlite profile.stdout output-directory
The source database can contain process environment and machine identifiers.
This exporter includes only allowlisted timing fields, never those tables.
"""
import collections
import datetime
import hashlib
import json
import pathlib
import sqlite3
import sys

database, log, destination = map(pathlib.Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
connection = sqlite3.connect(database)
connection.row_factory = sqlite3.Row
strings = dict(connection.execute("SELECT id, value FROM StringIds"))
epoch = connection.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
phases = []
for line in log.read_text().splitlines():
    try:
        value = json.loads(line)
    except ValueError:
        continue
    if value.get("test") == "profile_phase":
        phases.append(value)


def timestamp(value):
    instant = datetime.datetime.fromisoformat(value["utc"])
    return round(instant.timestamp() * 1_000_000_000) - epoch


begin, finish = timestamp(phases[0]), timestamp(phases[-1])
events = []
totals = {}


def read_events(table, kind, name_field=None):
    rows = []
    for row in connection.execute(f"SELECT * FROM {table} ORDER BY start"):
        if row["end"] <= begin or row["start"] >= finish:
            continue
        name = strings[row[name_field]] if name_field else f"memcpy_kind_{row['copyKind']}"
        start, end = max(row["start"], begin), min(row["end"], finish)
        args = {}
        if kind == "kernel":
            args = {key: row[key] for key in ["gridX", "gridY", "gridZ", "blockX",
                                             "blockY", "blockZ", "registersPerThread",
                                             "staticSharedMemory", "dynamicSharedMemory"]}
        elif kind == "memcpy":
            args = {"bytes": row["bytes"], "copyKind": row["copyKind"]}
        events.append({"name": name, "cat": kind, "ph": "X", "pid": 1,
                       "tid": {"kernel": 1, "memcpy": 2, "api": 3}[kind],
                       "ts": (start - begin) / 1000, "dur": (end - start) / 1000,
                       "args": args})
        rows.append((name, start, end))
    grouped = collections.defaultdict(list)
    for name, start, end in rows:
        grouped[name].append(end - start)
    total = sum(sum(values) for values in grouped.values())
    totals[kind] = {"totalMs": total / 1_000_000,
                    "groups": [{"name": name, "calls": len(values),
                                "totalMs": sum(values) / 1_000_000,
                                "meanUs": sum(values) / len(values) / 1000,
                                "sharePercent": 100 * sum(values) / total}
                               for name, values in sorted(grouped.items(),
                                                        key=lambda item: -sum(item[1]))]}
    return rows


kernels = read_events("CUPTI_ACTIVITY_KIND_KERNEL", "kernel", "shortName")
read_events("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")
read_events("CUPTI_ACTIVITY_KIND_RUNTIME", "api", "nameId")
intervals = sorted((start, end) for _, start, end in kernels)
busy = 0
if intervals:
    left, right = intervals[0]
    for start, end in intervals[1:]:
        if start <= right:
            right = max(right, end)
        else:
            busy += right - left
            left, right = start, end
    busy += right - left
for phase in phases:
    events.append({"name": phase["phase"] + (f"_{phase['step']}" if "step" in phase else ""),
                   "cat": "phase", "ph": "i", "s": "g", "pid": 1, "tid": 4,
                   "ts": (timestamp(phase) - begin) / 1000, "args": {}})
for kind, tid in [("GPU kernels", 1), ("GPU copies", 2), ("CUDA API", 3), ("Phases", 4)]:
    events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
                   "args": {"name": kind}})
summary = {
    "model": "SmolLM-135M.Q8_0.gguf", "device": "NVIDIA H20",
    "workload": "one warm cached 32-token prefill plus eight forced decode steps",
    "clockAlignment": "UTC application phase timestamps minus Nsight session epoch; sub-microsecond rounding",
    "intervalMs": (finish - begin) / 1_000_000,
    "kernelBusyUnionMs": busy / 1_000_000,
    "kernelBusyFraction": busy / (finish - begin),
    "limitations": [
        "Interval includes managed submission, logging, scheduling, CUDA synchronization and copies.",
        "Kernel busy fraction is timeline occupancy, not SM occupancy or hardware utilization.",
        "API durations overlap GPU execution; do not subtract or sum them as exclusive costs.",
        "Single profiled run includes profiler overhead; use unprofiled CLI runs for throughput.",
        "Initialization, load, first warm-up and destruction are excluded."
    ],
    "timings": totals, "phases": phases,
    "rawSqliteSha256": hashlib.sha256(database.read_bytes()).hexdigest(),
}
(destination / "nsys-steady-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(destination / "cuda-steady-trace.json").write_text(json.dumps({"traceEvents": events,
                                                              "displayTimeUnit": "ms"}))
print(json.dumps({key: summary[key] for key in ["intervalMs", "kernelBusyUnionMs",
                                               "kernelBusyFraction"]}))
for kind in totals:
    print(kind, json.dumps(totals[kind]))
