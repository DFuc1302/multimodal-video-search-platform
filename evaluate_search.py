"""Compute per-session Recall@K/MRR from logs and independently judged qrels.

Qrels JSON: [{"query_id": "<hash from log>", "unit": "frame",
              "relevant": ["L01_V001/000123.jpg"], "session_id": "<optional>"}]
For ASR use unit=segment and IDs video_id@start-end (two decimal places).
For Video Match use unit=video and IDs such as L01_V001.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def retrieval_metrics(ranking, relevant, ks=(1, 5, 10, 50, 100)):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Relevant judgments must be nonempty")
    def item_id(value):
        if isinstance(value, dict):
            return value.get("file") or value.get("id") or value.get("video_id")
        return value
    ranked_ids = [item_id(value) for value in ranking]
    # Missing pages use None placeholders: they never become relevant hits.
    first = next((i for i, value in enumerate(ranked_ids, 1) if value in relevant), None)
    return {**{f"recall@{k}": len(set(ranked_ids[:k]) & relevant)/len(relevant) for k in ks},
            "mrr": 1/first if first else 0.0}


def percentile(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q*len(values))-1)]


def evaluate(events, qrels, ks=(1, 5, 10, 50, 100)):
    if not isinstance(qrels, list):
        raise ValueError("Qrels must be a JSON list")
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("K values must be positive integers")
    judgments = {}
    for row in qrels:
        if not isinstance(row, dict):
            raise ValueError("Each qrel must be an object")
        if not row.get("query_id") or not row.get("unit"):
            raise ValueError("Each qrel needs query_id and unit")
        if not isinstance(row.get("relevant"), list) or not row["relevant"]:
            raise ValueError("Each qrel needs independently judged relevant IDs")
        if any(not isinstance(item, (str, int)) or not str(item) for item in row["relevant"]):
            raise ValueError("Relevant IDs must be non-empty strings or integers")
        intervals = row.get("relevant_intervals", [])
        if not isinstance(intervals, list):
            raise ValueError("relevant_intervals must be a list")
        for interval in intervals:
            try:
                start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
                video_id = interval["video_id"]
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError("Each relevant interval needs video_id and numeric start_seconds/end_seconds") from exc
            if not video_id or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("Relevant intervals must have a video ID and finite non-negative ordered bounds")
        key = (row.get("session_id"), row["query_id"], row["unit"])
        if key in judgments:
            raise ValueError("Duplicate qrel identity")
        judgments[key] = row
    searches = [e for e in events if e.get("event") == "search"]
    searches.sort(key=lambda e: e.get("at", ""))
    latest = {}
    by_session = defaultdict(list)
    for event in searches:
        by_session[event["session_id"]].append(event)
        if event["status"] != 200:
            continue
        key = (event["session_id"], event["query_id"], event["unit"], event.get("configuration_id", "unknown"))
        offset = max(0, int(event.get("rank_offset", 0)))
        if offset == 0 or key not in latest:
            latest[key] = {"ranking": [], "offset_zero": offset == 0, "event": event}
        group = latest[key]
        end = offset+len(event["ranking"])
        group["ranking"].extend([None]*max(0, end-len(group["ranking"])))
        group["ranking"][offset:end] = event["ranking"]
        group["event"] = event
    details = []
    for key, group in latest.items():
        judgment = judgments.get(key[:3], judgments.get((None, key[1], key[2])))
        if judgment is None or not group["offset_zero"]:
            continue
        # Do not infer metrics across pages that were never retrieved.
        if None in group["ranking"]:
            continue
        relevant = judgment["relevant"]
        metrics = retrieval_metrics(group["ranking"], relevant, ks)
        # Optional temporal judgments are intervals in seconds. A ranked frame
        # is relevant when its video matches and its timestamp falls in an
        # independently judged interval (with an optional tolerance).
        intervals = judgment.get("relevant_intervals", [])
        if intervals:
            try:
                tolerance = float(judgment.get("temporal_tolerance_seconds", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("temporal_tolerance_seconds must be a finite non-negative number") from exc
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("temporal_tolerance_seconds must be a finite non-negative number")
            hits = []
            for rank, value in enumerate(group["ranking"], 1):
                if not isinstance(value, dict):
                    continue
                stamp = value.get("timestamp_seconds")
                video = value.get("video_id")
                if stamp is None:
                    continue
                try:
                    stamp = float(stamp)
                except (ValueError, TypeError):
                    continue
                if any(i.get("video_id") == video and
                       float(i["start_seconds"]) - tolerance <= stamp <= float(i["end_seconds"]) + tolerance
                       for i in intervals):
                    hits.append(rank)
            metrics["temporal_recall@10"] = int(any(rank <= 10 for rank in hits))
            metrics["temporal_mrr"] = 1 / hits[0] if hits else 0.0
        details.append({"session_id": key[0], "query_id": key[1], "unit": key[2], "configuration_id": key[3],
                        "query_type": judgment.get("query_type", "unspecified"),
                        "topic": judgment.get("topic", "unspecified"),
                        "retrieved": len(group["ranking"]),
                        "metrics": metrics})
    reports = []
    for session, rows in by_session.items():
        # Keep retrieval units separate: a segment rank is not a frame rank.
        for unit, configuration in sorted({(e["unit"], e.get("configuration_id", "unknown")) for e in rows}):
            selected = [d for d in details if d["session_id"] == session and d["unit"] == unit and d["configuration_id"] == configuration]
            runs = [e for e in rows if e["unit"] == unit and e.get("configuration_id", "unknown") == configuration]
            latency = [e["latency_ms"] for e in runs]
            names = [f"recall@{k}" for k in ks]+["mrr"]
            reports.append({"session_id": session, "unit": unit, "configuration_id": configuration, "requests": len(runs),
                            "failed_requests": sum(e["status"] >= 400 for e in runs),
                            "no_result_requests": sum(int(e.get("total", 0) or 0) == 0 and e["status"] == 200 for e in runs),
                            "no_result_rate": (sum(int(e.get("total", 0) or 0) == 0 and e["status"] == 200 for e in runs) /
                                               max(1, sum(e["status"] == 200 for e in runs))),
                            "judged_queries": len(selected),
                            "unjudged_or_incomplete_queries": sum(k[0] == session and k[2] == unit and k[3] == configuration for k in latest)-len(selected),
                            "latency_p50_ms": percentile(latency, .5), "latency_p95_ms": percentile(latency, .95),
                            "metrics": {name: sum(d["metrics"][name] for d in selected)/len(selected)
                                        for name in names} if selected else None})
    strata = []
    for config in sorted({d["configuration_id"] for d in details}):
        for field in ("query_type", "topic"):
            for value in sorted({d[field] for d in details if d["configuration_id"] == config}):
                rows = [d for d in details if d["configuration_id"] == config and d[field] == value]
                names = sorted({metric for row in rows for metric in row["metrics"]})
                strata.append({"configuration_id": config, "stratified_by": field, "value": value,
                               "queries": len(rows),
                               "metrics": {name: sum(row["metrics"][name] for row in rows if name in row["metrics"])/
                                           sum(name in row["metrics"] for row in rows) for name in names}})
    return {"sessions": reports, "queries": details, "strata": strata,
            "method": "macro average of last successful retrieval per session/query/unit/configuration; independent qrels only",
            "mrr_scope": "observed ranking; missing relevant IDs count as zero",
            "shortlist_is_ground_truth": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True,
                        help="Frozen benchmark qrels; include query_type/topic and judged relevant IDs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [args.logs] if args.logs.is_file() else sorted(args.logs.glob("search-*.jsonl"))
    events = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON log: {path.name}:{number}") from exc
    raw_qrels = args.qrels.read_bytes()
    qrels = json.loads(raw_qrels)
    report = evaluate(events, qrels)
    report["benchmark_sha256"] = hashlib.sha256(raw_qrels).hexdigest()
    report["benchmark_queries"] = len(qrels)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output+"\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
