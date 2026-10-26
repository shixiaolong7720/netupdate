"""Read-only NPZ label audit; also usable without PyTorch.

Training class semantics follow the ID folder, as in the legacy trainer. Strict
mode detects inconsistent metadata before training. Explicit folder mode may
be used only after verifying that the folder is the intended ground truth;
every inconsistency remains recorded and no NPZ is rewritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def inspect_label_metadata(data, folder_cid):
    """Return all class_id/label conflicts, including malformed scalar values."""
    issues = []
    for field, expected in (("class_id", int(folder_cid)), ("label", int(folder_cid == 0))):
        if field not in data.files:
            continue  # Metadata is optional; folder labels remain required.
        try:
            value = np.asarray(data[field])
            stored = np.array2string(value, threshold=4, edgeitems=2)[:160]
            valid = value.size == 1 and value.dtype.kind in "iuf"
            number = float(value.reshape(-1)[0]) if valid else float("nan")
            valid = valid and math.isfinite(number) and number >= 0 and number.is_integer()
            if field == "label":
                valid = valid and number in (0, 1)
            if valid and int(number) == expected:
                continue
            kind = "invalid_metadata" if not valid else "class_id_mismatch"
            if valid and ((field == "label") or ((number == 0) != (folder_cid == 0))):
                kind = "wake_nonwake_conflict"
        except (TypeError, ValueError, OverflowError) as exc:
            stored, kind = f"unreadable: {exc}"[:160], "invalid_metadata"
        issues.append({"field": field, "stored_value": stored, "folder_expected": expected, "issue_type": kind})
    return issues


class LabelAudit:
    def __init__(self, policy="strict", report_path=None):
        if policy not in ("strict", "folder"):
            raise ValueError("label_policy must be 'strict' or 'folder'")
        self.policy = policy
        self.report_path = Path(report_path) if report_path is not None else None
        self.checked = self.conflicted_files = self.issue_count = 0
        self.counts = Counter()
        self.examples = []
        self._handle = self._writer = None
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.report_path.open("w", newline="", encoding="utf-8-sig")
            self._writer = csv.DictWriter(self._handle, fieldnames=(
                "path", "sid", "folder_class_id", "field", "stored_value", "folder_expected", "issue_type"))
            self._writer.writeheader()

    def record(self, data, folder_cid, path, sid):
        self.checked += 1
        issues = inspect_label_metadata(data, folder_cid)
        self.conflicted_files += bool(issues)
        for issue in issues:
            row = {"path": str(path), "sid": str(sid), "folder_class_id": int(folder_cid), **issue}
            self.issue_count += 1
            self.counts[issue["issue_type"]] += 1
            if len(self.examples) < 10:
                self.examples.append(row)
            if self._writer is not None:
                self._writer.writerow(row)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def summary(self):
        return {"policy": self.policy, "checked_files": self.checked,
                "conflicted_files": int(self.conflicted_files), "issue_count": self.issue_count,
                "issue_counts": dict(self.counts), "examples": self.examples,
                "report_path": str(self.report_path) if self.report_path is not None else None}

    def finish(self):
        self.close()
        result = self.summary()
        print(f"[Label audit] policy={self.policy} checked={self.checked} "
              f"conflicted_files={self.conflicted_files} issues={dict(self.counts)}", flush=True)
        if self.conflicted_files:
            first = self.examples[0]
            detail = (f"{first['path']}: stored {first['field']}={first['stored_value']}, "
                      f"folder expects {first['folder_expected']}")
            report = f" Full report: {self.report_path}." if self.report_path is not None else ""
            if self.policy == "strict":
                raise ValueError(
                    f"Label audit failed before training: {self.conflicted_files} conflicting NPZ files. "
                    f"{detail}.{report} Verify the original annotations. If ID folders are confirmed "
                    "ground truth and NPZ metadata is stale, explicitly pass --label-policy folder. "
                    "This never changes the source NPZs.")
            print(f"[Label audit WARNING] Explicit folder policy overrides inconsistent metadata. "
                  f"{detail}.{report} Verify folder ground truth, especially wake/nonwake conflicts.", flush=True)
        return result


def audit_root(root, report_path):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {root}")
    audit = LabelAudit("folder", report_path)
    try:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("ID_") or folder.name.casefold() == "id_25_noise":
                continue
            try:
                cid = int(folder.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            if cid < 0:
                raise ValueError(f"Negative folder class ID: {folder}")
            for path in sorted(folder.glob("*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    audit.record(data, cid, path, path.relative_to(root).as_posix())
                if audit.checked % 10000 == 0:
                    print(f"[Audit progress] {root.name}: {audit.checked} files checked", flush=True)
        if audit.checked == 0:
            raise ValueError(f"No usable NPZ files under {root}")
    finally:
        audit.close()
    result = audit.summary()
    result["root"] = str(root)
    result["policy"] = "audit_only_no_training"
    print(f"[Audit only] root={root} files={audit.checked} conflicts={audit.conflicted_files} "
          f"counts={dict(audit.counts)} report={report_path}", flush=True)
    for issue in audit.examples[:3]:
        print(f"  {issue['sid']}: {issue['field']}={issue['stored_value']} "
              f"expected={issue['folder_expected']} ({issue['issue_type']})", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit folder labels against NPZ metadata without changing training data")
    parser.add_argument("--root", action="append", required=True, help="Feature root with ID_* folders; repeat for multiple splits")
    parser.add_argument("--report-dir", default=None, help="New/empty directory for CSV and JSON reports")
    args = parser.parse_args(argv)
    output = Path(args.report_dir) if args.report_dir else Path(datetime.now().strftime("label_audit_%Y%m%d_%H%M%S_%f"))
    if output.exists() and any(output.iterdir()):
        parser.error(f"Report directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summaries = [audit_root(root, output / f"root_{index:02d}_label_conflicts.csv")
                 for index, root in enumerate(args.root)]
    (output / "label_audit_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit completed. Reports: {output.resolve()}", flush=True)
    return int(any(summary["conflicted_files"] for summary in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
