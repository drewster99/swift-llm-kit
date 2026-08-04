#!/usr/bin/env python3
"""Migrate probe records to the current ProbeRecord schema (v3).

Background
----------
"Effort" was two different things wearing one name:

  * GENERAL effort   — Anthropic's `output_config.effort`, which applies even with reasoning off.
  * REASONING effort — `reasoning_effort` on OpenAI-compatible endpoints.

`ModelProfile` stored one `effortLevels` dict, so what a record MEANT depended on which provider
produced it. The prober sends whichever parameter that provider actually accepts, so:

    Anthropic  -> the measurement was of GENERAL effort
    everyone else -> the measurement was of REASONING effort

That is the whole migration rule, and it is exhaustive for the corpus: the only providers that ever
recorded a non-empty ladder are Anthropic and OpenRouter.

Why every record must be rewritten, not just the non-empty ones
---------------------------------------------------------------
`effortLevels` was NON-OPTIONAL with an explicit CodingKey, so every record carries the key —
including the empty ones. `ProbeRecordStore` decodes with `try?` and SKIPS failures silently, so a
record left in the old shape does not error: it vanishes, taking every other finding in it (chat,
vision, tool calling, limits) with it. Partial migration would quietly delete the empirical layer.

Safety
------
* Refuses to run while AgentSmith is running (it would overwrite from memory).
* Writes a timestamped backup of every directory it touches before changing anything.
* Verifies afterwards that every record is at the current schema and the count is unchanged,
  restoring the backup on ANY failure — including a malformed file that the migration pass
  skipped, which would otherwise crash verification after valid files were already rewritten.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ANTHROPIC_PROVIDER_IDS = {"builtin.anthropic"}
ANTHROPIC_API_TYPE = "anthropic"

OLD_KEY = "effortLevels"
GENERAL_KEY = "generalEffortLevels"
REASONING_KEY = "reasoningEffortLevels"
NEW_SCHEMA_VERSION = 3
CAPABILITY_FINDINGS_KEY = "capabilityFindings"


def target_key(record: dict) -> str:
    """Which construct a record's ladder actually measured.

    Anthropic emits `output_config.effort` unconditionally, so its probe measured GENERAL effort.
    Every other backend is flag-gated on `reasoning_effort`, and the runner forces that raw
    parameter to measure it — so those records describe REASONING effort.

    Keyed on `key.apiType`, NOT on the provider id. A USER-CREATED Anthropic provider has a UUID
    for an id, so an id-based rule silently filed its general-effort ladder as reasoning effort —
    the one classification error this migration can make, and a silent one.  The record carries the
    apiType precisely because it is the authoritative answer.
    """
    api_type = (record.get("key") or {}).get("apiType")
    if api_type:
        return GENERAL_KEY if api_type == ANTHROPIC_API_TYPE else REASONING_KEY
    # Pre-key records: fall back to the built-in id, which is all they carry.
    return GENERAL_KEY if record.get("providerID") in ANTHROPIC_PROVIDER_IDS else REASONING_KEY


def migrate_profile(profile: dict, record: dict) -> bool:
    """Brings one profile up to the current schema. Returns True if anything changed.

    Idempotent and step-wise, so a corpus at mixed versions converges in one pass.
    """
    changed = False

    # v1 -> v2: one ladder became two, split by which parameter the prober actually sent.
    if OLD_KEY in profile:
        ladder = profile.pop(OLD_KEY)
        key = target_key(record)
        # Both keys must exist afterwards: they are non-optional on the Swift side, so a missing
        # one fails the decode exactly as the old key's absence would.
        profile[GENERAL_KEY] = ladder if key == GENERAL_KEY else {}
        profile[REASONING_KEY] = ladder if key == REASONING_KEY else {}
        changed = True

    # v2 -> v3: capabilityFindings is non-optional, so every record must carry it. Empty is the
    # honest value — these capabilities have never been probed on any existing record.
    # (maxThinkingBudgetTokens is optional and needs no backfill.)
    if CAPABILITY_FINDINGS_KEY not in profile:
        profile[CAPABILITY_FINDINGS_KEY] = {}
        changed = True

    return changed


def migrate_record(record: dict) -> bool:
    changed = migrate_profile(record.get("profile", {}), record)
    if changed:
        record["schemaVersion"] = NEW_SCHEMA_VERSION
    return changed


def agent_smith_running() -> bool:
    return subprocess.run(["pgrep", "-x", "AgentSmith"], capture_output=True).returncode == 0


def backup(path: Path, stamp: str) -> Path:
    dest = path.with_name(f"{path.name}.bak-effort-split-{stamp}")
    if path.is_dir():
        shutil.copytree(path, dest)
    else:
        shutil.copy2(path, dest)
    return dest


def migrate_directory(directory: Path, stamp: str, dry_run: bool) -> tuple[int, int]:
    files = sorted(directory.glob("*.json"))
    changed = 0
    if not dry_run and files:
        print(f"  backup -> {backup(directory, stamp).name}")
    for f in files:
        try:
            record = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  !! unreadable, left alone: {f.name}")
            continue
        # Called ONCE: it mutates in place, so a second call sees an already-migrated record and
        # returns False — which made every dry run report zero changes.
        if migrate_record(record):
            changed += 1
            if not dry_run:
                f.write_text(json.dumps(record, indent=2))
    return len(files), changed


def migrate_bundled(path: Path, stamp: str, dry_run: bool) -> tuple[int, int]:
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload.get("records", [])
    changed = sum(1 for r in records if migrate_record(r))
    if changed and not dry_run:
        print(f"  backup -> {backup(path, stamp).name}")
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return len(records), changed


def verify_directory(directory: Path, expected_total: int) -> None:
    files = sorted(directory.glob("*.json"))
    assert len(files) == expected_total, f"record count changed: {len(files)} != {expected_total}"
    for f in files:
        record = json.loads(f.read_text())
        profile = record.get("profile", {})
        assert OLD_KEY not in profile, f"{f.name} still carries {OLD_KEY}"
        assert GENERAL_KEY in profile and REASONING_KEY in profile, f"{f.name} missing an effort key"
        assert CAPABILITY_FINDINGS_KEY in profile, f"{f.name} missing {CAPABILITY_FINDINGS_KEY}"
        assert record.get("schemaVersion") == NEW_SCHEMA_VERSION, f"{f.name} not at v{NEW_SCHEMA_VERSION}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes-dir", type=Path, help="Local probe record directory")
    parser.add_argument("--bundled", type=Path, help="bundled_probe_records.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and agent_smith_running():
        print("AgentSmith is running — quit it first (it would overwrite these from memory).")
        return 1

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.bundled:
        print(f"bundled: {args.bundled}")
        total, changed = migrate_bundled(args.bundled, stamp, args.dry_run)
        print(f"  {changed} of {total} records migrated")

    if args.probes_dir:
        print(f"probes: {args.probes_dir}")
        total, changed = migrate_directory(args.probes_dir, stamp, args.dry_run)
        print(f"  {changed} of {total} records migrated")
        if not args.dry_run:
            # Restore on failure rather than leaving a half-migrated directory: the store SKIPS
            # records it cannot decode, so a partial migration silently deletes findings.
            try:
                verify_directory(args.probes_dir, total)
            except Exception as failure:   # AssertionError, JSONDecodeError, OSError — all restore
                backup_dir = args.probes_dir.with_name(f"{args.probes_dir.name}.bak-effort-split-{stamp}")
                print(f"  VERIFICATION FAILED: {failure}")
                if backup_dir.is_dir():
                    shutil.rmtree(args.probes_dir)
                    shutil.copytree(backup_dir, args.probes_dir)
                    print(f"  restored from {backup_dir.name}")
                return 1
            print("  verified: every record is at the current schema, count unchanged")

    return 0


if __name__ == "__main__":
    sys.exit(main())
