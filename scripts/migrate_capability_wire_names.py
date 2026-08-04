#!/usr/bin/env python3
"""ONE-TIME migration: rewrite persisted ModelCapability keys to their clarified names.

Not part of the app, and deliberately not: this converges a fixed set of files ONCE, and a
migration living in the app is a startup cost paid forever to fix a state that exists for a day.
Run it manually, with the app quit, then ship the matching rawValue change.

Background
----------
`ModelCapability` is `String`-backed and its rawValues ARE the persisted keys. Several names were
ambiguous — `toolChoiceRequired` reads as "a tool_choice is required" rather than "the tool_choice
parameter accepts the value `required`" — so the Swift cases were renamed while the wire strings
stayed pinned to the old spellings, deliberately, so no record was orphaned in the meantime.

This closes that gap: the data moves to the new spellings, after which the pins in
`ModelCapabilities.swift` are updated to match and the legacy spellings cease to exist.

ORDER MATTERS. The app can read exactly one spelling at a time:

    1. quit the app
    2. run this script            (data: old -> new)
    3. update the pinned rawValues in ModelCapabilities.swift to the new names
    4. rebuild

Between 2 and 3 the app would not read the migrated keys, which is why they are one change.

Where the keys live
-------------------
* `probes/*.json`      -> `profile.capabilityFindings` (keys are rawValues)
* `model_catalog.json` -> each model's `capabilities` (keys are rawValues)

Deliberately NOT touched:
* `downloaded_overrides.json`, the app's `model_overrides.json`, and both bundled resources —
  verified to contain none of these keys. Rewriting a file with nothing to change risks a
  formatting-only diff on data the user owns.
* `ModelCapabilitiesOverride`'s property names, which are a separate wire surface that also
  carries none of these keys today.

Safety
------
* Refuses to run while AgentSmith is running — it would overwrite from memory.
* `--dry-run` reports exactly what would change and writes nothing.
* Writes a timestamped backup of every file it touches BEFORE changing anything, and restores
  from it if verification fails for any reason.
* Verifies afterwards that no legacy key survives, that no new key collided with an existing one,
  and that the record and model counts are unchanged.
* Idempotent: a second run finds nothing to do rather than double-applying.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: Legacy wire string -> clarified wire string. The Swift cases already use the new names; only the
#: pinned rawValues still say the old ones, and step 3 above changes those to match.
RENAMES: dict[str, str] = {
    "toolChoice": "toolChoiceSupported",
    "toolChoiceRequired": "toolChoiceSupportsValueRequired",
    "toolChoiceNone": "toolChoiceSupportsValueNone",
    "toolChoiceSpecificFunction": "toolChoiceSupportsNamedFunction",
    "structuredOutputJSONObject": "structuredOutputSupportsJSONObject",
    "responseSchema": "structuredOutputSupportsJSONSchema",
    "thinkingKeepAll": "thinkingSupportsKeepAll",
    "thinkingBudgetTokens": "thinkingSupportsTokenBudget",
    "strictToolDefinitions": "toolDefinitionsSupportStrict",
    "reasoningEnableable": "reasoningCanBeEnabled",
    "reasoningDisableable": "reasoningCanBeDisabled",
}


def rename_keys(mapping: dict) -> tuple[dict, int]:
    """Returns the mapping with legacy keys renamed, and how many were renamed.

    A legacy key whose NEW name is already present is left alone and reported by the caller's
    collision check rather than silently overwriting the newer value.
    """
    out, renamed = {}, 0
    for key, value in mapping.items():
        target = RENAMES.get(key)
        if target and target not in mapping:
            out[target] = value
            renamed += 1
        else:
            out[key] = value
    return out, renamed


def collisions(mapping: dict) -> list[str]:
    """Legacy keys that cannot be renamed because the new name is already taken."""
    return [k for k in mapping if k in RENAMES and RENAMES[k] in mapping]


def migrate_probe_record(record: dict) -> tuple[bool, list[str]]:
    profile = record.get("profile") or {}
    findings = profile.get("capabilityFindings")
    if not isinstance(findings, dict):
        return False, []
    clashes = collisions(findings)
    renamed_map, count = rename_keys(findings)
    if count:
        profile["capabilityFindings"] = renamed_map
    return count > 0, clashes


def migrate_catalog_model(model: dict) -> tuple[bool, list[str]]:
    capabilities = model.get("capabilities")
    if not isinstance(capabilities, dict):
        return False, []
    clashes = collisions(capabilities)
    renamed_map, count = rename_keys(capabilities)
    if count:
        model["capabilities"] = renamed_map
    return count > 0, clashes


def agent_smith_running() -> bool:
    return subprocess.run(["pgrep", "-x", "AgentSmith"], capture_output=True).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--support-dir", type=Path, required=True,
                        help="SwiftLLMKit/<bundle id> directory holding probes/ and model_catalog.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and agent_smith_running():
        print("AgentSmith is running — quit it first (it would overwrite these from memory).")
        return 1

    probes_dir = args.support_dir / "probes"
    catalog = args.support_dir / "model_catalog.json"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backups: list[tuple[Path, Path]] = []

    def back_up(path: Path) -> None:
        if args.dry_run or not path.exists():
            return
        dest = path.with_name(f"{path.name}.bak-capability-rename-{stamp}")
        shutil.copytree(path, dest) if path.is_dir() else shutil.copy2(path, dest)
        backups.append((path, dest))
        print(f"  backup -> {dest.name}")

    def restore() -> None:
        for original, backup in backups:
            if original.is_dir():
                shutil.rmtree(original)
                shutil.copytree(backup, original)
            else:
                shutil.copy2(backup, original)
            print(f"  restored {original.name}")

    all_clashes: list[str] = []

    # --- probe records -------------------------------------------------------------------------
    probe_files = sorted(probes_dir.glob("*.json")) if probes_dir.is_dir() else []
    changed_probes = 0
    if probe_files:
        back_up(probes_dir)
    for path in probe_files:
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  !! unreadable, left alone: {path.name}")
            continue
        changed, clashes = migrate_probe_record(record)
        all_clashes += [f"{path.name}: {c}" for c in clashes]
        if changed:
            changed_probes += 1
            if not args.dry_run:
                path.write_text(json.dumps(record, indent=2))
    print(f"probes: {changed_probes} of {len(probe_files)} records migrated")

    # --- model catalog -------------------------------------------------------------------------
    changed_models = 0
    total_models = 0
    if catalog.exists():
        payload = json.loads(catalog.read_text())
        models = payload if isinstance(payload, list) else payload.get("models", [])
        total_models = len(models)
        touched = False
        for model in models:
            changed, clashes = migrate_catalog_model(model)
            all_clashes += [f"catalog/{model.get('modelID','?')}: {c}" for c in clashes]
            if changed:
                changed_models += 1
                touched = True
        if touched and not args.dry_run:
            back_up(catalog)
            catalog.write_text(json.dumps(payload, indent=2))
    print(f"catalog: {changed_models} of {total_models} models migrated")

    if all_clashes:
        print("\nCOLLISIONS — a legacy key whose new name was already present, left untouched:")
        for c in all_clashes[:20]:
            print(f"   {c}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    # --- verify --------------------------------------------------------------------------------
    try:
        survivors: list[str] = []
        for path in sorted(probes_dir.glob("*.json")) if probes_dir.is_dir() else []:
            findings = (json.loads(path.read_text()).get("profile") or {}).get("capabilityFindings") or {}
            survivors += [f"{path.name}:{k}" for k in findings if k in RENAMES]
        assert len(list(probes_dir.glob("*.json")) if probes_dir.is_dir() else []) == len(probe_files), \
            "probe record count changed"
        if catalog.exists():
            payload = json.loads(catalog.read_text())
            models = payload if isinstance(payload, list) else payload.get("models", [])
            assert len(models) == total_models, "catalog model count changed"
            for model in models:
                survivors += [f"catalog/{model.get('modelID','?')}:{k}"
                              for k in (model.get("capabilities") or {}) if k in RENAMES]
        assert not survivors, f"legacy keys survived: {survivors[:10]}"
    except Exception as failure:
        print(f"\nVERIFICATION FAILED: {failure}")
        restore()
        return 1

    print("\nverified: no legacy key survives, counts unchanged")
    print("NEXT: update the pinned rawValues in ModelCapabilities.swift to the new names, then rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
