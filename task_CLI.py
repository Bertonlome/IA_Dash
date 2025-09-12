#!/usr/bin/env python3
"""
CLI-only: Task management via command line. No Dash/web code.
"""

from typing import Optional, List
from enum import Enum

from task import (
    Task, Performer, Supporter, InformationItem, TeamingElement,
    ExecutionType, Modality, TimeConstraintType, ReliabilityLevel,
    load_tasks, save_tasks_json, export_tasks_csv, JSON_PATH
)


# =====================
# Interactive CLI (add)
# =====================

def prompt_enum(prompt: str, enum_cls: Enum, default: Optional[str] = None) -> Enum:
    """
    Prompt user to pick an enum value. Shows choices as 'value'.
    """
    choices = [e.value for e in enum_cls]
    default_str = f" [{default}]" if default else ""
    while True:
        ans = input(f"{prompt} {choices}{default_str}: ").strip().lower()
        if not ans and default:
            ans = default
        if ans in choices:
            return enum_cls(ans)
        print(f"Please choose one of: {choices}")

def prompt_float(prompt: str, default: Optional[float] = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    Prompt for a float with optional bounds and default.
    """
    default_str = f" [{default}]" if default is not None else ""
    while True:
        ans = input(f"{prompt}{default_str}: ").strip()
        if ans == "" and default is not None:
            return default
        try:
            val = float(ans)
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a number.")

def prompt_optional_float(prompt: str, default: Optional[float] = None) -> Optional[float]:
    """
    Prompt for an optional float (empty input returns None).
    """
    default_str = f" [{default}]" if default is not None else " [empty]"
    while True:
        ans = input(f"{prompt}{default_str}: ").strip()
        if ans == "":
            return None if default is None else default
        try:
            return float(ans)
        except ValueError:
            print("Please enter a number or leave blank.")

def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_str = " [Y/n]" if default else " [y/N]"
    ans = input(f"{prompt}{default_str}: ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")

def prompt_list_information(label: str) -> List[InformationItem]:
    """
    Repeatedly prompt for InformationItem entries.
    """
    items: List[InformationItem] = []
    print(f"\nAdd {label} (leave name empty to stop):")
    while True:
        name = input("  name: ").strip()
        if not name:
            break
        mod = prompt_enum("  modality", Modality, default="other")
        notes = input("  notes (optional): ").strip() or None
        items.append(InformationItem(name=name, modality=mod, notes=notes))
    return items

def prompt_list_teaming(label: str) -> List[TeamingElement]:
    """
    Repeatedly prompt for TeamingElement entries.
    """
    items: List[TeamingElement] = []
    print(f"\nAdd {label} (leave 'target' empty to stop):")
    while True:
        target = input("  target: ").strip()
        if not target:
            break
        what = input("  what: ").strip()
        mod = prompt_enum("  modality", Modality, default="other")
        notes = input("  notes (optional): ").strip() or None
        items.append(TeamingElement(target=target, what=what, modality=mod, notes=notes))
    return items

def cmd_add() -> None:
    """
    Interactive 'add' command. Prompts for a minimal viable Task.
    You can always edit tasks.json later for advanced details.
    """
    print("\n=== Add a new Task ===")
    name = input("task.name: ").strip()
    procedure = input("task.procedure (optional): ").strip() or None

    criticality = prompt_float("task.criticality (0 = facultative .. 1 = critical)", default=0.5, min_val=0.0, max_val=1.0)

    exec_type = prompt_enum("task.execution_type", ExecutionType, default=ExecutionType.ONE_OFF.value)
    tctype = prompt_enum("task.time_constraint.type", TimeConstraintType, default=TimeConstraintType.NONE.value)
    tcs = None
    if tctype != TimeConstraintType.NONE:
        tcs = prompt_optional_float("task.time_constraint.s (after how many seconds it can be considered miss or failed?)", default=None)

    # Performer
    print("\n--- Performer ---")
    p_kind = input("performer.kind (human/TARS/UGV/UAV...): ").strip().lower() or "human"
    p_rel = prompt_enum("performer.reliability", ReliabilityLevel, default=ReliabilityLevel.RED.value)
    p_time = prompt_float("performer.efficiency_time_required_s", default=60.0, min_val=0.0)
    p_familiarity = prompt_float("performer.familiarity (0 = routine .. 1 = never done before)", default=0.0, min_val=0.0, max_val=1.0)
    p_notes = input("performer.notes (optional): ").strip() or None
    performer = Performer(kind=p_kind, reliability=p_rel,
                          efficiency_time_required_s=p_time, familiarity=p_familiarity, notes=p_notes)

    # Supporter (optional)
    supporter = None
    if prompt_yes_no("Add a supporter?", default=False):
        print("\n--- Supporter ---")
        s_kind = input("supporter.kind (human/TARS/UGV/UAV...): ").strip().lower() or "automation"
        s_cap = prompt_enum("supporter.capacity", ReliabilityLevel, default=ReliabilityLevel.GREEN.value)
        s_notes = input("supporter.notes (optional): ").strip() or None
        supporter = Supporter(kind=s_kind, capacity=s_cap, notes=s_notes)

    # Lists
    info_reqs = prompt_list_information("information requirements")
    mem_reqs = prompt_list_information("memory requirements")
    observability = prompt_list_teaming("team observability")
    predictability = prompt_list_teaming("team predictability")
    directability = prompt_list_teaming("team directability")

    tags_str = input("\nTags (comma-separated, optional): ").strip()
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    notes = input("task.notes (optional): ").strip() or None

    task = Task(
        name=name,
        procedure=procedure,
        criticality=criticality,
        performer=performer,
        supporter=supporter,
        time_constraint_type=tctype,
        time_constraint_s=tcs,
        execution_type=exec_type,
        information_requirements=info_reqs,
        memory_requirements=mem_reqs,
        observability=observability,
        predictability=predictability,
        directability=directability,
        tags=tags,
        notes=notes
    )

    # Append to JSON and export CSV
    tasks = load_tasks()
    tasks.append(task)

    save_tasks_json(tasks)
    print(f"\nSaved {len(tasks)} task(s) to {JSON_PATH}")

    export_tasks_csv(tasks)


def cmd_export() -> None:
    """
    Rebuild CSV from JSON (useful after manual edits to tasks.json).
    """
    tasks = load_tasks()
    export_tasks_csv(tasks)


# === Entrypoint ===

if __name__ == "__main__":
    cmd_add()
