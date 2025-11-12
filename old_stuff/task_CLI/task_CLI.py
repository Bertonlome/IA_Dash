#!/usr/bin/env python3
"""
CLI-only: Task management via command line. No Dash/web code.
"""


from typing import Optional, List
from enum import Enum
try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import WordCompleter
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from task import *


# =====================
# Interactive CLI (add)
# =====================

def prompt_enum(prompt: str, enum_cls: Enum, default: Optional[str] = None) -> Enum:
    """
    Prompt user to pick an enum value. Shows choices as 'value'.
    Uses tab-completion if prompt_toolkit is available.
    """
    choices = [e.value for e in enum_cls]
    default_str = f" [{default}]" if default else ""
    while True:
        if PROMPT_TOOLKIT_AVAILABLE:
            completer = WordCompleter(choices, ignore_case=True)
            ans = pt_prompt(f"{prompt} {choices}{default_str}: ", completer=completer).strip().lower()
        else:
            ans = input(f"{prompt} {choices}{default_str}: ").strip().lower()
        if not ans and default:
            ans = default
        elif not ans and not default:
            return None
        for c in choices:
            if ans.lower() == c.lower():
                return enum_cls(c)
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
    mod = None
    items: List[InformationItem] = []
    print(f"\nAdd {label} == what info are needed to have to complete the task ? (leave name empty to stop):")
    while True:
        name = input("  info name: ").strip()
        if not name:
            break
        if label == "information requirements":
            mod = prompt_enum(" info modality or how can the human sense the information", InformationModality, default = InformationModality.VISUAL.value)
        notes = input("  notes (optional): ").strip() or None
        items.append(InformationItem(name=name.upper(), modality=mod, notes=notes))
    return items

def prompt_list_teaming(label: str) -> List[TeamingElement]:
    """
    Repeatedly prompt for TeamingElement entries.
    """
    items: List[TeamingElement] = []
    print(f"\nAdd {label} requirement (leave 'source' empty to stop):")
    target_default = None
    while True:
        if label == "team observability":
            print("  Who needs to observe what from whom ?")
        elif label == "team predictability":
            print("  Who needs to predict what from whom ?")
        elif label == "team directability":
            print("  Who needs to direct what to who ?")
        source = prompt_enum("  source (who): ", Teammate)
        if not source:
            return items
        if source.value == Teammate.HUMAN.value:
            target_default = Teammate.TARS.value
        elif source.value == Teammate.TARS.value:
            target_default = Teammate.HUMAN.value
        target = prompt_enum(f"  target (who): ", Teammate, default=target_default) or target_default
        what = input("  what: ").strip()
        mod = prompt_enum(f"  How can the {label} be achieved?", RequirementModality, default=RequirementModality.VISUAL_SCREEN.value)
        notes = input("  notes (optional): ").strip() or None
        items.append(TeamingElement(source=source, target=target, what=what.upper(), modality=mod, notes=notes))
    return items

def prompt_procedure(existing_procedures: list, default: Optional[str] = None) -> str:
    """
    Prompt user to select or enter a procedure.
    Shows existing procedures as numbered options.
    """
    if existing_procedures:
        print("\nExisting procedures:")
        for idx, proc in enumerate(existing_procedures, 1):
            print(f"  {idx}. {proc}")
        print("  0. Enter a new procedure")
    default_str = f" [{default}]" if default else ""
    while True:
        ans = input(f"task.procedure (pick number, or type new){default_str}: ").strip()
        if ans == "" and default:
            return default
        if ans.isdigit():
            idx = int(ans)
            if idx == 0:
                # Enter new procedure
                new_proc = input("Enter new procedure name: ").strip()
                if new_proc:
                    return new_proc
            elif 1 <= idx <= len(existing_procedures):
                return existing_procedures[idx - 1]
        elif ans:
            return ans
        else:
            print("Please pick a number or enter a procedure name.")

def cmd_add() -> None:
    """
    Interactive 'add' command. Prompts for a minimal viable Task.
    You can always edit tasks.json later for advanced details.
    """
    print("\n=== Add a new Task ===")
    name = input("task.name: ").strip()

    tasks = load_tasks()
    existing_procs = []
    for t in tasks:
        if getattr(t, "procedure", None):
            existing_procs.append(t.procedure)
    existing_procs = sorted(set(existing_procs), key=lambda x: (x is None, x))
    default_proc = existing_procs[-1] if existing_procs else None
    procedure = prompt_procedure(existing_procs, default=default_proc)

    criticality = prompt_float("task.criticality (0 = facultative .. 1 = critical)", default=0.5, min_val=0.0, max_val=1.0)

    exec_type = prompt_enum("task.execution_type", ExecutionType, default=ExecutionType.ONE_OFF.value)
    tctype = prompt_enum("task.time_constraint.type", TimeConstraintType, default=TimeConstraintType.NONE.value)
    tcs = None
    if tctype != TimeConstraintType.NONE:
        tcs = prompt_optional_float("task.time_constraint.s (after how many seconds it can be considered miss or failed?)", default=60)

    # Performer 1 == human
    performer_human = None
    print("\n--- Performer 1 == human ---")
    p_kind = Teammate.HUMAN.value
    p_rel = prompt_enum("human.reliability", CapacityLevel, default=CapacityLevel.RED.value)
    if p_rel is not CapacityLevel.RED:
        p_time = prompt_float("human.efficiency_time_required_s", default=tcs, min_val=0.0)
        p_familiarity = prompt_float("human.familiarity (0 = routine .. 1 = never done before)", default=0.0, min_val=0.0, max_val=1.0)
        #p_notes = input("performer.notes (optional): ").strip() or None
        p_notes = None
        performer_human = Performer(kind=p_kind, reliability=p_rel, efficiency_time_required_s=p_time, familiarity=p_familiarity, notes=p_notes)

    # Performer 2 == TARS/automation 
    performer_tars = None
    print("\n--- Performer 2 == TARS/automation ---")
    p_kind = Teammate.TARS.value
    p_rel = prompt_enum("TARS.reliability", CapacityLevel, default=CapacityLevel.RED.value)
    if p_rel is not CapacityLevel.RED:
        p_time = prompt_float("TARS.efficiency_time_required_s", default=60.0, min_val=0.0)
        #p_familiarity = prompt_float("TARS.familiarity (0 = routine .. 1 = never done before)", default=0.0, min_val=0.0, max_val=1.0)
        p_familiarity = None
        #p_notes = input("performer.notes (optional): ").strip() or None
        p_notes = None
        performer_tars = Performer(kind=p_kind, reliability=p_rel, efficiency_time_required_s=p_time, familiarity=p_familiarity, notes=p_notes)

    # Supporter human (optional)
    supporter_human = None
    if performer_tars is not None:
        if prompt_yes_no("If TARS is performer, can the human support ?", default=False):
            print("\n--- Supporter 1 == human ---")
            s_kind = Teammate.HUMAN.value
            s_cap = prompt_enum("human.reliability", CapacityLevel, default=CapacityLevel.GREEN.value)
            #s_notes = input("supporter.notes (optional): ").strip() or None
            s_notes = None
            supporter_human = Supporter(kind=s_kind, capacity=s_cap, notes=s_notes)

    # Supporter TARS (optional)
    supporter_tars = None
    if performer_human is not None:
        if prompt_yes_no("If human is performer, can TARS support ?", default=False):
            print("\n--- Supporter 2 == TARS/automation ---")
            s_kind = Teammate.TARS.value
            s_cap = prompt_enum("TARS.reliability", CapacityLevel, default=CapacityLevel.GREEN.value)
            #s_notes = input("supporter.notes (optional): ").strip() or None
            s_notes = None
            supporter_tars = Supporter(kind=s_kind, capacity=s_cap, notes=s_notes)

    # Lists
    info_reqs = prompt_list_information("information requirements")
    mem_reqs = prompt_list_information("memory requirements")
    observability = prompt_list_teaming("team observability")
    predictability = prompt_list_teaming("team predictability")
    directability = prompt_list_teaming("team directability")

    #tags_str = input("\nTags (comma-separated, optional): ").strip()
    #tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    tags = []
    notes = None
    task_1 = None
    task_2 = None
    #notes = input("task.notes (optional): ").strip() or None

    if performer_human is not None:
        task_1 = Task(
            name=name.upper(),
            procedure=procedure.upper(),
            criticality=criticality,
            performer=performer_human,
            supporter=supporter_tars,
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

    if performer_tars is not None:
        task_2 = Task(
            name=name.upper(),
            procedure=procedure.upper(),
            criticality=criticality,
            performer=performer_tars,
            supporter=supporter_human,
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
    if task_1 is not None:
        tasks.append(task_1)
    if task_2 is not None:
        tasks.append(task_2)

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
