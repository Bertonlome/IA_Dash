#!/usr/bin/env python3
"""
tasks_cli.py
------------
A compact, commented reference implementation for:
- A Task OOP model (with Teaming/Info lists)
- JSON (authoritative store) and CSV (flat view) persistence
- A tiny CLI to add tasks interactively and to export CSV

Usage:
  python tasks_cli.py add
  python tasks_cli.py export

Files:
  tasks.json  -> authoritative list of tasks (nested, full fidelity)
  tasks.csv   -> flattened export suitable for spreadsheets/BI tools
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import json
import csv
import os
import argparse
# Dash imports for web form
import dash
from dash import dcc, html, Input, Output, State

def launch_dash_app():
    app = dash.Dash(__name__)
    # Helper dropdowns
    exec_type_opts = [{'label': e.value, 'value': e.value} for e in ExecutionType]
    tctype_opts = [{'label': e.value, 'value': e.value} for e in TimeConstraintType]
    rel_opts = [{'label': e.value, 'value': e.value} for e in ReliabilityLevel]
    modality_opts = [{'label': e.value, 'value': e.value} for e in Modality]

    app.layout = html.Div([
        html.H2("Add a New Task"),
        html.Div([
        dcc.Input(id='procedure_name', type='text', placeholder='Procedure Name'),
        ]),
        
        html.Div([
        dcc.Input(id='task_name', type='text', placeholder='Task Name'),
        dcc.Input(id='criticality', type='number', min=0, max=1, step=0.01, placeholder='Criticality (0-1)'),
        dcc.Dropdown(id='exec_type', options=exec_type_opts, value=ExecutionType.ONE_OFF.value, placeholder='Execution Type'),
        dcc.Dropdown(id='tctype', options=tctype_opts, value=TimeConstraintType.NONE.value, placeholder='Time Constraint Type'),
        dcc.Input(id='tcs', type='number', placeholder='Time Constraint (s)'),
        ]),
        
        html.Div([
        html.H4("Performer"),
        dcc.Input(id='p_id', type='text', placeholder='Performer ID'),
        dcc.Input(id='p_kind', type='text', placeholder='Performer Kind'),
        dcc.Dropdown(id='p_rel', options=rel_opts, value=ReliabilityLevel.RED.value, placeholder='Performer Reliability'),
        dcc.Input(id='p_time', type='number', placeholder='Efficiency Time Required (s)'),
        dcc.Input(id='p_familiarity', type='number', min=0, max=1, step=0.01, placeholder='Familiarity (0-1)'),
        dcc.Input(id='p_notes', type='text', placeholder='Performer Notes'),
        ]),
        
        html.Div([
        html.H4("Supporter (optional)"),
        dcc.Input(id='s_id', type='text', placeholder='Supporter ID'),
        dcc.Input(id='s_kind', type='text', placeholder='Supporter Kind'),
        dcc.Dropdown(id='s_cap', options=rel_opts, value=ReliabilityLevel.GREEN.value, placeholder='Supporter Capacity'),
        dcc.Input(id='s_notes', type='text', placeholder='Supporter Notes'),
        ]),
        
        html.Div([
        html.H4("Information Requirements (add multiple, separated by |)", style={"marginTop": "20px"}),
        dcc.Input(id='info_names', type='text', placeholder='Names (e.g. FMA_status|ATC_instruction)'),
        dcc.Dropdown(id='info_modality', options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
        dcc.Input(id='info_notes', type='text', placeholder='Notes (optional)'),
        ]),
        
        html.Div([
        html.H4("Memory Requirements (add multiple, separated by |)"),
        dcc.Input(id='mem_names', type='text', placeholder='Names (e.g. FMA_status|ATC_instruction)'),
        dcc.Dropdown(id='mem_modality', options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
        dcc.Input(id='mem_notes', type='text', placeholder='Notes (optional)'),
        ]),
        
        html.Div([
        html.H4("Team Observability (target:what, separated by |)"),
        dcc.Input(id='obs_targets', type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
        dcc.Dropdown(id='obs_modality', options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
        dcc.Input(id='obs_notes', type='text', placeholder='Notes (optional)'),
        ]),

        html.Div([
        html.H4("Team Predictability (target:what, separated by |)"),
        dcc.Input(id='pred_targets', type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
        dcc.Dropdown(id='pred_modality', options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
        dcc.Input(id='pred_notes', type='text', placeholder='Notes (optional)'),
        ]),
        
        html.Div([
        html.H4("Team Directability (target:what, separated by |)"),
        dcc.Input(id='dir_targets', type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
        dcc.Dropdown(id='dir_modality', options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
        dcc.Input(id='dir_notes', type='text', placeholder='Notes (optional)'),
        ]),
        
        html.Div([
        html.H4("Tags (comma separated)"),
        dcc.Input(id='tags', type='text', placeholder='Tags'),
        dcc.Input(id='notes', type='text', placeholder='Task Notes'),
        html.Button('Submit', id='submit_btn'),
        html.Div(id='output')
        ]),
    ])

    @app.callback(
        Output('output', 'children'),
        Input('submit_btn', 'n_clicks'),
        State('task_id', 'value'),
        State('task_name', 'value'),
        State('criticality', 'value'),
        State('exec_type', 'value'),
        State('tctype', 'value'),
        State('tcs', 'value'),
        State('p_id', 'value'),
        State('p_kind', 'value'),
        State('p_rel', 'value'),
        State('p_time', 'value'),
        State('p_familiarity', 'value'),
        State('p_notes', 'value'),
        State('s_id', 'value'),
        State('s_kind', 'value'),
        State('s_cap', 'value'),
        State('s_notes', 'value'),
        State('info_names', 'value'),
        State('info_modality', 'value'),
        State('info_notes', 'value'),
        State('mem_names', 'value'),
        State('mem_modality', 'value'),
        State('mem_notes', 'value'),
        State('obs_targets', 'value'),
        State('obs_modality', 'value'),
        State('obs_notes', 'value'),
        State('pred_targets', 'value'),
        State('pred_modality', 'value'),
        State('pred_notes', 'value'),
        State('dir_targets', 'value'),
        State('dir_modality', 'value'),
        State('dir_notes', 'value'),
        State('tags', 'value'),
        State('notes', 'value')
    )
    def submit_task(n_clicks, task_id, task_name, criticality, exec_type, tctype, tcs,
                    p_id, p_kind, p_rel, p_time, p_familiarity, p_notes,
                    s_id, s_kind, s_cap, s_notes,
                    info_names, info_modality, info_notes,
                    mem_names, mem_modality, mem_notes,
                    obs_targets, obs_modality, obs_notes,
                    pred_targets, pred_modality, pred_notes,
                    dir_targets, dir_modality, dir_notes,
                    tags, notes):
        if not n_clicks:
            return ""
        # Parse list fields
        def parse_info(names, modality, notes):
            if not names:
                return []
            return [InformationItem(name=n.strip(), modality=[Modality(modality)], notes=notes) for n in names.split('|') if n.strip()]
        def parse_team(targets, modality, notes):
            if not targets:
                return []
            items = []
            for t in targets.split('|'):
                if ':' in t:
                    target, what = t.split(':', 1)
                    items.append(TeamingElement(target=target.strip(), what=what.strip(), modality=[Modality(modality)], notes=notes))
            return items
        performer = Performer(
            id=p_id or "", kind=p_kind or "human", reliability=p_rel or ReliabilityLevel.RED.value,
            efficiency_time_required_s=p_time or 60.0, familiarity=p_familiarity or 0.0, notes=p_notes
        )
        supporter = None
        if s_id:
            supporter = Supporter(id=s_id, kind=s_kind or "automation", capacity=s_cap or ReliabilityLevel.GREEN.value, notes=s_notes)
        task = Task(
            id=task_id or "", name=task_name or "", criticality=criticality or 0.5,
            performer=performer, supporter=supporter,
            time_constraint_type=TimeConstraintType(tctype or TimeConstraintType.NONE.value),
            time_constraint_s=tcs,
            execution_type=ExecutionType(exec_type or ExecutionType.ONE_OFF.value),
            information_requirements=parse_info(info_names, info_modality, info_notes),
            memory_requirements=parse_info(mem_names, mem_modality, mem_notes),
            observability=parse_team(obs_targets, obs_modality, obs_notes),
            predictability=parse_team(pred_targets, pred_modality, pred_notes),
            directability=parse_team(dir_targets, dir_modality, dir_notes),
            tags=[t.strip() for t in (tags or "").split(",") if t.strip()], notes=notes
        )
        tasks = load_tasks()
        existing_ids = {t.id: idx for idx, t in enumerate(tasks)}
        if task.id in existing_ids:
            tasks[existing_ids[task.id]] = task
        else:
            tasks.append(task)
        save_tasks_json(tasks)
        export_tasks_csv(tasks)
        return f"Task '{task.id}' saved!"

    app.run(debug=True)

# =========================
# Enumerations (normalized)
# =========================

class ExecutionType(Enum):
    ONE_OFF = "one_off"
    CONTINUOUS = "continuous"
    CYCLIC = "cyclic"


class Modality(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    HAPTIC = "haptic"
    PROPRIOCEPTIVE = "proprioceptive"
    OTHER = "other"


class TimeConstraintType(Enum):
    NONE = "none"          # no explicit time limit
    SOFT_DEADLINE = "soft" # preferred completion time
    HARD_DEADLINE = "hard" # strict deadline, failure if exceeded


class ReliabilityLevel(Enum):
    GREEN = "green"    # High reliability
    YELLOW = "yellow"  # Medium reliability
    ORANGE = "orange"  # Low reliability
    RED = "red"        # Very low reliability
# ======================
# Atomic descriptor types
# ======================

@dataclass
class InformationItem:
    """
    A piece of information needed/remembered/observed.
    - name: label of the information (e.g., "FMA_status")
    - modality: sensory channel used (visual, auditory, etc.)
    - notes: optional context
    """
    name: str
    modality: List[Modality] = field(default_factory=lambda: [Modality.OTHER])
    notes: Optional[str] = None


@dataclass
class TeamingElement:
    """
    An element for teaming with a teammate (human or automation).
    - target: teammate or subsystem (e.g., "VirtualCopilot", "Autopilot")
    - what: state/intent/parameter of interest (e.g., "mode_transition")
    - modality: how it's perceived/communicated (optional)
    """
    target: str
    what: str
    modality: Optional[List[Modality]] = None
    notes: Optional[str] = None


# ============
# Actor types
# ============

@dataclass
class Performer:
    """
    The entity executing the task (human, AI, robot).
    - reliability: P(success) for this task [0..1]
    - efficiency_time_required_s: nominal completion time in seconds
    """
    id: str
    kind: str                        # "human", "ai", "robot", etc.
    reliability: str               # Color grading per IA
    efficiency_time_required_s: float

    # Optional metadata
    familiarity : Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Supporter:
    """
    Supporting entity or resource.
    - capacity to support the performer 
    """
    id: str
    kind: str                        # "human_assistant", "tool", "automation"
    capacity: str                      # Color grading per IA
    notes: Optional[str] = None


# =====
# Task
# =====

@dataclass
class Task:
    """
    Core task class capturing your selected attributes.
    """
    # Identity
    name: str

    # Selected attributes
    criticality: float                               # e.g., 0..1 or any risk score
    performer: Performer
    supporter: Optional[Supporter] = None
    time_constraint_type: TimeConstraintType = TimeConstraintType.NONE
    time_constraint_s: Optional[float] = None
    execution_type: ExecutionType = ExecutionType.ONE_OFF

    # Lists (will serialize to strings in CSV; JSON remains nested)
    information_requirements: List[InformationItem] = field(default_factory=list)
    memory_requirements: List[InformationItem] = field(default_factory=list)
    observability: List[TeamingElement] = field(default_factory=list)
    predictability: List[TeamingElement] = field(default_factory=list)
    directability: List[TeamingElement] = field(default_factory=list)

    # Optional metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # ----- Derived metrics -----

    def temporal_demand(self) -> Optional[float]:
        """
        performer_time / available_time
        <1 slack; =1 just-in-time; >1 infeasible under current constraint.
        """
        if self.time_constraint_type == TimeConstraintType.NONE or not self.time_constraint_s:
            return None
        if self.performer.efficiency_time_required_s <= 0:
            return 0.0
        return self.performer.efficiency_time_required_s / self.time_constraint_s

    # ----- Serialization helpers -----

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten to a CSV-friendly dict. List fields are joined with '|'
        and subfields tagged with '#'. Example: "FMA_status#visual|ATC_instruction#auditory"
        """
        def pack_info(items: List[InformationItem]) -> str:
            return "|".join(
                f"{i.name}#{i.modality.value}" if i.modality else i.name
                for i in items
            )

        def pack_team(items: List[TeamingElement]) -> str:
            return "|".join(
                f"{t.target}:{t.what}#{t.modality.value if t.modality else 'na'}"
                for t in items
            )

        row = {
            "task.id": self.id,
            "task.name": self.name,
            "task.criticality": self.criticality,
            "task.execution_type": self.execution_type.value,
            "task.time_constraint.type": self.time_constraint_type.value,
            "task.time_constraint.s": self.time_constraint_s if self.time_constraint_s is not None else "",
            "task.temporal_demand": self.temporal_demand() if self.temporal_demand() is not None else "",
            "performer.id": self.performer.id,
            "performer.kind": self.performer.kind,
            "performer.reliability": self.performer.reliability,
            "performer.efficiency_time_required_s": self.performer.efficiency_time_required_s,
            "performer.familiarity": self.performer.familiarity,
            "supporter.id": self.supporter.id if self.supporter else "",
            "supporter.kind": self.supporter.kind if self.supporter else "",
            "supporter.capacity": self.supporter.capacity if self.supporter else "",
            "info.requirements": pack_info(self.information_requirements),
            "memory.requirements": pack_info(self.memory_requirements),
            "team.observability": pack_team(self.observability),
            "team.predictability": pack_team(self.predictability),
            "team.directability": pack_team(self.directability),
            "tags": "|".join(self.tags),
            "notes": self.notes or "",
        }
        return row

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable nested dict. Enums are converted to their values.
        """
        d = asdict(self)
        d["time_constraint_type"] = self.time_constraint_type.value
        d["execution_type"] = self.execution_type.value
        # Convert modalities inside nested lists
        for k in ["information_requirements", "memory_requirements"]:
            for item in d[k]:
                item["modality"] = item["modality"].value if item.get("modality") else None
        for k in ["observability", "predictability", "directability"]:
            for item in d[k]:
                if item.get("modality"):
                    item["modality"] = item["modality"].value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Task:
        """
        Inverse of to_dict, to reconstruct dataclass instances from JSON dicts.
        """
        performer = Performer(**data["performer"])
        supporter = Supporter(**data["supporter"]) if data.get("supporter") else None

        def parse_info(lst):
            return [InformationItem(name=i["name"],
                                    modality=Modality(i.get("modality", "other")),
                                    notes=i.get("notes")) for i in (lst or [])]

        def parse_team(lst):
            return [TeamingElement(target=i["target"],
                                   what=i["what"],
                                   modality=Modality(i["modality"]) if i.get("modality") else None,
                                   notes=i.get("notes")) for i in (lst or [])]

        return Task(
            id=data["id"],
            name=data["name"],
            criticality=float(data["criticality"]),
            performer=performer,
            supporter=supporter,
            time_constraint_type=TimeConstraintType(data.get("time_constraint_type", "none")),
            time_constraint_s=data.get("time_constraint_s"),
            execution_type=ExecutionType(data.get("execution_type", "one_off")),
            novelty=float(data.get("novelty", 0.0)),
            information_requirements=parse_info(data.get("information_requirements")),
            memory_requirements=parse_info(data.get("memory_requirements")),
            observability=parse_team(data.get("observability")),
            predictability=parse_team(data.get("predictability")),
            directability=parse_team(data.get("directability")),
            tags=list(data.get("tags", [])),
            notes=data.get("notes")
        )


# ============================
# Persistence helper functions
# ============================

JSON_PATH = "tasks.json"
CSV_PATH = "tasks.csv"

def load_tasks() -> List[Task]:
    """
    Load the authoritative task list from tasks.json (returns [] if missing).
    """
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Task.from_dict(d) for d in raw]

def save_tasks_json(tasks: List[Task]) -> None:
    """
    Save the authoritative, nested JSON list.
    """
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2, ensure_ascii=False)

def export_tasks_csv(tasks: List[Task]) -> None:
    """
    Export a flat CSV view. Schema is taken from the first row's keys.
    """
    if not tasks:
        # Create empty file with header if you like; here we just skip.
        print("No tasks to export.")
        return
    rows = [t.to_flat_dict() for t in tasks]
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Exported {len(rows)} task(s) to {CSV_PATH}")


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
    task_id = input("task.id: ").strip()
    name = input("task.name: ").strip()

    criticality = prompt_float("task.criticality (0 = facultative .. 1 = critical)", default=0.5, min_val=0.0, max_val=1.0)

    exec_type = prompt_enum("task.execution_type", ExecutionType, default=ExecutionType.ONE_OFF.value)
    tctype = prompt_enum("task.time_constraint.type", TimeConstraintType, default=TimeConstraintType.NONE.value)
    tcs = None
    if tctype != TimeConstraintType.NONE:
        tcs = prompt_optional_float("task.time_constraint.s (after how many seconds it can be considered miss or failed?)", default=None)

    # Performer
    print("\n--- Performer ---")
    p_id = input("performer.id: ").strip()
    p_kind = input("performer.kind (human/TARS/UGV/UAV...): ").strip().lower() or "human"
    p_rel = prompt_enum("performer.reliability", ReliabilityLevel, default=ReliabilityLevel.RED.value)
    p_time = prompt_float("performer.efficiency_time_required_s", default=60.0, min_val=0.0)
    p_familiarity = prompt_float("performer.familiarity (0 = routine .. 1 = never done before)", default=0.0, min_val=0.0, max_val=1.0)
    p_notes = input("performer.notes (optional): ").strip() or None
    performer = Performer(id=p_id, kind=p_kind, reliability=p_rel,
                          efficiency_time_required_s=p_time, familiarity=p_familiarity, notes=p_notes)

    # Supporter (optional)
    supporter = None
    if prompt_yes_no("Add a supporter?", default=False):
        print("\n--- Supporter ---")
        s_id = input("supporter.id: ").strip()
        s_kind = input("supporter.kind (human/TARS/UGV/UAV...): ").strip().lower() or "automation"
        s_cap = prompt_enum("supporter.capacity", ReliabilityLevel, default=ReliabilityLevel.GREEN.value)
        s_notes = input("supporter.notes (optional): ").strip() or None
        supporter = Supporter(id=s_id, kind=s_kind, capacity=s_cap, notes=s_notes)

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
        id=task_id,
        name=name,
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
    # simple guard to avoid duplicate IDs; replace if same id
    existing_ids = {t.id: idx for idx, t in enumerate(tasks)}
    if task.id in existing_ids:
        print(f"\nTask with id '{task.id}' exists. Overwriting that entry.")
        tasks[existing_ids[task.id]] = task
    else:
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

def main():
    parser = argparse.ArgumentParser(description="Task manager CLI (JSON store, CSV export)")
    sub = parser.add_subparsers(dest="cmd")  # 'required' removed for compatibility
    sub.add_parser("add", help="Interactively add a new task to tasks.json and update tasks.csv")
    sub.add_parser("export", help="Regenerate tasks.csv from tasks.json")
    args = parser.parse_args()

    if args.cmd is None:
        # No command provided: launch Dash app for web-based form
        launch_dash_app()
        return
    if args.cmd == "add":
        cmd_add()
    elif args.cmd == "export":
        cmd_export()

if __name__ == "__main__":
    main()
