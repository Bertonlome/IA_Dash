#!/usr/bin/env python3
# Web app only: Dash interface for task management

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import json
import csv
import os
# Dash imports for web form
import dash
from dash import dcc, html, Input, Output, State

external_stylesheets = ["https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"]

SectionMargin = 10
FieldGap = 8

def launch_dash_app():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    # Helper dropdowns
    exec_type_opts = [{'label': e.value, 'value': e.value} for e in ExecutionType]
    tctype_opts = [{'label': e.value, 'value': e.value} for e in TimeConstraintType]
    rel_opts = [{'label': e.value, 'value': e.value} for e in ReliabilityLevel]
    modality_opts = [{'label': e.value, 'value': e.value} for e in Modality]

    app.layout = html.Div([
        html.H2("Add a New Task"),
        html.Div([
            html.H4("Procedure to which this task belongs (optional)"),
            dcc.Input(id='procedure_name', type='text', placeholder='Procedure Name'),
        ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap}),
        html.Div([
            html.H4("Task Details"),
            dcc.Input(id='task_name', type='text', placeholder='Task Name'),
            dcc.Input(id='criticality', type='number', min=0, max=1, step=0.01, placeholder='Criticality (0-1) 0 = facultative, 1 = critical)'),
            html.Label("Execution type : one_off (default), continuous, cyclic"),
            dcc.Dropdown(id='exec_type', options=exec_type_opts, value=ExecutionType.ONE_OFF.value, placeholder='Execution Type'),
            html.Label("Time Constraint Type : none (default), soft = should be done in less than <time>, hard = must be done in less than <time>"),
            dcc.Dropdown(id='tctype', options=tctype_opts, value=TimeConstraintType.NONE.value, placeholder='Time Constraint Type'),
            html.Label("If time constraint is soft or hard, specify the time in seconds"),
            dcc.Input(id='tcs', type='number', placeholder='Time Constraint (s)'),
        ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap}),
        html.Div([
            html.H4("Performer"),
            dcc.Input(id='p_id', type='text', placeholder='Performer ID'),
            html.Label("Performer Kind (e.g. human, ai, robot X)"),
            dcc.Input(id='p_kind', type='text', placeholder='Performer Kind'),
            html.Label("Performer Capacity refer to IA color coding"),
            dcc.Dropdown(id='p_rel', options=rel_opts, value=ReliabilityLevel.RED.value, placeholder='Performer Reliability'),
            html.Label("Performer estimated time to complete the task in seconds"),
            dcc.Input(id='p_time', type='number', placeholder='Time Required (s)'),
            html.Label("Performer familiarity with the task (0 = routine .. 1 = never done before)"),
            dcc.Input(id='p_familiarity', type='number', min=0, max=1, step=0.01, placeholder='Familiarity (0-1)'),
            dcc.Input(id='p_notes', type='text', placeholder='Performer optional notes'),
        ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap}),
        dcc.Store(id='supporter_count', data=0),
        html.Div(id='supporter_sections'),
        html.Button('Add Supporter for the task', id='add_supporter_btn'),
        dcc.Store(id='info_count', data=0),
        html.Div(id='info_sections'),
        html.Button('Add Information Requirement', id='add_info_btn'),
        dcc.Store(id='mem_count', data=0),
        html.Div(id='mem_sections'),
        html.Button('Add Memory Requirement', id='add_mem_btn'),
        dcc.Store(id='obs_count', data=0),
        html.Div(id='obs_sections'),
        html.Button('Add Team Observability', id='add_obs_btn'),
        dcc.Store(id='pred_count', data=0),
        html.Div(id='pred_sections'),
        html.Button('Add Team Predictability', id='add_pred_btn'),
        dcc.Store(id='dir_count', data=0),
        html.Div(id='dir_sections'),
        html.Button('Add Team Directability', id='add_dir_btn'),
        html.Div([
            dcc.Input(id='notes', type='text', placeholder='Task Notes'),
            html.Button('Submit', id='submit_btn'),
            html.Div(id='output')
        ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap}),
    ], style={'fontFamily': 'Roboto', 'margin': '20px'})
    # Callback to increment info count
    @app.callback(
        Output('info_count', 'data'),
        Input('add_info_btn', 'n_clicks'),
        State('info_count', 'data'),
        prevent_initial_call=True
    )
    def add_info(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render info sections
    @app.callback(
        Output('info_sections', 'children'),
        Input('info_count', 'data')
    )
    def render_infos(count):
        return [
            html.Div([
                html.H4(f"Information Requirement #{i+1}"),
                dcc.Input(id={'type': 'info_names', 'index': i}, type='text', placeholder='Names (e.g. FMA_status|ATC_instruction)'),
                dcc.Dropdown(id={'type': 'info_modality', 'index': i}, options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
                dcc.Input(id={'type': 'info_notes', 'index': i}, type='text', placeholder='Notes (optional)'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]

    # Callback to increment mem count
    @app.callback(
        Output('mem_count', 'data'),
        Input('add_mem_btn', 'n_clicks'),
        State('mem_count', 'data'),
        prevent_initial_call=True
    )
    def add_mem(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render mem sections
    @app.callback(
        Output('mem_sections', 'children'),
        Input('mem_count', 'data')
    )
    def render_mems(count):
        return [
            html.Div([
                html.H4(f"Memory Requirement #{i+1}"),
                dcc.Input(id={'type': 'mem_names', 'index': i}, type='text', placeholder='Names (e.g. FMA_status|ATC_instruction)'),
                dcc.Dropdown(id={'type': 'mem_modality', 'index': i}, options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
                dcc.Input(id={'type': 'mem_notes', 'index': i}, type='text', placeholder='Notes (optional)'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]

    # Callback to increment obs count
    @app.callback(
        Output('obs_count', 'data'),
        Input('add_obs_btn', 'n_clicks'),
        State('obs_count', 'data'),
        prevent_initial_call=True
    )
    def add_obs(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render obs sections
    @app.callback(
        Output('obs_sections', 'children'),
        Input('obs_count', 'data')
    )
    def render_obs(count):
        return [
            html.Div([
                html.H4(f"Team Observability #{i+1}"),
                dcc.Input(id={'type': 'obs_targets', 'index': i}, type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
                dcc.Dropdown(id={'type': 'obs_modality', 'index': i}, options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
                dcc.Input(id={'type': 'obs_notes', 'index': i}, type='text', placeholder='Notes (optional)'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]

    # Callback to increment pred count
    @app.callback(
        Output('pred_count', 'data'),
        Input('add_pred_btn', 'n_clicks'),
        State('pred_count', 'data'),
        prevent_initial_call=True
    )
    def add_pred(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render pred sections
    @app.callback(
        Output('pred_sections', 'children'),
        Input('pred_count', 'data')
    )
    def render_preds(count):
        return [
            html.Div([
                html.H4(f"Team Predictability #{i+1}"),
                dcc.Input(id={'type': 'pred_targets', 'index': i}, type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
                dcc.Dropdown(id={'type': 'pred_modality', 'index': i}, options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
                dcc.Input(id={'type': 'pred_notes', 'index': i}, type='text', placeholder='Notes (optional)'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]

    # Callback to increment dir count
    @app.callback(
        Output('dir_count', 'data'),
        Input('add_dir_btn', 'n_clicks'),
        State('dir_count', 'data'),
        prevent_initial_call=True
    )
    def add_dir(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render dir sections
    @app.callback(
        Output('dir_sections', 'children'),
        Input('dir_count', 'data')
    )
    def render_dirs(count):
        return [
            html.Div([
                html.H4(f"Team Directability #{i+1}"),
                dcc.Input(id={'type': 'dir_targets', 'index': i}, type='text', placeholder='Targets (e.g. VirtualCopilot:mode_transition|Autopilot:status)'),
                dcc.Dropdown(id={'type': 'dir_modality', 'index': i}, options=modality_opts, value=Modality.OTHER.value, placeholder='Modality'),
                dcc.Input(id={'type': 'dir_notes', 'index': i}, type='text', placeholder='Notes (optional)'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]
    # Callback to increment supporter count
    @app.callback(
        Output('supporter_count', 'data'),
        Input('add_supporter_btn', 'n_clicks'),
        State('supporter_count', 'data'),
        prevent_initial_call=True
    )
    def add_supporter(n_clicks, count):
        if n_clicks:
            return count + 1
        return count

    # Callback to render supporter sections
    from dash.dependencies import ALL
    @app.callback(
        Output('supporter_sections', 'children'),
        Input('supporter_count', 'data')
    )
    def render_supporters(count):
        return [
            html.Div([
                html.H4(f"Supporter #{i+1}"),
                dcc.Input(id={'type': 's_id', 'index': i}, type='text', placeholder='Supporter ID'),
                html.Label("Supporter Kind (e.g. human, ai, robot X)"),
                dcc.Input(id={'type': 's_kind', 'index': i}, type='text', placeholder='Supporter Kind'),
                html.Label("Supporter Capacity refer to IA color coding"),
                dcc.Dropdown(id={'type': 's_cap', 'index': i}, options=rel_opts, value=ReliabilityLevel.GREEN.value, placeholder='Supporter Capacity'),
                dcc.Input(id={'type': 's_notes', 'index': i}, type='text', placeholder='Supporter optional notes'),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': SectionMargin, 'gap': FieldGap})
            for i in range(count)
        ]

    @app.callback(
        Output('output', 'children'),
        Input('submit_btn', 'n_clicks'),
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
        State('notes', 'value')
    )
    def submit_task(n_clicks, task_name, criticality, exec_type, tctype, tcs,
                    p_id, p_kind, p_rel, p_time, p_familiarity, p_notes, notes):
        if not n_clicks:
            return ""
        # Template for collecting dynamic values from pattern-matching IDs
        from dash import ctx
        # Collect dynamic values using ctx.inputs and ctx.states
        # Supporters
        supporter_count = ctx.states.get('supporter_count.data', 0)
        supporters = []
        for i in range(supporter_count):
            s_id = ctx.states.get(f"{{'type': 's_id', 'index': {i}}}.value", "")
            s_kind = ctx.states.get(f"{{'type': 's_kind', 'index': {i}}}.value", "automation")
            s_cap = ctx.states.get(f"{{'type': 's_cap', 'index': {i}}}.value", ReliabilityLevel.GREEN.value)
            s_notes = ctx.states.get(f"{{'type': 's_notes', 'index': {i}}}.value", "")
            if s_id:
                supporters.append(Supporter(id=s_id, kind=s_kind, capacity=s_cap, notes=s_notes))

        # Information Requirements
        info_count = ctx.states.get('info_count.data', 0)
        information_requirements = []
        for i in range(info_count):
            info_names = ctx.states.get(f"{{'type': 'info_names', 'index': {i}}}.value", "")
            info_modality = ctx.states.get(f"{{'type': 'info_modality', 'index': {i}}}.value", Modality.OTHER.value)
            info_notes = ctx.states.get(f"{{'type': 'info_notes', 'index': {i}}}.value", "")
            if info_names:
                for name in info_names.split('|'):
                    if name.strip():
                        information_requirements.append(InformationItem(name=name.strip(), modality=[Modality(info_modality)], notes=info_notes))

        # Memory Requirements
        mem_count = ctx.states.get('mem_count.data', 0)
        memory_requirements = []
        for i in range(mem_count):
            mem_names = ctx.states.get(f"{{'type': 'mem_names', 'index': {i}}}.value", "")
            mem_modality = ctx.states.get(f"{{'type': 'mem_modality', 'index': {i}}}.value", Modality.OTHER.value)
            mem_notes = ctx.states.get(f"{{'type': 'mem_notes', 'index': {i}}}.value", "")
            if mem_names:
                for name in mem_names.split('|'):
                    if name.strip():
                        memory_requirements.append(InformationItem(name=name.strip(), modality=[Modality(mem_modality)], notes=mem_notes))

        # Team Observability
        obs_count = ctx.states.get('obs_count.data', 0)
        observability = []
        for i in range(obs_count):
            obs_targets = ctx.states.get(f"{{'type': 'obs_targets', 'index': {i}}}.value", "")
            obs_modality = ctx.states.get(f"{{'type': 'obs_modality', 'index': {i}}}.value", Modality.OTHER.value)
            obs_notes = ctx.states.get(f"{{'type': 'obs_notes', 'index': {i}}}.value", "")
            if obs_targets:
                for t in obs_targets.split('|'):
                    if ':' in t:
                        target, what = t.split(':', 1)
                        observability.append(TeamingElement(target=target.strip(), what=what.strip(), modality=[Modality(obs_modality)], notes=obs_notes))

        # Team Predictability
        pred_count = ctx.states.get('pred_count.data', 0)
        predictability = []
        for i in range(pred_count):
            pred_targets = ctx.states.get(f"{{'type': 'pred_targets', 'index': {i}}}.value", "")
            pred_modality = ctx.states.get(f"{{'type': 'pred_modality', 'index': {i}}}.value", Modality.OTHER.value)
            pred_notes = ctx.states.get(f"{{'type': 'pred_notes', 'index': {i}}}.value", "")
            if pred_targets:
                for t in pred_targets.split('|'):
                    if ':' in t:
                        target, what = t.split(':', 1)
                        predictability.append(TeamingElement(target=target.strip(), what=what.strip(), modality=[Modality(pred_modality)], notes=pred_notes))

        # Team Directability
        dir_count = ctx.states.get('dir_count.data', 0)
        directability = []
        for i in range(dir_count):
            dir_targets = ctx.states.get(f"{{'type': 'dir_targets', 'index': {i}}}.value", "")
            dir_modality = ctx.states.get(f"{{'type': 'dir_modality', 'index': {i}}}.value", Modality.OTHER.value)
            dir_notes = ctx.states.get(f"{{'type': 'dir_notes', 'index': {i}}}.value", "")
            if dir_targets:
                for t in dir_targets.split('|'):
                    if ':' in t:
                        target, what = t.split(':', 1)
                        directability.append(TeamingElement(target=target.strip(), what=what.strip(), modality=[Modality(dir_modality)], notes=dir_notes))

        # Build the Task object
        supporter = supporters[0] if supporters else None
        task = Task(
            id=task_name or "", name=task_name or "", criticality=criticality or 0.5,
            performer=performer, supporter=supporter,
            time_constraint_type=TimeConstraintType(tctype or TimeConstraintType.NONE.value),
            time_constraint_s=tcs,
            execution_type=ExecutionType(exec_type or ExecutionType.ONE_OFF.value),
            information_requirements=information_requirements,
            memory_requirements=memory_requirements,
            observability=observability,
            predictability=predictability,
            directability=directability,
            tags=[], notes=notes
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
        return
    rows = [t.to_flat_dict() for t in tasks]
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


