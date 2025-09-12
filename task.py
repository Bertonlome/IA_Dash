from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import json
import csv
import os

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
    name: str
    modality: List[Modality] = field(default_factory=lambda: [Modality.OTHER])
    notes: Optional[str] = None

@dataclass
class TeamingElement:
    target: str
    what: str
    modality: Optional[List[Modality]] = None
    notes: Optional[str] = None

# ============
# Actor types
# ============

@dataclass
class Performer:
    kind: str
    reliability: str
    efficiency_time_required_s: float
    familiarity : Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Supporter:
    kind: str
    capacity: str
    notes: Optional[str] = None

# =====
# Task
# =====

@dataclass
class Task:
    name: str
    criticality: float
    performer: Performer
    procedure: Optional[str] = None
    supporter: Optional[Supporter] = None
    time_constraint_type: TimeConstraintType = TimeConstraintType.NONE
    time_constraint_s: Optional[float] = None
    execution_type: ExecutionType = ExecutionType.ONE_OFF
    information_requirements: List[InformationItem] = field(default_factory=list)
    memory_requirements: List[InformationItem] = field(default_factory=list)
    observability: List[TeamingElement] = field(default_factory=list)
    predictability: List[TeamingElement] = field(default_factory=list)
    directability: List[TeamingElement] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def temporal_demand(self) -> Optional[float]:
        if self.time_constraint_type == TimeConstraintType.NONE or not self.time_constraint_s:
            return None
        if self.performer.efficiency_time_required_s <= 0:
            return 0.0
        return self.performer.efficiency_time_required_s / self.time_constraint_s

    def to_flat_dict(self) -> Dict[str, Any]:
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
            "task.procedure": self.procedure,
            "task.name": self.name,
            "task.criticality": self.criticality,
            "task.execution_type": self.execution_type.value,
            "task.time_constraint.type": self.time_constraint_type.value,
            "task.time_constraint.s": self.time_constraint_s if self.time_constraint_s is not None else "",
            "task.temporal_demand": self.temporal_demand() if self.temporal_demand() is not None else "",
            "performer.kind": self.performer.kind,
            "performer.reliability": self.performer.reliability,
            "performer.efficiency_time_required_s": self.performer.efficiency_time_required_s,
            "performer.familiarity": self.performer.familiarity,
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
        d = asdict(self)
        d["time_constraint_type"] = self.time_constraint_type.value
        d["execution_type"] = self.execution_type.value
        if d.get("performer"):
            if isinstance(d["performer"].get("reliability"), Enum):
                d["performer"]["reliability"] = d["performer"]["reliability"].value
        if d.get("supporter"):
            if isinstance(d["supporter"].get("capacity"), Enum):
                d["supporter"]["capacity"] = d["supporter"]["capacity"].value
        for k in ["information_requirements", "memory_requirements"]:
            for item in d[k]:
                if isinstance(item.get("modality"), list):
                    item["modality"] = [m.value if isinstance(m, Enum) else m for m in item["modality"]]
                elif isinstance(item.get("modality"), Enum):
                    item["modality"] = item["modality"].value
        for k in ["observability", "predictability", "directability"]:
            for item in d[k]:
                if item.get("modality"):
                    if isinstance(item["modality"], list):
                        item["modality"] = [m.value if isinstance(m, Enum) else m for m in item["modality"]]
                    elif isinstance(item["modality"], Enum):
                        item["modality"] = item["modality"].value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Task':
        performer = Performer(**data["performer"])
        supporter = Supporter(**data["supporter"]) if data.get("supporter") else None
        def parse_info(lst):
            return [InformationItem(name=i["name"], modality=Modality(i.get("modality", "other")), notes=i.get("notes")) for i in (lst or [])]
        def parse_team(lst):
            return [TeamingElement(target=i["target"], what=i["what"], modality=Modality(i["modality"]) if i.get("modality") else None, notes=i.get("notes")) for i in (lst or [])]
        return Task(
            name=data["name"],
            criticality=float(data["criticality"]),
            performer=performer,
            procedure=data.get("procedure"),
            supporter=supporter,
            time_constraint_type=TimeConstraintType(data.get("time_constraint_type", "none")),
            time_constraint_s=data.get("time_constraint_s"),
            execution_type=ExecutionType(data.get("execution_type", "one_off")),
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
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Task.from_dict(d) for d in raw]

def save_tasks_json(tasks: List[Task]) -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2, ensure_ascii=False)

def export_tasks_csv(tasks: List[Task]) -> None:
    if not tasks:
        return
    rows = [t.to_flat_dict() for t in tasks]
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
