from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

@dataclass
class MetaboliteData:
    id: str
    name: str
    compartment: str = ""
    formula: str = ""
    charge: Optional[float] = None
    annotations: dict = field(default_factory=dict)
    boundary_condition: bool = False

@dataclass
class ReactionData:
    id: str
    name: str
    lower_bound: float
    upper_bound: float
    stoichiometry: Dict[str, float] = field(default_factory=dict)
    subsystem: str = ""
    gene_rule: str = ""
    annotations: dict = field(default_factory=dict)

    @property
    def reversible(self) -> bool:
        return self.lower_bound < 0 < self.upper_bound

@dataclass
class ModelData:
    id: str
    name: str
    metabolites: Dict[str, MetaboliteData]
    reactions: Dict[str, ReactionData]
    compartments: Dict[str, str]
    objectives: Dict[str, float]
    objective_direction: str = "max"
    gene_count: int = 0
    source_path: Optional[Path] = None

    genes: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def objective_reaction_ids(self):
        return [rid for rid, coef in self.objectives.items() if abs(coef) > 0]
