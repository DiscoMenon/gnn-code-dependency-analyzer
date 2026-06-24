from pydantic import BaseModel
from typing import Optional

# --- Request ---

class AnalyzeRequest(BaseModel):
    github_url: str

# --- Response building blocks ---

class NodeSchema(BaseModel):
    id: str
    type: str                    # "file" | "function" | "class"
    anomaly_score: float         # 0.0 - 1.0
    in_degree: int
    out_degree: int
    pagerank: float
    in_cycle: bool
    issues: list[str]            # ["GOD FILE", "CIRCULAR DEP", "DEAD CODE"]

class EdgeSchema(BaseModel):
    source: str
    target: str
    type: str                    # "imports" | "calls" | "contains"

class IssueSchema(BaseModel):
    type: str                    # "god_file" | "circular_dep" | "dead_code"
    severity: str                # "high" | "medium" | "low"
    nodes: list[str]
    description: str

class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    file_count: int
    function_count: int
    class_count: int
    circular_dep_count: int
    god_file_count: int
    dead_code_count: int

# --- Final response ---

class AnalyzeResponse(BaseModel):
    repo_name: str
    status: str                  # "success" | "error"
    stats: GraphStats
    nodes: list[NodeSchema]
    edges: list[EdgeSchema]
    issues: list[IssueSchema]
    error: Optional[str] = None