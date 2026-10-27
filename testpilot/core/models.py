"""
Core Pydantic v2 schemas and domain data models for TestPilot AI.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PromptTechnique(str, Enum):
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    CHAIN_OF_THOUGHT = "cot"


class ParameterInfo(BaseModel):
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_required: bool = True


class BoundaryCandidate(BaseModel):
    parameter_name: str
    boundary_type: str  # e.g., "zero", "negative", "max_limit", "null", "empty", "off_by_one"
    suggested_value: Any
    rationale: str


class ASTFunctionDef(BaseModel):
    name: str
    class_name: Optional[str] = None
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    parameters: list[ParameterInfo] = []
    return_type: Optional[str] = None
    branch_conditions: list[str] = []
    boundary_candidates: list[BoundaryCandidate] = []
    raw_source: str = ""
    decorators: list[str] = []

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.name}" if self.class_name else self.name

    @property
    def is_staticmethod(self) -> bool:
        return "staticmethod" in self.decorators

    @property
    def is_classmethod(self) -> bool:
        return "classmethod" in self.decorators

    @property
    def is_property(self) -> bool:
        return "property" in self.decorators


class DiffAnalysis(BaseModel):
    modified_files: list[str] = []
    modified_functions: list[ASTFunctionDef] = []
    total_boundary_points: int = 0
    diff_text: str = ""


class GeneratedTestCase(BaseModel):
    test_name: str = Field(..., description="Valid Python test function name starting with test_")
    target_function: str = Field(..., description="Name of the function under test")
    boundary_focus: str = Field(..., description="Edge case being tested (e.g., negative balance, zero total)")
    input_values: dict[str, Any] = Field(default_factory=dict, description="Input parameters passed into function")
    expected_behavior: str = Field(..., description="Expected outcome or exception constraint")
    rationale: str = Field(..., description="Chain of thought explaining why this boundary condition is critical")
    code: Optional[str] = Field(default=None, description="Complete Python/pytest test implementation")


class GeneratedTestSuite(BaseModel):
    target_module: str = Field(..., description="Path or module name of the target file")
    technique_used: PromptTechnique = Field(default=PromptTechnique.CHAIN_OF_THOUGHT)
    test_cases: list[GeneratedTestCase] = Field(..., min_length=1)
    reasoning_trace: Optional[str] = Field(default=None, description="Step-by-step reasoning trace from LLM")
    complete_pytest_code: Optional[str] = Field(default=None, description="Executable, syntax-valid pytest script")


class VerdictType(str, Enum):
    TRUE_CODE_DEFECT = "TRUE_CODE_DEFECT"
    INVALID_TEST_ASSERTION = "INVALID_TEST_ASSERTION"
    SPEC_AMBIGUITY_OR_DEFECT = "SPEC_AMBIGUITY_OR_DEFECT"
    SPEC_PASS = "SPEC_PASS"


class OracleVerdict(BaseModel):
    test_name: str
    verdict: VerdictType
    contract_violated: Optional[str] = None
    details: str
