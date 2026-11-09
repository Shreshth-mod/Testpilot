"""
Persistent storage and curated benchmark suite definitions for TestPilot Evaluation.
Stores benchmark cases and execution run records with complete provenance.
"""

import json
from pathlib import Path
from typing import Optional

from testpilot.evaluation.models import (
    BenchmarkCase,
    EvaluationRun,
    GroundTruth,
)


class EvaluationStorage:
    """Manages benchmark cases and saved evaluation run results."""

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir).resolve()
        else:
            # Default to testbed/evaluation_store if available, else ~/.testpilot_eval
            workspace_store = Path("testbed/evaluation_store").resolve()
            if workspace_store.parent.exists():
                self.storage_dir = workspace_store
            else:
                self.storage_dir = Path.home() / ".testpilot_eval"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.storage_dir / "evaluation_runs.json"
        self._init_benchmark_cases()

    def _init_benchmark_cases(self) -> None:
        """Initializes curated real-world and controlled validation benchmark cases."""
        self.cases: dict[str, BenchmarkCase] = {}

        # 1. Pallets Flask (Real-World)
        default_flask_path = Path.home() / ".testpilot_repos" / "flask"
        alt_flask_path = Path.home() / "testpilot-external-test" / "flask"
        flask_path = str(default_flask_path if default_flask_path.exists() else alt_flask_path)
        self.cases["flask_ipv6"] = BenchmarkCase(
            case_id="flask_ipv6",
            name="Flask IPv6 Server Name Parsing (#6096)",
            repository="pallets/flask",
            repository_path=flask_path,
            base_commit="de8429ffda8cfb54db175529e2bae72a24e1fa7e",
            target_commit="7203feabf723edae0286ae5dc64fec8ac4c91735",
            description="Fix IPv6 server name parsing via urlsplit instead of partition(':') in Flask.run.",
            category="real_world",
            total_tests_suite=375,
            ground_truth=GroundTruth(
                case_id="flask_ipv6",
                expected_changed_symbols=["run", "Flask.run", "test_run_from_config"],
                expected_impacted_symbols=["Flask", "run_simple", "urlsplit"],
                expected_test_callers=[
                    "test_run_from_config",
                    "test_run_server_port",
                    "test_run_defaults",
                    "test_werkzeug_passthrough_errors",
                    "test_templates_auto_reload_debug_run",
                ],
                provenance="Independent manual verification in ~/testpilot-evaluation/flask/ground_truth.md",
                notes="Tests exercise Flask.run either directly or through parameterization.",
            ),
        )

        # 2. Django (Real-World)
        default_django_path = Path.home() / ".testpilot_repos" / "django"
        alt_django_path = Path.home() / "testpilot-external-test" / "django"
        django_path = str(default_django_path if default_django_path.exists() else alt_django_path)
        self.cases["django_model_rename"] = BenchmarkCase(
            case_id="django_model_rename",
            name="Django Chained RenameModel Permissions (#37361)",
            repository="django/django",
            repository_path=django_path,
            base_commit="3db0898eee4061d44de5052953ef0fe3afb5951f",
            target_commit="7847227a3fecde4b2a169552b84c60c6286b6025",
            description="Fix permission renames for chained RenameModel operations in rename_permissions_after_model_rename.",
            category="real_world",
            total_tests_suite=18079,
            ground_truth=GroundTruth(
                case_id="django_model_rename",
                expected_changed_symbols=["rename_permissions_after_model_rename"],
                expected_impacted_symbols=[],
                expected_test_callers=[
                    "test_permission_rename_chain",
                    "test_permission_rename_duplicate_codename_across_content_types",
                    "test_permission_rename_chain_to_zero_uninterrupted",
                    "test_rename_skipped_if_router_disallows",
                    "test_rename_backward_without_permissions",
                    "test_rename_permission_conflict",
                    "test_permission_rename_respects_other_db",
                    "test_verbosity_prints",
                ],
                provenance="Django Issue #37361 & PermissionRenameOperationsTests audit (tests/auth_tests/test_management.py)",
                notes="Verified suite of 8 genuine behavioral tests exercising rename_permissions_after_model_rename.",
            ),
        )

        # 3. Home Assistant Core (Real-World False-Positive Elimination)
        ha_path = str(Path.home() / ".testpilot_repos" / "core")
        self.cases["homeassistant_hue_init"] = BenchmarkCase(
            case_id="homeassistant_hue_init",
            name="Home Assistant Hue Constructor Matching (Collision Elimination)",
            repository="home-assistant/core",
            repository_path=ha_path,
            base_commit="8b54f0db5a91d5542a339fbb62de45c694598ac2",
            target_commit="42706c6bd625a800fb324581b25df73e163223ea",
            description="HueButtonEventEntity.__init__ modified. Demonstrates elimination of generic __init__ token collisions.",
            category="real_world",
            total_tests_suite=48462,
            ground_truth=GroundTruth(
                case_id="homeassistant_hue_init",
                expected_changed_symbols=["HueButtonEventEntity.__init__", "__init__"],
                expected_impacted_symbols=[],
                expected_test_callers=[],  # 0 existing test callers in base suite; new test added in target commit
                provenance="Home Assistant Core Hue Integration ground-truth inspection",
                notes="Critical negative control: naive name matching falsely selects 19 tests due to generic __init__ overlap.",
            ),
        )

        # 4. Controlled Validation: Testbed Discount Deficit
        self.cases["testbed_discount_deficit"] = BenchmarkCase(
            case_id="testbed_discount_deficit",
            name="Testbed OrderService Discount Deficit",
            repository="testbed/order_service",
            repository_path=str(Path(".").resolve()),
            base_commit="main",
            target_commit="HEAD",
            description="Controlled testbed scenario: OrderService.calculate_discount boundary bug.",
            category="controlled_validation",
            total_tests_suite=133,
            ground_truth=GroundTruth(
                case_id="testbed_discount_deficit",
                expected_changed_symbols=["calculate_discount"],
                expected_impacted_symbols=["calculate_order_totals", "apply_coupon", "create_order"],
                expected_test_callers=[
                    "test_bug_1_discount_deficit_negative_total",
                    "test_valid_coupon",
                    "test_invalid_coupon",
                    "test_empty_cart",
                    "test_subtotal_below_free_shipping_threshold",
                    "test_subtotal_at_free_shipping_threshold",
                    "test_subtotal_above_free_shipping_threshold",
                ],
                provenance="Controlled testbed specification in testpilot/evolution/evaluator.py",
                notes="Unit-level controlled validation testbed.",
            ),
        )

        # 5. Controlled Validation: Testbed Tax Precision
        self.cases["testbed_tax_precision"] = BenchmarkCase(
            case_id="testbed_tax_precision",
            name="Testbed OrderService Tax Precision",
            repository="testbed/order_service",
            repository_path=str(Path(".").resolve()),
            base_commit="main",
            target_commit="HEAD",
            description="Controlled testbed scenario: OrderService.calculate_tax truncation discrepancy.",
            category="controlled_validation",
            total_tests_suite=14,
            ground_truth=GroundTruth(
                case_id="testbed_tax_precision",
                expected_changed_symbols=["calculate_tax"],
                expected_impacted_symbols=["calculate_order_totals", "create_order"],
                expected_test_callers=[
                    "test_bug_2_tax_truncation_rounding",
                ],
                provenance="Controlled testbed specification in testpilot/evolution/evaluator.py",
                notes="Unit-level controlled validation testbed.",
            ),
        )

    def get_benchmark_cases(self) -> list[BenchmarkCase]:
        """Returns all configured benchmark cases."""
        return list(self.cases.values())

    def get_benchmark_case(self, case_id: str) -> Optional[BenchmarkCase]:
        """Retrieves a specific benchmark case by ID."""
        return self.cases.get(case_id)

    def save_run(self, run: EvaluationRun) -> None:
        """Saves or updates an evaluation run in persistent storage."""
        runs = self.load_runs()
        runs[run.case_id] = run
        data = {k: v.model_dump() for k, v in runs.items()}
        self.runs_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_runs(self) -> dict[str, EvaluationRun]:
        """Loads all persisted evaluation runs."""
        if not self.runs_file.exists():
            return {}
        try:
            raw = json.loads(self.runs_file.read_text(encoding="utf-8"))
            return {k: EvaluationRun(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def get_latest_run(self, case_id: str) -> Optional[EvaluationRun]:
        """Retrieves the latest run for a given benchmark case, or None if not yet evaluated."""
        runs = self.load_runs()
        return runs.get(case_id)
