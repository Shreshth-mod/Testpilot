"""
Comprehensive test suite for the 30-Question Quantitative Analysis Matrix
and Qualitative Analysis Findings implementation.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient
from testpilot.web.api import app, custom_repo_mgr
from testpilot.evaluation.custom_models import CustomRepositoryRecord

JSON_PATH = Path("testpilot/evaluation/quantitative_evaluation_questions.json")
INDEX_HTML_PATH = Path("testpilot/web/static/index.html")

REQUIRED_TOP_LEVEL_KEYS = [
    "title",
    "description",
    "version",
    "project",
    "total_questions",
    "question_categories",
    "evaluation_configurations",
    "aggregation_policy",
    "questions"
]

REQUIRED_QUESTION_FIELDS = [
    "id",
    "category",
    "question",
    "objective",
    "metrics",
    "required_data",
    "methodology",
    "formula",
    "unit",
    "comparison",
    "aggregation",
    "interpretation",
    "limitations",
    "answerability_requirements",
    "status",
    "result_source"
]

REQUIRED_METRIC_FIELDS = [
    "name",
    "definition",
    "formula",
    "unit",
    "required_data"
]

CATEGORY_RANGES = {
    "Category A — Test Selection Quality": [f"Q{i:02d}" for i in range(1, 7)],
    "Category B — Dependency and Impact Analysis": [f"Q{i:02d}" for i in range(7, 13)],
    "Category C — Repository RAG and Evidence Retrieval": [f"Q{i:02d}" for i in range(13, 19)],
    "Category D — Test Generation and Ground Truth": [f"Q{i:02d}" for i in range(19, 23)],
    "Category E — Performance and Scalability": [f"Q{i:02d}" for i in range(23, 26)],
    "Category F — Reliability and Generalization": [f"Q{i:02d}" for i in range(26, 29)],
    "Category G — Evaluation Validity and Usability": [f"Q{i:02d}" for i in range(29, 31)]
}

REQUIRED_CONFIGS = [
    "full_regression",
    "naive_baseline",
    "testpilot",
    "testpilot_rag"
]


def test_quantitative_json_existence_and_parsing():
    assert JSON_PATH.exists(), f"File does not exist at {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "Top-level structure must be a JSON object"
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in data, f"Missing required top-level key: {key}"


def test_quantitative_json_exact_30_questions_and_ids():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["total_questions"] == 30
    questions = data["questions"]
    assert len(questions) == 30, f"Expected 30 questions, found {len(questions)}"

    expected_ids = [f"Q{i:02d}" for i in range(1, 31)]
    actual_ids = [q["id"] for q in questions]
    assert actual_ids == expected_ids, f"Mismatch in question IDs: {actual_ids} vs {expected_ids}"


def test_quantitative_json_categories_distribution():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    category_map = {}
    for q in questions:
        category_map.setdefault(q["category"], []).append(q["id"])

    assert len(category_map) == 7, f"Expected 7 categories, got {len(category_map)}"

    for cat_name, expected_ids in CATEGORY_RANGES.items():
        assert cat_name in category_map, f"Category '{cat_name}' not found"
        assert category_map[cat_name] == expected_ids, (
            f"Category '{cat_name}' questions mismatch: got {category_map[cat_name]}, expected {expected_ids}"
        )


def test_quantitative_json_fields_and_no_fabricated_results():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check configurations
    configs = data["evaluation_configurations"]
    config_ids = [c["id"] for c in configs]
    for req_cfg in REQUIRED_CONFIGS:
        assert req_cfg in config_ids, f"Required configuration '{req_cfg}' not found"

    for q in data["questions"]:
        qid = q["id"]
        # Required fields
        for field in REQUIRED_QUESTION_FIELDS:
            assert field in q, f"Question {qid} missing field '{field}'"

        # Status must be not_measured
        assert q["status"] == "not_measured", f"Question {qid} status should be 'not_measured', got {q['status']}"

        # Metrics verification
        assert isinstance(q["metrics"], list) and len(q["metrics"]) > 0, f"Question {qid} has no metrics"
        for metric in q["metrics"]:
            for mfield in REQUIRED_METRIC_FIELDS:
                assert mfield in metric, f"Metric in {qid} missing field '{mfield}'"
            # formula can be null or string
            assert metric["formula"] is None or isinstance(metric["formula"], str)


def test_api_quantitative_questions_endpoint():
    client = TestClient(app)
    response = client.get("/api/evaluation/quantitative-questions")
    assert response.status_code == 200
    data = response.json()
    assert data["total_questions"] == 30
    assert len(data["questions"]) == 30
    assert data["questions"][0]["id"] == "Q01"
    assert data["questions"][29]["id"] == "Q30"


def test_api_qualitative_findings_endpoint_global():
    client = TestClient(app)
    response = client.get("/api/evaluation/qualitative-findings")
    assert response.status_code == 200
    data = response.json()
    assert "findings" in data
    findings = data["findings"]
    assert len(findings) == 7, f"Expected 7 findings across 7 categories, got {len(findings)}"

    expected_categories = [
        "Test Selection Behavior",
        "Structural and Semantic Impact Analysis",
        "Repository RAG Contributions",
        "Test Generation and Ground Truth",
        "Performance and Reliability",
        "Cross-Repository Differences",
        "Limitations and Improvement Opportunities"
    ]
    actual_categories = [f["category"] for f in findings]
    assert actual_categories == expected_categories

    for f in findings:
        for field in ["finding_id", "title", "category", "observation", "supporting_evidence", "interpretation", "limitations", "confidence", "status"]:
            assert field in f, f"Finding missing field: {field}"
        assert f["status"] in ["evidence_supported", "preliminary", "insufficient_evidence"]
        assert f["confidence"] in ["high", "medium", "low"]
        assert len(f["observation"]) > 0
        assert len(f["supporting_evidence"]) > 0


def test_api_qualitative_findings_insufficient_evidence():
    client = TestClient(app)

    # 1. Non-existent repo returns 404
    resp_404 = client.get("/api/evaluation/qualitative-findings?repo_id=nonexistent_dummy_repo_999")
    assert resp_404.status_code == 404

    # 2. Existing repo with NO evaluated runs returns insufficient_evidence for all findings
    test_record = CustomRepositoryRecord(
        repo_id="test_repo_unevaluated_xyz",
        name="Test Unevaluated Repo",
        canonical_path="C:/tmp/test_repo",
        latest_evaluation_runs={}
    )
    records = custom_repo_mgr._load_all_records()
    records[test_record.repo_id] = test_record
    custom_repo_mgr._save_all_records(records)

    try:
        response = client.get(f"/api/evaluation/qualitative-findings?repo_id={test_record.repo_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["has_sufficient_evidence"] is False
        assert len(data["findings"]) == 7
        expected_msg = "Insufficient validated evaluation evidence to establish this finding. Run eligible benchmarks and validate the relevant ground-truth cases."
        for f in data["findings"]:
            assert f["status"] == "insufficient_evidence"
            assert f["observation"] == expected_msg
            assert f["confidence"] == "low"
    finally:
        # Cleanup
        records = custom_repo_mgr._load_all_records()
        if test_record.repo_id in records:
            del records[test_record.repo_id]
            custom_repo_mgr._save_all_records(records)


def test_ui_html_sections_and_placement():
    assert INDEX_HTML_PATH.exists()
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # Verify presence of Quantitative Comparison section
    assert 'id="quantitative-comparison-section"' in content
    assert "Quantitative Comparison" in content

    # Verify presence of Qualitative Analysis section
    assert 'id="qualitative-analysis-section"' in content
    assert "Qualitative Analysis Findings" in content

    # Verify presence of Quantitative Matrix reference section
    assert 'id="quantitative-matrix-section"' in content
    assert "Quantitative Analysis" in content
    assert "30-Question Matrix" in content

    # Verify modal is present
    assert 'id="modal-quantitative-questions"' in content

    # Verify exact ordering:
    # 1. Quantitative Comparison (Section 4)
    # 2. Qualitative Analysis Findings (Section 4b)
    # 3. Quantitative Analysis — 30-Question Matrix Reference (Section 4c)
    pos_quant_comp = content.find('id="quantitative-comparison-section"')
    pos_qual_findings = content.find('id="qualitative-analysis-section"')
    pos_quant_matrix = content.find('id="quantitative-matrix-section"')

    assert pos_quant_comp != -1, "Quantitative Comparison section not found"
    assert pos_qual_findings != -1, "Qualitative Analysis Findings section not found"
    assert pos_quant_matrix != -1, "Quantitative Matrix Reference section not found"

    assert pos_quant_comp < pos_qual_findings, (
        "Qualitative Analysis Findings must be placed immediately after Quantitative Comparison"
    )
    assert pos_qual_findings < pos_quant_matrix, (
        "Quantitative Matrix reference must be placed immediately after Qualitative Analysis Findings"
    )

    # Verify JS functions exist
    assert "fetchQualitativeFindings" in content
    assert "renderQualitativeFindings" in content
    assert "openQuantitativeMatrixModal" in content
    assert "closeQuantitativeMatrixModal" in content
