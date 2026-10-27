"""
TestPilot Capability Detector & Ecosystem Intelligence Engine.
Performs deterministic inspection of repository manifests, workspaces, test frameworks,
external services, and computes per-stage pipeline capabilities with explicit justifications.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from typing import Any, Optional

from testpilot.capabilities.models import (
    EcosystemType,
    ExternalServiceRequirement,
    ManifestInfo,
    PipelineStage,
    RepositoryCompatibilityReport,
    RepositoryEcosystem,
    StageCapability,
    SupportLevel,
    WorkspaceStructure,
)

logger = logging.getLogger("testpilot.capabilities.detector")


class CapabilityDetector:
    """
    Analyzes codebases to identify ecosystem types, manifest files, workspace layouts,
    test frameworks, and backing services. Classifies readiness across all 7 pipeline stages.
    """

    IGNORE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "vendor",
    }

    @classmethod
    def inspect_repository(cls, repo_path: Path | str) -> RepositoryCompatibilityReport:
        """Alias for generate_report."""
        return cls.generate_report(Path(repo_path).resolve())

    @classmethod
    def generate_report(cls, repo_path: Path) -> RepositoryCompatibilityReport:
        """
        Conducts a comprehensive capability audit of the specified repository path.
        """
        canonical = Path(repo_path).resolve()
        repo_name = canonical.name or "repository"

        if not canonical.exists():
            # Inaccessible / non-existent path
            ecosystem = RepositoryEcosystem(primary_language="Unknown")
            unsupported_cap = StageCapability(
                stage=PipelineStage.ONBOARDING,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason=f"Repository path does not exist on disk: {canonical}",
                missing_adapter=None,
                recommendations=["Verify repository path exists and is readable."],
            )
            stage_caps = {stage.value: unsupported_cap for stage in PipelineStage}
            return RepositoryCompatibilityReport(
                repo_name=repo_name,
                canonical_path=str(canonical),
                ecosystem=ecosystem,
                stage_capabilities=stage_caps,
                overall_support_level=SupportLevel.UNSUPPORTED,
                can_run_pipeline=False,
                summary=f"Repository path '{canonical}' does not exist on local filesystem.",
                unsupported_stages=[s.value for s in PipelineStage],
                blocking_reasons=[f"Repository path does not exist: {canonical}"],
                recommended_actions=["Check repository path and file permissions."],
            )

        is_git = (canonical / ".git").exists()
        manifests = cls.detect_manifests(canonical)
        workspace = cls.detect_workspace_structure(canonical, manifests)
        test_frameworks = cls.detect_test_frameworks(canonical, manifests)
        external_services = cls.detect_external_services(canonical, manifests)
        languages, primary_lang = cls.detect_languages(canonical, manifests)
        env_files = cls.detect_env_files(canonical)
        ci_configs = cls.detect_ci_configs(canonical)

        ecosystem = RepositoryEcosystem(
            primary_language=primary_lang,
            detected_languages=languages,
            manifests=manifests,
            workspace=workspace,
            detected_test_frameworks=test_frameworks,
            external_services=external_services,
            env_files=env_files,
            ci_configs=ci_configs,
        )

        stage_caps = cls.assess_stage_capabilities(canonical, ecosystem, is_git=is_git)

        unsupported_stages = [
            s for s, cap in stage_caps.items() if cap.support_level == SupportLevel.UNSUPPORTED
        ]
        partially_supported_stages = [
            s for s, cap in stage_caps.items() if cap.support_level == SupportLevel.PARTIALLY_SUPPORTED
        ]

        blocking_reasons = []
        for stage_name, cap in stage_caps.items():
            if not cap.available:
                blocking_reasons.append(f"[{stage_name.upper()}] {cap.reason}")

        # Compute overall support level
        if not unsupported_stages and not partially_supported_stages:
            overall_support = SupportLevel.FULLY_SUPPORTED
            can_run = True
            summary = (
                f"Repository '{repo_name}' is fully supported across all {len(PipelineStage)} "
                f"pipeline stages ({primary_lang} ecosystem)."
            )
        elif not stage_caps[PipelineStage.ONBOARDING.value].available or not stage_caps[PipelineStage.ANALYSIS.value].available:
            overall_support = SupportLevel.UNSUPPORTED
            can_run = False
            summary = (
                f"Repository '{repo_name}' cannot be processed. Core onboarding/analysis is unsupported "
                f"for {primary_lang}."
            )
        elif unsupported_stages:
            overall_support = SupportLevel.PARTIALLY_SUPPORTED
            can_run = stage_caps[PipelineStage.ONBOARDING.value].available and stage_caps[PipelineStage.TEST_DISCOVERY.value].available
            summary = (
                f"Repository '{repo_name}' is partially supported ({primary_lang}). "
                f"{len(unsupported_stages)} pipeline stages have missing adapters or requirements: "
                f"{', '.join(unsupported_stages)}."
            )
        else:
            overall_support = SupportLevel.PARTIALLY_SUPPORTED
            can_run = True
            summary = (
                f"Repository '{repo_name}' is partially supported ({primary_lang}) with degraded "
                f"features in: {', '.join(partially_supported_stages)}."
            )

        recommended_actions = []
        for cap in stage_caps.values():
            recommended_actions.extend(cap.recommendations)

        # Deduplicate recommendations preserving order
        dedup_recs = list(dict.fromkeys(recommended_actions))

        return RepositoryCompatibilityReport(
            repo_name=repo_name,
            canonical_path=str(canonical),
            ecosystem=ecosystem,
            stage_capabilities=stage_caps,
            overall_support_level=overall_support,
            can_run_pipeline=can_run,
            summary=summary,
            unsupported_stages=unsupported_stages,
            partially_supported_stages=partially_supported_stages,
            blocking_reasons=blocking_reasons,
            recommended_actions=dedup_recs,
        )

    # -------------------------------------------------------------------------
    # Manifest & Workspace Discovery
    # -------------------------------------------------------------------------

    @classmethod
    def detect_manifests(cls, repo_path: Path) -> list[ManifestInfo]:
        """Scans for manifest files across the repository, parsing dependencies and scripts."""
        manifests: list[ManifestInfo] = []

        # Find manifests within top 3 directory depths
        manifest_filenames = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "Pipfile",
            "package.json",
            "tsconfig.json",
            "pnpm-workspace.yaml",
        }

        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if d not in cls.IGNORE_DIRS]
            rel_dir = Path(root).relative_to(repo_path)
            # Limit depth to 3 levels for manifest discovery
            if len(rel_dir.parts) > 3:
                continue

            for f in files:
                if f in manifest_filenames:
                    file_path = Path(root) / f
                    rel_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
                    info = cls._parse_manifest_file(file_path, rel_path)
                    if info:
                        manifests.append(info)

        return manifests

    @classmethod
    def _parse_manifest_file(cls, path: Path, rel_path: str) -> Optional[ManifestInfo]:
        filename = path.name
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        if filename == "pyproject.toml":
            return cls._parse_pyproject_toml(content, rel_path)
        elif filename == "requirements.txt" or filename == "requirements-dev.txt":
            return cls._parse_requirements_txt(content, rel_path, filename)
        elif filename == "setup.py" or filename == "setup.cfg":
            return ManifestInfo(
                manifest_type=filename,
                path=rel_path,
                ecosystem="python",
                package_manager="setuptools/pip",
                dependencies=[],
                dev_dependencies=[],
                test_runner_command="pytest",
            )
        elif filename == "package.json":
            return cls._parse_package_json(content, rel_path)
        elif filename == "tsconfig.json":
            return ManifestInfo(
                manifest_type="tsconfig.json",
                path=rel_path,
                ecosystem="typescript",
                package_manager="npm/yarn/pnpm",
                dependencies=[],
                dev_dependencies=[],
                test_runner_command=None,
            )
        elif filename == "pnpm-workspace.yaml":
            return ManifestInfo(
                manifest_type="pnpm-workspace.yaml",
                path=rel_path,
                ecosystem="javascript/typescript",
                package_manager="pnpm",
                dependencies=[],
                dev_dependencies=[],
                test_runner_command="pnpm test",
            )
        return None

    @classmethod
    def _parse_pyproject_toml(cls, content: str, rel_path: str) -> ManifestInfo:
        deps: list[str] = []
        dev_deps: list[str] = []
        pkg_mgr = "pip"

        try:
            parsed = tomllib.loads(content)
            # Standard PEP 621 [project.dependencies]
            project = parsed.get("project", {})
            if "dependencies" in project and isinstance(project["dependencies"], list):
                for d in project["dependencies"]:
                    m = re.match(r"^([a-zA-Z0-9_\-\.]+)", str(d).strip())
                    if m:
                        deps.append(m.group(1))
            optional = project.get("optional-dependencies", {})
            for grp, grp_deps in optional.items():
                if isinstance(grp_deps, list):
                    for d in grp_deps:
                        m = re.match(r"^([a-zA-Z0-9_\-\.]+)", str(d).strip())
                        if m:
                            dev_deps.append(m.group(1))

            # Poetry [tool.poetry.dependencies]
            tool_poetry = parsed.get("tool", {}).get("poetry", {})
            if tool_poetry:
                pkg_mgr = "poetry"
                for k in tool_poetry.get("dependencies", {}).keys():
                    if k.lower() != "python":
                        deps.append(k)
                for k in tool_poetry.get("dev-dependencies", {}).keys():
                    dev_deps.append(k)
                groups = tool_poetry.get("group", {})
                for gname, gdata in groups.items():
                    for k in gdata.get("dependencies", {}).keys():
                        dev_deps.append(k)

            # Flit / Hatch
            if "tool" in parsed and "hatch" in parsed["tool"]:
                pkg_mgr = "hatch"
            elif "tool" in parsed and "flit" in parsed["tool"]:
                pkg_mgr = "flit"

        except Exception as e:
            logger.debug("Failed parsing %s with tomllib: %s", rel_path, e)

        return ManifestInfo(
            manifest_type="pyproject.toml",
            path=rel_path,
            ecosystem="python",
            package_manager=pkg_mgr,
            dependencies=sorted(set(deps)),
            dev_dependencies=sorted(set(dev_deps)),
            test_runner_command="pytest",
        )

    @classmethod
    def _parse_requirements_txt(cls, content: str, rel_path: str, filename: str) -> ManifestInfo:
        deps: list[str] = []
        for line in content.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#") or cleaned.startswith("-r"):
                continue
            # Extract package name before ==, >=, <=, etc.
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)", cleaned)
            if match:
                deps.append(match.group(1))

        is_dev = "dev" in filename.lower() or "test" in filename.lower()
        return ManifestInfo(
            manifest_type=filename,
            path=rel_path,
            ecosystem="python",
            package_manager="pip",
            dependencies=[] if is_dev else sorted(set(deps)),
            dev_dependencies=sorted(set(deps)) if is_dev else [],
            test_runner_command="pytest",
        )

    @classmethod
    def _parse_package_json(cls, content: str, rel_path: str) -> ManifestInfo:
        deps: list[str] = []
        dev_deps: list[str] = []
        test_cmd = None
        pkg_mgr = "npm"

        try:
            data = json.loads(content)
            deps = list(data.get("dependencies", {}).keys())
            dev_deps = list(data.get("devDependencies", {}).keys())
            scripts = data.get("scripts", {})
            test_cmd = scripts.get("test")
            # Detect package manager
            if "pnpm" in str(data.get("packageManager", "")):
                pkg_mgr = "pnpm"
            elif "yarn" in str(data.get("packageManager", "")):
                pkg_mgr = "yarn"
        except Exception as e:
            logger.debug("Failed parsing package.json at %s: %s", rel_path, e)

        return ManifestInfo(
            manifest_type="package.json",
            path=rel_path,
            ecosystem="javascript/typescript",
            package_manager=pkg_mgr,
            dependencies=sorted(deps),
            dev_dependencies=sorted(dev_deps),
            test_runner_command=test_cmd or "npm test",
        )



    @classmethod
    def detect_workspace_structure(
        cls,
        repo_path: Path,
        manifests: list[ManifestInfo],
    ) -> WorkspaceStructure:
        """Determines if repository is a monorepo or single package structure."""
        # 1. Check pnpm workspace
        pnpm_ws = repo_path / "pnpm-workspace.yaml"
        if pnpm_ws.exists():
            packages = cls._find_subpackages(repo_path, ["packages", "apps", "libs"])
            return WorkspaceStructure(
                is_monorepo=True,
                workspace_type="pnpm_workspace",
                root_manifest="pnpm-workspace.yaml",
                packages=packages,
            )

        # 2. Check root package.json for workspaces
        root_pkg = repo_path / "package.json"
        if root_pkg.exists():
            try:
                pkg_data = json.loads(root_pkg.read_text(encoding="utf-8", errors="ignore"))
                workspaces = pkg_data.get("workspaces")
                if workspaces:
                    packages = cls._find_subpackages(repo_path, ["packages", "apps", "services"])
                    return WorkspaceStructure(
                        is_monorepo=True,
                        workspace_type="npm_workspaces",
                        root_manifest="package.json",
                        packages=packages,
                    )
            except Exception:
                pass

        # 3. Check if multiple sub-manifests exist under packages/ or services/
        sub_manifests = [m for m in manifests if "/" in m.path]
        if len(sub_manifests) >= 2 and any(
            m.path.startswith(("packages/", "apps/", "services/", "modules/"))
            for m in sub_manifests
        ):
            pkgs = [
                {
                    "name": Path(m.path).parent.name,
                    "path": str(Path(m.path).parent).replace("\\", "/"),
                    "manifest": m.path,
                }
                for m in sub_manifests
            ]
            return WorkspaceStructure(
                is_monorepo=True,
                workspace_type="multi_package",
                root_manifest=manifests[0].path if manifests else None,
                packages=pkgs,
            )

        # Single package layout
        return WorkspaceStructure(
            is_monorepo=False,
            workspace_type="single_package",
            root_manifest=manifests[0].path if manifests else None,
            packages=[],
        )

    @classmethod
    def _find_subpackages(cls, repo_path: Path, candidate_roots: list[str]) -> list[dict[str, str]]:
        packages = []
        for cr in candidate_roots:
            dir_path = repo_path / cr
            if dir_path.is_dir():
                for sub in dir_path.iterdir():
                    if sub.is_dir() and not sub.name.startswith("."):
                        # Look for manifest
                        manifest_name = None
                        for candidate_manifest in ("package.json", "pyproject.toml"):
                            if (sub / candidate_manifest).exists():
                                manifest_name = f"{cr}/{sub.name}/{candidate_manifest}"
                                break
                        packages.append({
                            "name": sub.name,
                            "path": f"{cr}/{sub.name}",
                            "manifest": manifest_name or f"{cr}/{sub.name}",
                        })
        return packages

    # -------------------------------------------------------------------------
    # Test Framework & External Service Detection
    # -------------------------------------------------------------------------

    @classmethod
    def detect_test_frameworks(
        cls,
        repo_path: Path,
        manifests: list[ManifestInfo],
    ) -> list[str]:
        """Detects all test runner and assertion frameworks across languages."""
        frameworks: set[str] = set()

        # Combine all dependencies across manifests
        all_deps = set()
        for m in manifests:
            all_deps.update(d.lower() for d in m.dependencies)
            all_deps.update(d.lower() for d in m.dev_dependencies)

        # Python frameworks
        if "pytest" in all_deps or (repo_path / "conftest.py").exists() or (repo_path / "pytest.ini").exists():
            frameworks.add("pytest")
        if "unittest" in all_deps:
            frameworks.add("unittest")

        # JS/TS frameworks
        if "jest" in all_deps or any((repo_path / f).exists() for f in ["jest.config.js", "jest.config.ts", "jest.config.json"]):
            frameworks.add("jest")
        if "vitest" in all_deps or any((repo_path / f).exists() for f in ["vitest.config.ts", "vitest.config.js"]):
            frameworks.add("vitest")
        if "mocha" in all_deps:
            frameworks.add("mocha")
        if "playwright" in all_deps or "@playwright/test" in all_deps:
            frameworks.add("playwright")
        if "cypress" in all_deps:
            frameworks.add("cypress")



        # Fallback inspection for Python test files if not detected from dependencies
        if not frameworks:
            for root, dirs, files in os.walk(str(repo_path)):
                dirs[:] = [d for d in dirs if d not in cls.IGNORE_DIRS]
                if any(f.startswith("test_") and f.endswith(".py") for f in files):
                    frameworks.add("pytest")
                    break

        return sorted(frameworks)

    @classmethod
    def detect_external_services(
        cls,
        repo_path: Path,
        manifests: list[ManifestInfo],
    ) -> list[ExternalServiceRequirement]:
        """Detects required external databases, caches, queues, and container services."""
        requirements: list[ExternalServiceRequirement] = []

        all_deps = set()
        for m in manifests:
            all_deps.update(d.lower() for d in m.dependencies)

        # 1. Docker Compose scanning
        compose_files = ["docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"]
        for cf in compose_files:
            cp = repo_path / cf
            if cp.exists():
                try:
                    text = cp.read_text(encoding="utf-8", errors="ignore").lower()
                    if "postgres" in text or "timescale" in text:
                        requirements.append(
                            ExternalServiceRequirement(
                                service_name="postgresql",
                                detected_via=f"{cf} (service definition)",
                                is_available=False,
                                is_critical=True,
                                remediation_notes="Requires PostgreSQL instance running on port 5432.",
                            )
                        )
                    if "redis" in text:
                        requirements.append(
                            ExternalServiceRequirement(
                                service_name="redis",
                                detected_via=f"{cf} (service definition)",
                                is_available=False,
                                is_critical=False,
                                remediation_notes="Requires Redis instance running on port 6379.",
                            )
                        )
                    if "mysql" in text or "mariadb" in text:
                        requirements.append(
                            ExternalServiceRequirement(
                                service_name="mysql",
                                detected_via=f"{cf} (service definition)",
                                is_available=False,
                                is_critical=True,
                                remediation_notes="Requires MySQL/MariaDB on port 3306.",
                            )
                        )
                    if "mongo" in text:
                        requirements.append(
                            ExternalServiceRequirement(
                                service_name="mongodb",
                                detected_via=f"{cf} (service definition)",
                                is_available=False,
                                is_critical=True,
                                remediation_notes="Requires MongoDB on port 27017.",
                            )
                        )
                except Exception:
                    pass

        # 2. Dependency scanning for external service clients
        if any(d in all_deps for d in ["psycopg2", "asyncpg", "pg", "sqlalchemy"]):
            if not any(r.service_name == "postgresql" for r in requirements):
                requirements.append(
                    ExternalServiceRequirement(
                        service_name="postgresql",
                        detected_via="database driver dependency",
                        is_available=False,
                        is_critical=False,
                        remediation_notes="PostgreSQL database client detected in dependencies.",
                    )
                )

        if any(d in all_deps for d in ["redis", "aioredis", "ioredis"]):
            if not any(r.service_name == "redis" for r in requirements):
                requirements.append(
                    ExternalServiceRequirement(
                        service_name="redis",
                        detected_via="redis driver dependency",
                        is_available=False,
                        is_critical=False,
                        remediation_notes="Redis cache driver detected in dependencies.",
                    )
                )

        # 3. Docker Daemon Requirement
        docker_available = False
        docker_bin = shutil.which("docker")
        if docker_bin:
            try:
                res = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=False,
                )
                docker_available = (res.returncode == 0 and bool(res.stdout.strip()))
            except Exception:
                docker_available = False

        requirements.append(
            ExternalServiceRequirement(
                service_name="docker_daemon",
                detected_via="TestPilot container sandbox isolation requirement",
                is_available=docker_available,
                is_critical=True,
                remediation_notes=(
                    "Docker Desktop or WSL2 engine must be running to execute tests in an isolated sandbox. "
                    "Host fallback is prohibited."
                    if not docker_available
                    else "Docker daemon is active and responsive."
                ),
            )
        )

        return requirements

    @classmethod
    def detect_languages(
        cls,
        repo_path: Path,
        manifests: list[ManifestInfo],
    ) -> tuple[list[str], str]:
        """Scans source file extensions and manifest data to classify languages."""
        lang_counts: dict[str, int] = {
            "Python": 0,
            "JavaScript": 0,
            "TypeScript": 0,
            "Go": 0,
            "Java": 0,
            "Rust": 0,
            "C/C++": 0,
        }

        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".mjs": "JavaScript",
            ".cjs": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".go": "Go",
            ".java": "Java",
            ".rs": "Rust",
            ".c": "C/C++",
            ".cpp": "C/C++",
            ".h": "C/C++",
        }

        for root, dirs, files in os.walk(str(repo_path)):
            dirs[:] = [d for d in dirs if d not in cls.IGNORE_DIRS]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in ext_map:
                    lang_counts[ext_map[ext]] += 1

        detected = [lang for lang, count in lang_counts.items() if count > 0]

        # Prioritize manifests if extensions are zero
        if not detected and manifests:
            eco_map = {
                "python": "Python",
                "javascript": "JavaScript",
                "typescript": "TypeScript",
                "javascript/typescript": "JavaScript",
                "go": "Go",
                "java": "Java",
                "rust": "Rust",
            }
            for m in manifests:
                if m.ecosystem in eco_map:
                    detected.append(eco_map[m.ecosystem])

        detected = sorted(set(detected))
        if not detected:
            return (["Generic/Text"], "Generic/Text")

        # Determine primary language by highest file count
        primary = max(detected, key=lambda l: lang_counts.get(l, 0))
        return (detected, primary)

    @classmethod
    def detect_env_files(cls, repo_path: Path) -> list[str]:
        env_files = []
        for name in [".env.example", ".env.sample", ".env.test", ".env.template", ".env"]:
            if (repo_path / name).exists():
                env_files.append(name)
        return env_files

    @classmethod
    def detect_ci_configs(cls, repo_path: Path) -> list[str]:
        ci_configs = []
        gh_workflows = repo_path / ".github" / "workflows"
        if gh_workflows.is_dir():
            for f in gh_workflows.glob("*.yml"):
                ci_configs.append(f".github/workflows/{f.name}")
            for f in gh_workflows.glob("*.yaml"):
                ci_configs.append(f".github/workflows/{f.name}")
        for conf in [".travis.yml", ".gitlab-ci.yml", "Jenkinsfile", "azure-pipelines.yml"]:
            if (repo_path / conf).exists():
                ci_configs.append(conf)
        return ci_configs

    # -------------------------------------------------------------------------
    # Pipeline Stage Capability Assessment
    # -------------------------------------------------------------------------

    @classmethod
    def assess_stage_capabilities(
        cls,
        repo_path: Path,
        ecosystem: RepositoryEcosystem,
        is_git: bool,
    ) -> dict[str, StageCapability]:
        """
        Evaluates each of the 7 discrete pipeline stages.
        Assigns FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED with clear diagnostics.
        """
        caps: dict[str, StageCapability] = {}
        primary = ecosystem.primary_language
        has_python = "Python" in ecosystem.detected_languages
        has_js_ts = any(l in ecosystem.detected_languages for l in ("JavaScript", "TypeScript"))

        # Check Docker daemon availability from external_services
        docker_req = next((s for s in ecosystem.external_services if s.service_name == "docker_daemon"), None)
        docker_available = docker_req.is_available if docker_req else False

        # 1. ONBOARDING
        caps[PipelineStage.ONBOARDING.value] = StageCapability(
            stage=PipelineStage.ONBOARDING,
            support_level=SupportLevel.FULLY_SUPPORTED,
            available=True,
            reason="Repository files and git structure are verified and accessible.",
            missing_adapter=None,
            missing_services=[],
            diagnostics={"is_git": is_git, "primary_language": primary},
            recommendations=[],
        )

        # 2. ANALYSIS (AST & Code Intelligence)
        if primary == "Python" or has_python:
            caps[PipelineStage.ANALYSIS.value] = StageCapability(
                stage=PipelineStage.ANALYSIS,
                support_level=SupportLevel.FULLY_SUPPORTED,
                available=True,
                reason="Full AST, symbol call graph, cyclomatic branch, and boundary analysis supported for Python.",
                missing_adapter=None,
                recommendations=[],
            )
        elif has_js_ts:
            caps[PipelineStage.ANALYSIS.value] = StageCapability(
                stage=PipelineStage.ANALYSIS,
                support_level=SupportLevel.PARTIALLY_SUPPORTED,
                available=True,
                reason="JavaScript/TypeScript symbol extraction supported via structural function/export inspection.",
                missing_adapter="TreeSitter/Babel AST semantic call-graph adapter",
                recommendations=[
                    "Provide explicit function-to-test mapping or configure Babel/TreeSitter parser for JS/TS."
                ],
            )
        else:
            caps[PipelineStage.ANALYSIS.value] = StageCapability(
                stage=PipelineStage.ANALYSIS,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason=f"Language '{primary}' is not supported by TestPilot (scope is restricted to Python and JavaScript/TypeScript only).",
                missing_adapter=f"Language '{primary}' is outside supported scope",
                recommendations=["TestPilot currently supports Python and JavaScript/TypeScript repositories only."],
            )

        # 3. TEST_DISCOVERY
        if ecosystem.detected_test_frameworks:
            if primary == "Python" or has_python:
                caps[PipelineStage.TEST_DISCOVERY.value] = StageCapability(
                    stage=PipelineStage.TEST_DISCOVERY,
                    support_level=SupportLevel.FULLY_SUPPORTED,
                    available=True,
                    reason=f"Native test discovery supported for {', '.join(ecosystem.detected_test_frameworks)}.",
                    missing_adapter=None,
                    recommendations=[],
                )
            elif has_js_ts:
                caps[PipelineStage.TEST_DISCOVERY.value] = StageCapability(
                    stage=PipelineStage.TEST_DISCOVERY,
                    support_level=SupportLevel.FULLY_SUPPORTED,
                    available=True,
                    reason=f"JavaScript/TypeScript test discovery supported for {', '.join(ecosystem.detected_test_frameworks)}.",
                    missing_adapter=None,
                    recommendations=[],
                )
            else:
                caps[PipelineStage.TEST_DISCOVERY.value] = StageCapability(
                    stage=PipelineStage.TEST_DISCOVERY,
                    support_level=SupportLevel.UNSUPPORTED,
                    available=False,
                    reason=f"Test discovery not supported for '{primary}'. Only Python and JavaScript/TypeScript are supported.",
                    missing_adapter=f"Test discovery for {primary}",
                    recommendations=[],
                )
        else:
            caps[PipelineStage.TEST_DISCOVERY.value] = StageCapability(
                stage=PipelineStage.TEST_DISCOVERY,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason="No recognized test framework (pytest, unittest, jest, vitest, mocha) or test files detected.",
                missing_adapter="Test framework discovery adapter",
                recommendations=[
                    "Add standard test files (e.g. test_*.py, *.test.ts) or configure a recognized test framework."
                ],
            )

        # 4. TEST_GENERATION (Autonomous Test Synthesis)
        if primary == "Python" or has_python:
            caps[PipelineStage.TEST_GENERATION.value] = StageCapability(
                stage=PipelineStage.TEST_GENERATION,
                support_level=SupportLevel.FULLY_SUPPORTED,
                available=True,
                reason="Autonomous test synthesis and AST boundary test generation fully operational for Python pytest.",
                missing_adapter=None,
                recommendations=[],
            )
        else:
            caps[PipelineStage.TEST_GENERATION.value] = StageCapability(
                stage=PipelineStage.TEST_GENERATION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason=(
                    f"Autonomous test synthesis adapter is not implemented for '{primary}'. "
                    "Benchmark evaluation requires pre-existing tests or historical bug-fix ground truth."
                ),
                missing_adapter=f"Autonomous test generator for {primary}",
                recommendations=[
                    f"Use Method A (Historical Bug Mining) or manual benchmark ground-truth definition for {primary} projects."
                ],
            )

        # 5. GROUND_TRUTH_VALIDATION
        if not is_git:
            caps[PipelineStage.GROUND_TRUTH_VALIDATION.value] = StageCapability(
                stage=PipelineStage.GROUND_TRUTH_VALIDATION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason="Target folder is not a Git repository. Commit-level bug-fix ground truth mining requires Git history.",
                missing_adapter=None,
                recommendations=["Initialize a Git repository and commit changes to enable ground truth mining."],
            )
        elif primary == "Python" or has_python:
            caps[PipelineStage.GROUND_TRUTH_VALIDATION.value] = StageCapability(
                stage=PipelineStage.GROUND_TRUTH_VALIDATION,
                support_level=SupportLevel.FULLY_SUPPORTED,
                available=True,
                reason="Historical bug-fix mining, parent commit checkout, and test overlay validation fully supported.",
                missing_adapter=None,
                recommendations=[],
            )
        else:
            caps[PipelineStage.GROUND_TRUTH_VALIDATION.value] = StageCapability(
                stage=PipelineStage.GROUND_TRUTH_VALIDATION,
                support_level=SupportLevel.PARTIALLY_SUPPORTED,
                available=True,
                reason=(
                    f"Git commit diff mining is supported for {primary}, but containerized test overlay execution "
                    f"requires an ecosystem test runner adapter."
                ),
                missing_adapter=f"Overlay test execution adapter for {primary}",
                recommendations=[f"Review mined commit diffs manually or attach an external test runner for {primary}."],
            )

        # 6. DOCKER_EXECUTION
        if not docker_available:
            caps[PipelineStage.DOCKER_EXECUTION.value] = StageCapability(
                stage=PipelineStage.DOCKER_EXECUTION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason="Docker daemon is not running or not responsive. Isolated container execution is disabled; silent host fallback is prohibited.",
                missing_adapter=None,
                missing_services=["docker_daemon"],
                recommendations=["Start Docker Desktop or the WSL2 Docker daemon before running tests."],
            )
        elif primary == "Python" or has_python:
            caps[PipelineStage.DOCKER_EXECUTION.value] = StageCapability(
                stage=PipelineStage.DOCKER_EXECUTION,
                support_level=SupportLevel.FULLY_SUPPORTED,
                available=True,
                reason="Disposable Docker container sandbox with read-only repository mount active for Python pytest.",
                missing_adapter=None,
                recommendations=[],
            )
        elif has_js_ts:
            caps[PipelineStage.DOCKER_EXECUTION.value] = StageCapability(
                stage=PipelineStage.DOCKER_EXECUTION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason="JavaScript/TypeScript Docker test runner image not configured. Only Python pytest image is loaded.",
                missing_adapter="Node.js Docker test runner adapter and image",
                recommendations=["Configure a Node.js test runner Docker image for isolated Jest/Vitest execution."],
            )
        else:
            caps[PipelineStage.DOCKER_EXECUTION.value] = StageCapability(
                stage=PipelineStage.DOCKER_EXECUTION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason=f"No container sandbox runner exists for '{primary}' (supported scope: Python and JavaScript/TypeScript only).",
                missing_adapter=f"Docker test runner for {primary}",
                recommendations=["TestPilot sandbox execution is restricted to Python and JavaScript/TypeScript."],
            )

        # 7. EVALUATION
        discovery_avail = caps[PipelineStage.TEST_DISCOVERY.value].available
        if not discovery_avail:
            caps[PipelineStage.EVALUATION.value] = StageCapability(
                stage=PipelineStage.EVALUATION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason="No tests discovered; regression selection evaluation cannot be executed without test suite.",
                missing_adapter=None,
                recommendations=["Ensure test discovery finds valid test targets before triggering evaluation."],
            )
        elif primary == "Python" or has_python:
            caps[PipelineStage.EVALUATION.value] = StageCapability(
                stage=PipelineStage.EVALUATION,
                support_level=SupportLevel.FULLY_SUPPORTED,
                available=True,
                reason="Full Regression, Naive, and TestPilot selection evaluation baselines fully supported.",
                missing_adapter=None,
                recommendations=[],
            )
        elif has_js_ts:
            caps[PipelineStage.EVALUATION.value] = StageCapability(
                stage=PipelineStage.EVALUATION,
                support_level=SupportLevel.PARTIALLY_SUPPORTED,
                available=True,
                reason=(
                    f"Test selection baselines can run against discovered {primary} tests, "
                    "but dynamic runtime fault-injection requires containerized execution adapter."
                ),
                missing_adapter=f"Runtime dynamic evaluation runner for {primary}",
                recommendations=["Test selection algorithms can be benchmarked against static test suites."],
            )
        else:
            caps[PipelineStage.EVALUATION.value] = StageCapability(
                stage=PipelineStage.EVALUATION,
                support_level=SupportLevel.UNSUPPORTED,
                available=False,
                reason=f"Evaluation is not supported for '{primary}' (supported scope: Python and JavaScript/TypeScript only).",
                missing_adapter=f"Evaluation runner for {primary}",
                recommendations=["TestPilot evaluation is restricted to Python and JavaScript/TypeScript."],
            )

        return caps
