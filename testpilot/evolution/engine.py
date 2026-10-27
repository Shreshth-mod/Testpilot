"""
Repository Evolution Intelligence Engine for TestPilot AI.
Analyzes code evolution between Git revisions, tracks changed symbols,
constructs transitive caller impact graphs, and prioritizes existing tests.
"""

import ast
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from testpilot.ast_engine.treesitter_parser import ASTDiffParser
from testpilot.evolution.events import EventRegistrationDetector
from testpilot.evolution.models import (
    ChangedSymbol,
    ChangeType,
    EventMatchClassification,
    EvidenceItem,
    EvidenceTrail,
    EvolutionReport,
    EvolutionRequest,
    ImpactNode,
    ImpactType,
    PrioritizedTest,
    PriorityTier,
    SemanticImpactAssessment,
    SemanticImpactClassification,
)
from testpilot.sourcegraph.client import LocalCodeGraphFallback, SourcegraphClient


class InvalidGitReferenceError(ValueError):
    """Raised when a user-supplied Git reference cannot be resolved."""
    pass


@dataclass
class CommandDispatchRecord:
    """Detailed record of a recognized command dispatch invocation."""
    command_name: str
    dispatch_api: str
    call_expr: str
    extra_args: list[str] = field(default_factory=list)
    line_number: int = 0


class RepositoryEvolutionEngine:
    """
    Coordinates Git revision diffing, AST symbol correlation,
    transitive blast radius traversal, and test impact prioritization.
    """

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.sg_client = SourcegraphClient(repo_root=str(self.repo_root))
        self.sourcegraph = self.sg_client
        self.event_detector = EventRegistrationDetector(repo_root=str(self.repo_root))
        self.last_renamed_files: dict[str, str] = {}
        self.last_file_statuses: dict[str, str] = {}
        self.last_discovered_tests_count: int = 0
        self.last_event_candidates: list[PrioritizedTest] = []

    def _get_git_file_content(self, ref: str, file_path: str) -> Optional[str]:
        """Safely retrieve file content from Git object storage at given ref."""
        try:
            res = subprocess.run(
                ["git", "show", f"{ref}:{file_path}"],
                cwd=str(self.repo_root),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass
        return None

    def _get_target_file_content(self, t_ref: Optional[str], file_path: str) -> Optional[str]:
        """Safely retrieve file content at target revision (or working tree if t_ref is None)."""
        if t_ref is None:
            p = self.repo_root / file_path
            if p.exists() and p.is_file():
                try:
                    return p.read_text(encoding="utf-8")
                except Exception:
                    return None
            return None
        return self._get_git_file_content(t_ref, file_path)

    def resolve_git_diff(
        self,
        base_ref: str = "main",
        target_ref: Optional[str] = "HEAD",
    ) -> tuple[str, list[str], str]:
        """
        Executes non-destructive git diff between base_ref and target_ref.
        Validates git references and tracks file statuses (including renames).
        Raises InvalidGitReferenceError if user-supplied refs are invalid.
        Returns:
            (unified_diff_text, changed_files_list, diff_stat)
        """
        # Validate that repo_root is a git repository
        try:
            check_repo = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(self.repo_root),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if check_repo.returncode != 0:
                raise InvalidGitReferenceError(f"Directory is not a git repository: {self.repo_root}")
        except FileNotFoundError as err:
            raise InvalidGitReferenceError("Git executable not found in PATH") from err

        # Clean ref inputs
        b_ref = base_ref.strip() if base_ref else "main"
        # Check if comparing working tree against base_ref
        is_working_tree = target_ref is None or target_ref.strip().upper() in ("WORKING_TREE", "DIRTY", "UNCOMMITTED", "")
        t_ref = None if is_working_tree else target_ref.strip()

        # Validate base_ref
        verify_b = subprocess.run(
            ["git", "rev-parse", "--verify", b_ref],
            cwd=str(self.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if verify_b.returncode != 0:
            raise InvalidGitReferenceError(f"Invalid base reference: '{b_ref}'")

        if not is_working_tree and t_ref:
            verify_t = subprocess.run(
                ["git", "rev-parse", "--verify", t_ref],
                cwd=str(self.repo_root),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if verify_t.returncode != 0:
                raise InvalidGitReferenceError(f"Invalid target reference: '{t_ref}'")

        if is_working_tree:
            diff_args = ["git", "--no-pager", "diff", "--unified=3", "-M", b_ref]
            stat_args = ["git", "--no-pager", "diff", "--stat", "-M", b_ref]
            status_args = ["git", "--no-pager", "diff", "--name-status", "-M", b_ref]
        else:
            diff_range = f"{b_ref}..{t_ref}"
            diff_args = ["git", "--no-pager", "diff", "--unified=3", "-M", diff_range]
            stat_args = ["git", "--no-pager", "diff", "--stat", "-M", diff_range]
            status_args = ["git", "--no-pager", "diff", "--name-status", "-M", diff_range]

        diff_res = subprocess.run(
            diff_args,
            cwd=str(self.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_text = diff_res.stdout

        stat_res = subprocess.run(
            stat_args,
            cwd=str(self.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        diff_stat = stat_res.stdout.strip()

        status_res = subprocess.run(
            status_args,
            cwd=str(self.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        renamed_files: dict[str, str] = {}
        file_statuses: dict[str, str] = {}
        changed_files: list[str] = []

        for line in status_res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            code = parts[0]
            if code.startswith("R") and len(parts) >= 3:
                old_p, new_p = parts[1].strip(), parts[2].strip()
                renamed_files[new_p] = old_p
                file_statuses[new_p] = "RENAMED"
                changed_files.append(new_p)
            elif code.startswith("A") and len(parts) >= 2:
                p = parts[1].strip()
                file_statuses[p] = "ADDED"
                changed_files.append(p)
            elif code.startswith("D") and len(parts) >= 2:
                p = parts[1].strip()
                file_statuses[p] = "DELETED"
                changed_files.append(p)
            elif len(parts) >= 2:
                p = parts[1].strip()
                file_statuses[p] = "MODIFIED"
                changed_files.append(p)

        self.last_renamed_files = renamed_files
        self.last_file_statuses = file_statuses

        return diff_text, changed_files, diff_stat

    def analyze_diff(
        self,
        base_ref: str = "main",
        target_ref: Optional[str] = "HEAD",
    ) -> list[ChangedSymbol]:
        """Convenience method to resolve git diff and extract changed symbols."""
        diff_text, changed_files, _ = self.resolve_git_diff(base_ref=base_ref, target_ref=target_ref)
        return self.extract_changed_symbols(
            diff_text=diff_text,
            changed_files=changed_files,
            base_ref=base_ref,
            target_ref=target_ref,
            renamed_files=self.last_renamed_files,
            file_statuses=self.last_file_statuses,
        )

    def extract_changed_symbols(
        self,
        diff_text: str,
        changed_files: list[str],
        base_ref: Optional[str] = None,
        target_ref: Optional[str] = None,
        renamed_files: Optional[dict[str, str]] = None,
        file_statuses: Optional[dict[str, str]] = None,
    ) -> list[ChangedSymbol]:
        """
        Parses unified git diff output and compares base vs target AST function definitions
        to accurately classify changes into ADDED, DELETED, or MODIFIED.
        Preserves rename tracking and retrieves deleted file contents safely from Git storage.
        """
        symbols: list[ChangedSymbol] = []
        seen_keys: set[str] = set()

        renames = renamed_files if renamed_files is not None else getattr(self, "last_renamed_files", {})
        statuses = file_statuses if file_statuses is not None else getattr(self, "last_file_statuses", {})

        is_working_tree = target_ref is None or target_ref.strip().upper() in ("WORKING_TREE", "DIRTY", "UNCOMMITTED", "")
        t_ref = None if is_working_tree else target_ref.strip()
        b_ref = base_ref.strip() if base_ref else None

        # Parse diff hunks to identify modified line ranges per file
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        file_target_hunks: dict[str, list[tuple[int, int]]] = {}
        file_base_hunks: dict[str, list[tuple[int, int]]] = {}
        curr_b_file: Optional[str] = None
        curr_a_file: Optional[str] = None

        if diff_text:
            for line in diff_text.splitlines():
                if line.startswith("--- a/"):
                    curr_a_file = line[6:].strip()
                elif line.startswith("+++ b/"):
                    curr_b_file = line[6:].strip()
                    if curr_b_file and curr_b_file != "/dev/null":
                        file_target_hunks.setdefault(curr_b_file, [])
                    if curr_a_file and curr_a_file != "/dev/null":
                        file_base_hunks.setdefault(curr_a_file, [])
                elif line.startswith("@@"):
                    match = hunk_re.match(line)
                    if match:
                        base_start = int(match.group(1))
                        base_count = int(match.group(2)) if match.group(2) else 1
                        target_start = int(match.group(3))
                        target_count = int(match.group(4)) if match.group(4) else 1
                        if curr_b_file and curr_b_file != "/dev/null":
                            file_target_hunks.setdefault(curr_b_file, []).append((target_start, target_start + target_count))
                        if curr_a_file and curr_a_file != "/dev/null":
                            file_base_hunks.setdefault(curr_a_file, []).append((base_start, base_start + base_count))

        for file_path in changed_files:
            if not file_path.endswith(".py"):
                continue

            status = statuses.get(file_path, "MODIFIED")
            old_path = renames.get(file_path)

            # Retrieve base source
            base_source: Optional[str] = None
            if b_ref:
                base_query_path = old_path if old_path else file_path
                base_source = self._get_git_file_content(b_ref, base_query_path)

            # Retrieve target source
            target_source: Optional[str] = None
            if status != "DELETED":
                target_source = self._get_target_file_content(t_ref, file_path)

            # Case 1: Deleted file
            if status == "DELETED" or (b_ref is not None and target_source is None and base_source is not None):
                if base_source:
                    try:
                        base_funcs = ASTDiffParser.parse_source(base_source, file_path=file_path)
                        for f in base_funcs:
                            key = f"{file_path}:{f.class_name}.{f.name}" if f.class_name else f"{file_path}:{f.name}"
                            if key not in seen_keys:
                                seen_keys.add(key)
                                symbols.append(
                                    ChangedSymbol(
                                        name=f.name,
                                        class_name=f.class_name,
                                        file_path=file_path,
                                        line_start=f.line_start,
                                        line_end=f.line_end,
                                        change_type=ChangeType.DELETED,
                                        parameters=[p.name for p in f.parameters],
                                        raw_signature=f"def {f.name}({', '.join(p.name for p in f.parameters)}) -> {f.return_type or 'None'}",
                                    )
                                )
                    except Exception:
                        pass
                continue

            # Case 2: Added file
            if status == "ADDED" or (b_ref is not None and base_source is None and target_source is not None and not old_path):
                if target_source:
                    try:
                        target_funcs = ASTDiffParser.parse_source(target_source, file_path=file_path)
                        for f in target_funcs:
                            key = f"{file_path}:{f.class_name}.{f.name}" if f.class_name else f"{file_path}:{f.name}"
                            if key not in seen_keys:
                                seen_keys.add(key)
                                symbols.append(
                                    ChangedSymbol(
                                        name=f.name,
                                        class_name=f.class_name,
                                        file_path=file_path,
                                        line_start=f.line_start,
                                        line_end=f.line_end,
                                        change_type=ChangeType.ADDED,
                                        parameters=[p.name for p in f.parameters],
                                        raw_signature=f"def {f.name}({', '.join(p.name for p in f.parameters)}) -> {f.return_type or 'None'}",
                                    )
                                )
                    except Exception:
                        pass
                continue

            # Case 3: Both base and target sources exist (Modified or Renamed)
            if base_source is not None and target_source is not None:
                try:
                    base_funcs = {
                        (f.class_name, f.name): f
                        for f in ASTDiffParser.parse_source(base_source, file_path=old_path or file_path)
                    }
                    target_funcs = {
                        (f.class_name, f.name): f
                        for f in ASTDiffParser.parse_source(target_source, file_path=file_path)
                    }

                    target_hunks = file_target_hunks.get(file_path, [])

                    # Find Added and Modified symbols
                    for (c_name, s_name), tf in target_funcs.items():
                        key = f"{file_path}:{c_name}.{s_name}" if c_name else f"{file_path}:{s_name}"
                        if (c_name, s_name) not in base_funcs:
                            # Symbol is newly introduced in target
                            if key not in seen_keys:
                                seen_keys.add(key)
                                symbols.append(
                                    ChangedSymbol(
                                        name=tf.name,
                                        class_name=tf.class_name,
                                        file_path=file_path,
                                        line_start=tf.line_start,
                                        line_end=tf.line_end,
                                        change_type=ChangeType.ADDED,
                                        previous_file_path=old_path,
                                        parameters=[p.name for p in tf.parameters],
                                        raw_signature=f"def {tf.name}({', '.join(p.name for p in tf.parameters)}) -> {tf.return_type or 'None'}",
                                    )
                                )
                        else:
                            # Symbol exists in both revisions - verify if it actually changed
                            bf = base_funcs[(c_name, s_name)]
                            source_differs = bf.raw_source.strip() != tf.raw_source.strip()
                            overlaps_hunk = any(
                                not (tf.line_end < h_start or tf.line_start > h_end)
                                for h_start, h_end in target_hunks
                            ) if target_hunks else False

                            if source_differs or overlaps_hunk:
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    symbols.append(
                                        ChangedSymbol(
                                            name=tf.name,
                                            class_name=tf.class_name,
                                            file_path=file_path,
                                            line_start=tf.line_start,
                                            line_end=tf.line_end,
                                            change_type=ChangeType.MODIFIED,
                                            previous_file_path=old_path,
                                            parameters=[p.name for p in tf.parameters],
                                            raw_signature=f"def {tf.name}({', '.join(p.name for p in tf.parameters)}) -> {tf.return_type or 'None'}",
                                        )
                                    )

                    # Find Deleted symbols (present in base but missing in target)
                    for (c_name, s_name), bf in base_funcs.items():
                        key = f"{file_path}:{c_name}.{s_name}" if c_name else f"{file_path}:{s_name}"
                        if (c_name, s_name) not in target_funcs:
                            if key not in seen_keys:
                                seen_keys.add(key)
                                symbols.append(
                                    ChangedSymbol(
                                        name=bf.name,
                                        class_name=bf.class_name,
                                        file_path=old_path or file_path,
                                        line_start=bf.line_start,
                                        line_end=bf.line_end,
                                        change_type=ChangeType.DELETED,
                                        previous_file_path=old_path,
                                        parameters=[p.name for p in bf.parameters],
                                        raw_signature=f"def {bf.name}({', '.join(p.name for p in bf.parameters)}) -> {bf.return_type or 'None'}",
                                    )
                                )
                    continue
                except Exception:
                    pass

            # Fallback when git base/target sources are not available (e.g. pure synthetic diff unit test)
            if diff_text:
                diff_analysis = ASTDiffParser.parse_diff(diff_text, repo_root=str(self.repo_root))
                for f in diff_analysis.modified_functions:
                    rel_file = f.file_path
                    try:
                        rel_file = str(Path(f.file_path).resolve().relative_to(self.repo_root.resolve()))
                    except Exception:
                        pass
                    key = f"{rel_file}:{f.class_name}.{f.name}" if f.class_name else f"{rel_file}:{f.name}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        symbols.append(
                            ChangedSymbol(
                                name=f.name,
                                class_name=f.class_name,
                                file_path=rel_file,
                                line_start=f.line_start,
                                line_end=f.line_end,
                                change_type=ChangeType.MODIFIED,
                                previous_file_path=old_path,
                                parameters=[p.name for p in f.parameters],
                                raw_signature=f"def {f.name}({', '.join(p.name for p in f.parameters)}) -> {f.return_type or 'None'}",
                            )
                        )

        return symbols

    def build_transitive_impact_graph(
        self,
        changed_symbols: list[ChangedSymbol],
        max_depth: int = 3,
    ) -> tuple[list[ImpactNode], list[ImpactNode]]:
        """
        Executes breadth-first search across the codebase using Sourcegraph / AST Fallback
        to discover direct (depth 1) and indirect (depth >= 2) affected callers.
        Filters out tests (which are handled in test prioritization).
        """
        direct_impacts: list[ImpactNode] = []
        indirect_impacts: list[ImpactNode] = []

        visited_nodes: set[tuple[str, str]] = set()  # (symbol_qualified_name, file_path)
        # Queue stores: (current_symbol_name, current_file, current_class_name, current_depth, root_symbol_name, root_symbol_file, call_chain)
        queue: list[tuple[str, str, Optional[str], int, str, str, list[str]]] = []

        for cs in changed_symbols:
            # Skip test files from being considered root symbols for production caller graph
            if any(t_dir in cs.file_path for t_dir in ("tests/", "test_", "/tests", "_test.py")):
                continue
            visited_nodes.add((cs.qualified_name, cs.file_path))
            visited_nodes.add((cs.name, cs.file_path))
            if cs.class_name:
                visited_nodes.add((f"{cs.class_name}.{cs.name}", cs.file_path))
            queue.append((cs.name, cs.file_path, cs.class_name, 1, cs.qualified_name, cs.file_path, [cs.qualified_name]))

            # Discover static event registrations for changed symbol
            regs = self.event_detector.find_registrations_for_handler(cs.name)
            for reg in regs:
                node_key = (f"event:{reg.event_name}", reg.source_file)
                if node_key in visited_nodes:
                    continue
                visited_nodes.add(node_key)

                match_quality = "CONFIRMED_EVENT_REGISTRATION" if reg.is_confirmed else "HEURISTIC_EVENT_REGISTRATION"
                unc_reason = (
                    f"Static event registration in {reg.source_file}:{reg.line_number}"
                    if reg.is_confirmed
                    else "Dynamic event registration with unresolved handler argument"
                )
                lims = [] if reg.is_confirmed else ["Dynamic or unresolved event registration"]

                event_node = ImpactNode(
                    symbol_name=f"event:{reg.event_name}",
                    file_path=reg.source_file,
                    line_number=reg.line_number,
                    depth=1,
                    impact_type=ImpactType.EVENT_REGISTRATION,
                    root_changed_symbol=cs.qualified_name,
                    root_changed_file=cs.file_path,
                    evidence=EvidenceTrail(
                        call_chain=[cs.qualified_name, f"event:{reg.event_name}"],
                        caller_file=reg.source_file,
                        line_number=reg.line_number,
                        resolution_engine="event_registration_detector",
                        match_quality=match_quality,
                        evidence_source=f"Event Registration ({reg.registration_type})",
                        call_path_description=f"'{cs.qualified_name}' registered to '{reg.event_name}' via {reg.registration_expr}",
                        is_ambiguous=not reg.is_confirmed,
                    ),
                    uncertainty_score=reg.uncertainty_score,
                    confidence=reg.confidence,
                    uncertainty_reason=unc_reason,
                    is_confirmed=reg.is_confirmed,
                    limitations=lims,
                    reason=f"Event registration: '{cs.qualified_name}' is registered to listen to event '{reg.event_name}' in {reg.source_file}:{reg.line_number}",
                )
                direct_impacts.append(event_node)

                # Also enqueue event name to discover callers/triggers of the event
                if max_depth >= 2:
                    queue.append((reg.event_name, reg.source_file, None, 2, cs.qualified_name, cs.file_path, [cs.qualified_name, f"event:{reg.event_name}"]))

        while queue:
            curr_sym, curr_file, curr_class, depth, root_sym, root_file, chain = queue.pop(0)
            if depth > max_depth:
                continue

            # Query callers through Sourcegraph / AST client with class context
            raw_callers = self.sg_client.get_function_callers(curr_sym, curr_file, class_name=curr_class)

            for caller in raw_callers:
                c_name = caller.get("caller_name", "<module>")
                c_file = caller.get("file_path", "")
                c_line = caller.get("line_number")
                c_class = caller.get("class_name")
                engine_type = caller.get("source_type", "local_ast_fallback")
                is_ambiguous = caller.get("is_ambiguous", False)
                match_quality = caller.get(
                    "match_quality",
                    "EXACT_AST_CALL" if engine_type == "local_ast_fallback" else "SOURCEGRAPH_REF",
                )

                # Skip module-level execution wrappers or tests in this impact graph
                # (tests are evaluated separately in prioritize_tests)
                if any(t_dir in c_file for t_dir in ("tests/", "test_", "/tests", "_test.py")):
                    continue

                if c_name in ("<module>", "<module_level>"):
                    continue

                node_key = (c_name, c_file)
                c_qual = f"{c_class}.{c_name}" if c_class else c_name
                qual_key = (c_qual, c_file)

                # Deduplication and cycle prevention:
                if node_key in visited_nodes or qual_key in visited_nodes:
                    continue

                # Prevent self-references / self-loops:
                if (c_name == root_sym or c_qual == root_sym) and c_file == root_file:
                    continue

                visited_nodes.add(node_key)
                visited_nodes.add(qual_key)

                # Compute evidence and uncertainty
                new_chain = [*chain, c_name]
                evidence_source = "Sourcegraph OSS" if engine_type == "sourcegraph_graphql" else "Local AST Call Graph"
                call_path_desc = " -> ".join(new_chain)

                evidence = EvidenceTrail(
                    call_chain=new_chain,
                    caller_file=c_file,
                    line_number=c_line,
                    resolution_engine=engine_type,
                    match_quality=match_quality,
                    evidence_source=evidence_source,
                    call_path_description=call_path_desc,
                    is_ambiguous=is_ambiguous,
                )

                uncertainty = self._compute_uncertainty_score(
                    depth=depth,
                    symbol_name=c_name,
                    engine=engine_type,
                    is_ambiguous=is_ambiguous,
                )
                unc_reason = self._compute_uncertainty_reason(
                    depth=depth,
                    symbol_name=c_name,
                    engine=engine_type,
                    is_ambiguous=is_ambiguous,
                )
                confidence = round(1.0 - uncertainty, 2)
                is_confirmed = (not is_ambiguous) and (uncertainty < 0.40)

                limitations: list[str] = []
                if is_ambiguous:
                    limitations.append("Unresolved dynamic dispatch or unanchored receiver object")
                if depth >= 3:
                    limitations.append(f"High transitive graph distance ({depth} hops) may degrade reachability")

                impact_type = ImpactType.DIRECT if depth == 1 else ImpactType.INDIRECT
                reason = (
                    f"Directly calls changed symbol '{root_sym}'"
                    if depth == 1
                    else f"Transitively depends on '{root_sym}' via {len(new_chain)-1} hops ({call_path_desc})"
                )

                impact_node = ImpactNode(
                    symbol_name=c_name,
                    file_path=c_file,
                    line_number=c_line,
                    depth=depth,
                    impact_type=impact_type,
                    root_changed_symbol=root_sym,
                    root_changed_file=root_file,
                    evidence=evidence,
                    uncertainty_score=uncertainty,
                    confidence=confidence,
                    uncertainty_reason=unc_reason,
                    is_confirmed=is_confirmed,
                    limitations=limitations,
                    reason=reason,
                )

                if depth == 1:
                    direct_impacts.append(impact_node)
                else:
                    indirect_impacts.append(impact_node)

                # Enqueue for next hop if depth < max_depth
                if depth < max_depth and c_name not in ("<module>", "<module_level>"):
                    queue.append((c_name, c_file, c_class, depth + 1, root_sym, root_file, new_chain))

        return direct_impacts, indirect_impacts

    GENERIC_VERBS = {
        "__init__", "setUp", "tearDown", "setUpClass", "tearDownClass", "setUpTestData",
        "setup_method", "teardown_method", "setup_class", "teardown_class",
        "get", "post", "put", "delete", "patch", "options", "head",
        "save", "load", "close", "open", "read", "write", "run", "start", "stop", "reset", "clear",
        "update", "create", "filter", "all", "exists", "count",
        "assertTrue", "assertFalse", "assertEqual", "assertNotEqual", "assertRaises", "assertIn",
        "render", "dispatch", "execute", "validate", "status", "send", "connect", "handle", "process",
    }
    NON_TEST_METHODS = {
        "setUp", "tearDown", "setUpClass", "tearDownClass", "setUpTestData",
        "setup_method", "teardown_method", "setup_class", "teardown_class",
        "setUpModule", "tearDownModule",
    }
    COMMAND_DISPATCH_APIS = {
        "call_command",
        "execute_command",
        "run_command",
        "handle_command",
        "call_cmd",
    }

    def _discover_test_files(self) -> list[Path]:
        """
        Discovers unique test files across tests/, test/, and repository subpackages.
        Filters out virtualenvs, git directories, caches, and build artifacts.
        """
        discovered: set[Path] = set()
        ignored = {
            ".git", ".venv", "venv", "env", "node_modules",
            "build", "dist", ".tox", ".eggs", "__pycache__", "site-packages"
        }

        # 1. Search tests/ and test/ directories recursively
        for dname in ("tests", "test"):
            td = self.repo_root / dname
            if td.exists() and td.is_dir():
                for p in td.rglob("*.py"):
                    if not any(part in ignored for part in p.parts):
                        if p.name.startswith("test_") or p.name.endswith("_test.py") or "/tests/" in str(p) or "/test/" in str(p):
                            discovered.add(p)

        # 2. Also search top-level test files
        for pattern in ("test_*.py", "*_test.py"):
            for p in self.repo_root.glob(pattern):
                if not any(part in ignored for part in p.parts):
                    discovered.add(p)

        return sorted(discovered)

    @staticmethod
    def _extract_call_names(node: ast.AST) -> set[str]:
        """Extracts direct function and method call names from an AST block."""
        calls: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        return calls

    @classmethod
    def _extract_command_dispatch_records(cls, node: ast.AST) -> list[CommandDispatchRecord]:
        """
        Extract detailed command dispatch records from recognized command-dispatch APIs.
        Identifies:
        - call_command("migrate", "auth_tests", ...)
        - call_command(command_name="migrate")
        - self.call_command("migrate")
        - MigrationExecutor(...)
        - executor.migrate(...)
        Only extracts commands from recognized dispatch functions or methods,
        avoiding arbitrary string literal matching.
        """
        records: list[CommandDispatchRecord] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                is_dispatch = False
                dispatch_api = ""
                cmd_name = None
                extra_args: list[str] = []
                line_no = getattr(child, "lineno", 0)

                if isinstance(child.func, ast.Name):
                    if child.func.id in cls.COMMAND_DISPATCH_APIS:
                        is_dispatch = True
                        dispatch_api = child.func.id
                    elif child.func.id == "MigrationExecutor":
                        records.append(
                            CommandDispatchRecord(
                                command_name="MigrationExecutor",
                                dispatch_api="MigrationExecutor",
                                call_expr="MigrationExecutor(...)",
                                extra_args=[],
                                line_number=line_no,
                            )
                        )
                elif isinstance(child.func, ast.Attribute):
                    attr_name = child.func.attr
                    rec_name = EventRegistrationDetector._unparse_expr(child.func.value)
                    if attr_name in cls.COMMAND_DISPATCH_APIS:
                        is_dispatch = True
                        dispatch_api = f"{rec_name}.{attr_name}" if rec_name else attr_name
                    elif attr_name == "MigrationExecutor":
                        records.append(
                            CommandDispatchRecord(
                                command_name="MigrationExecutor",
                                dispatch_api="MigrationExecutor",
                                call_expr=f"{rec_name}.MigrationExecutor(...)" if rec_name else "MigrationExecutor(...)",
                                extra_args=[],
                                line_number=line_no,
                            )
                        )
                    elif attr_name == "migrate" and any(w in rec_name.lower() for w in ("executor", "migrationexecutor")):
                        records.append(
                            CommandDispatchRecord(
                                command_name="migrate",
                                dispatch_api=f"{rec_name}.migrate",
                                call_expr=f"{rec_name}.migrate(...)",
                                extra_args=[],
                                line_number=line_no,
                            )
                        )

                if is_dispatch:
                    if child.args:
                        for i, arg in enumerate(child.args):
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                if i == 0 and not cmd_name:
                                    cmd_name = arg.value
                                else:
                                    extra_args.append(arg.value)
                    kw_matched = False
                    for kw in child.keywords:
                        if kw.arg in {"command_name", "command", "name", "cmd"} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            if not cmd_name:
                                cmd_name = kw.value.value
                                kw_matched = True
                        elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            extra_args.append(kw.value.value)

                    if cmd_name:
                        api_label = f"{dispatch_api}(keyword)" if kw_matched else dispatch_api
                        records.append(
                            CommandDispatchRecord(
                                command_name=cmd_name,
                                dispatch_api=api_label,
                                call_expr=f"{dispatch_api}('{cmd_name}')",
                                extra_args=extra_args,
                                line_number=line_no,
                            )
                        )
        return records

    @classmethod
    def _extract_command_dispatch_args(cls, node: ast.AST) -> set[str]:
        """Backward-compatible extraction of command names."""
        return {r.command_name for r in cls._extract_command_dispatch_records(node)}

    @classmethod
    def _extract_text_tokens(cls, text: str) -> set[str]:
        """Extract individual word tokens from camelCase, snake_case, paths, and identifiers."""
        if not text:
            return set()
        s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        parts = re.split(r"[^a-zA-Z0-9]+", s1.lower())
        tokens: set[str] = set()
        for p in parts:
            if p and len(p) > 1:
                tokens.add(p)
                tokens.add(p.rstrip("s"))
        return tokens

    @classmethod
    def _matches_any_token(cls, text: str, target_tokens: set[str]) -> bool:
        """Token-boundary match: checks if any target token matches a discrete token in text."""
        if not text or not target_tokens:
            return False
        tokens = cls._extract_text_tokens(text)
        return any(tok in tokens or tok.rstrip("s") in tokens for tok in target_tokens)

    @classmethod
    def _analyze_behavioral_evidence(
        cls,
        node: ast.AST,
        test_identifier: str,
        rel_test_path: str,
        ev_node: ImpactNode,
        changed_symbols: list[ChangedSymbol],
        changed_files: list[str],
        imported_symbols: dict[str, str],
        direct_called: set[str],
        attribute_exprs: set[str],
        command_records: list[CommandDispatchRecord],
        lineno: int,
    ) -> tuple[list[EvidenceItem], list[str]]:
        """
        Statically evaluates whether a test that touches event infrastructure possesses
        concrete behavioral coverage connecting it to the changed handler.
        Checks:
        1. Domain references & assertions (queries/asserts on domain models or state managed by the handler).
        2. Specific command arguments (e.g. call_command('migrate', 'auth_tests') targeting the specific domain).
        3. Diff co-change proximity (test file was modified/added in the diff alongside the handler).
        4. Direct or transitive call-graph reachability to the changed handler or its domain helpers.
        """
        evidence_items: list[EvidenceItem] = []
        handler_name = ev_node.root_changed_symbol
        handler_file = ev_node.root_changed_file or ""
        event_name = ev_node.symbol_name.replace("event:", "")
        event_tokens = {t.lower() for t in event_name.split("_") if len(t) > 2}
        if event_name in EventRegistrationDetector.KNOWN_TRIGGER_MAP:
            for hint in EventRegistrationDetector.KNOWN_TRIGGER_MAP[event_name]:
                event_tokens.update(t.lower() for t in hint.split("_") if len(t) > 2)

        # 1. Extract domain tokens from handler name and file path (excluding generic & event tokens)
        raw_tokens = [t.lower() for t in handler_name.split("_") if len(t) > 2]
        generic_tokens = {
            "after", "before", "with", "when", "handler", "receiver",
            "callback", "handle", "action", "signal", "send", "connect",
            "disconnect", "call", "event", "model", "models", "migration",
            "migrations", "app", "apps", "field", "fields", "state",
            "states", "operation", "operations", "executor",
            "rename", "renamed", "renaming", "change", "changed", "changing",
            "create", "created", "creating", "update", "updated", "updating",
            "delete", "deleted", "deleting", "remove", "removed", "removing",
            "get", "set", "check", "test", "run", "make", "mutate", "print",
            "django", "contrib", "core", "base", "common", "default", "objects",
        }
        domain_tokens = {
            t.rstrip("s") for t in raw_tokens
            if t not in generic_tokens and t not in event_tokens and t.rstrip("s") not in event_tokens
        }

        file_tokens = set()
        if handler_file:
            p_parts = Path(handler_file).parts
            for part in p_parts:
                stem = part.replace(".py", "").lower()
                if (
                    stem not in {"__init__", "management", "commands", "utils", "core", "models", "views", "migrations", "django", "contrib"}
                    and stem not in event_tokens
                    and stem not in generic_tokens
                ):
                    file_tokens.add(stem.rstrip("s"))

        # 2. Check Domain Model References & Behavioral Assertions
        # A query is not an assertion. Require evidence that the test actually asserts or
        # checks relevant domain behavior (e.g. permission codename/name state, expected conflict, or rename output).
        domain_behavior_exprs = set()
        relevant_fields = {"codename", "name", "content_type"}

        # Track local variables assigned from domain model queries specifying relevant fields
        domain_assigned_vars: dict[str, str] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                val_text = ast.unparse(child.value)
                val_tokens = cls._extract_text_tokens(val_text)
                if any(dt in val_tokens for dt in domain_tokens):
                    kw_tokens = set()
                    if isinstance(child.value, ast.Call):
                        for kw in child.value.keywords:
                            if kw.arg:
                                kw_tokens.update(cls._extract_text_tokens(kw.arg))
                    if any(rf in kw_tokens or rf in val_tokens for rf in relevant_fields):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                domain_assigned_vars[target.id] = val_text

        # Collect assertion nodes (assert statement, self.assert* call, or with self.assertRaises/assertWarns)
        assert_nodes: list[ast.AST] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                assert_nodes.append(child)
            elif isinstance(child, ast.Call):
                func_name = ast.unparse(child.func)
                if "assert" in func_name.lower():
                    assert_nodes.append(child)
            elif isinstance(child, ast.With):
                for item in child.items:
                    ctx_expr = ast.unparse(item.context_expr)
                    if "assert" in ctx_expr.lower():
                        assert_nodes.append(item.context_expr)

        # Inspect assertion nodes for behavior-specific checks
        for a_node in assert_nodes:
            for sub in ast.walk(a_node):
                # Check assertion string constants for rename/conflict messages
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    s_tokens = cls._extract_text_tokens(sub.value)
                    if any(dt in s_tokens for dt in domain_tokens) and any(
                        v in s_tokens for v in ("rename", "renamed", "renaming", "conflict", "fail", "failed")
                    ):
                        prefix = sub.value[:45]
                        domain_behavior_exprs.add(f"assertion_message(\"{prefix}...\")")

                # Check domain model queries directly inside assertions with relevant fields
                elif isinstance(sub, ast.Call):
                    call_text = ast.unparse(sub.func)
                    call_tokens = cls._extract_text_tokens(call_text)
                    if any(dt in call_tokens for dt in domain_tokens):
                        kw_tokens = set()
                        for kw in sub.keywords:
                            if kw.arg:
                                kw_tokens.update(cls._extract_text_tokens(kw.arg))
                        if any(rf in kw_tokens for rf in relevant_fields):
                            kw_summary = ", ".join(kw.arg for kw in sub.keywords if kw.arg)
                            domain_behavior_exprs.add(f"{call_text}({kw_summary})")

                # Check attribute accesses on domain entities inside assertions (e.g. permission.codename)
                elif isinstance(sub, ast.Attribute):
                    if sub.attr in ("codename", "name"):
                        val_text = ast.unparse(sub.value)
                        val_tokens = cls._extract_text_tokens(val_text)
                        if any(dt in val_tokens for dt in domain_tokens) or "perm" in val_text.lower():
                            domain_behavior_exprs.add(f"{val_text}.{sub.attr}")

                # Check assertion evaluating a domain variable assigned from a relevant query
                elif isinstance(sub, ast.Name) and sub.id in domain_assigned_vars:
                    domain_behavior_exprs.add(f"{domain_assigned_vars[sub.id]} -> assert {sub.id}")

        if domain_behavior_exprs:
            sample_expr = sorted(domain_behavior_exprs)[0]
            evidence_items.append(
                EvidenceItem(
                    type="BEHAVIORAL_ASSERTION",
                    description=f"Asserts on domain behavior '{sample_expr}' managed by '{handler_name}'.",
                    source_file=rel_test_path,
                    line=lineno,
                )
            )

        # 3. Check Targeted Command Arguments with Token-boundary matching
        for rec in command_records:
            if rec.extra_args:
                matched_args = [
                    arg for arg in rec.extra_args
                    if cls._matches_any_token(arg, domain_tokens) or cls._matches_any_token(arg, file_tokens)
                ]
                if matched_args:
                    arg_summary = ", ".join(matched_args)
                    evidence_items.append(
                        EvidenceItem(
                            type="COMMAND_DISPATCH",
                            description=f"Invokes {rec.dispatch_api} with targeted arguments ({arg_summary}) specific to '{handler_name}'.",
                            source_file=rel_test_path,
                            line=rec.line_number or lineno,
                        )
                    )
                    break

        # 4. Check Diff Co-Change Proximity
        if rel_test_path in changed_files:
            evidence_items.append(
                EvidenceItem(
                    type="DIFF_CO_CHANGE",
                    description=f"Test file '{rel_test_path}' was modified or added in the diff alongside '{handler_name}'.",
                    source_file=rel_test_path,
                    line=lineno,
                )
            )

        # 5. Check Direct Call / Reachability to handler
        if handler_name in direct_called:
            evidence_items.append(
                EvidenceItem(
                    type="CALL_GRAPH_REACHABILITY",
                    description=f"Directly invokes changed handler '{handler_name}'.",
                    source_file=rel_test_path,
                    line=lineno,
                )
            )

        has_behavioral_coverage = any(
            item.type in ("BEHAVIORAL_ASSERTION", "CALL_GRAPH_REACHABILITY") for item in evidence_items
        )
        if has_behavioral_coverage:
            missing_evidence: list[str] = []
        else:
            missing_evidence = [
                "No behavior-specific assertion or changed-symbol relationship was established.",
                f"The test may trigger '{event_name}' through migration execution, but behavior-specific coverage of '{handler_name}' is unconfirmed.",
            ]

        return evidence_items, missing_evidence

    @classmethod
    def _is_valid_direct_match(
        cls,
        s_name: str,
        changed_sym_map: dict[str, ChangedSymbol],
        imported_symbols: Any,
        qualified_receivers: set[str],
    ) -> bool:
        """Guards against false positives from generic verbs and assertions."""
        if s_name not in cls.GENERIC_VERBS and s_name not in LocalCodeGraphFallback.GENERIC_METHOD_NAMES:
            return True
        cs = changed_sym_map.get(s_name)
        if not cs:
            return False
        if s_name in imported_symbols:
            return True
        if cs.class_name and cs.class_name in imported_symbols:
            return True
        target_mod = Path(cs.file_path).stem
        if target_mod in qualified_receivers or (cs.class_name and cs.class_name in qualified_receivers):
            return True
        return False

    @classmethod
    def _extract_local_var_types(cls, node: ast.AST) -> dict[str, str]:
        """
        Extracts local variable to class name bindings from assignments in an AST block.
        E.g.: entity = HueButtonEventEntity(...) -> var_types['entity'] = 'HueButtonEventEntity'
        """
        var_types: dict[str, str] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                target_names = []
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                if target_names and isinstance(child.value, ast.Call):
                    if isinstance(child.value.func, ast.Name):
                        cls_name = child.value.func.id
                        for t_name in target_names:
                            var_types[t_name] = cls_name
                    elif isinstance(child.value.func, ast.Attribute):
                        cls_name = child.value.func.attr
                        for t_name in target_names:
                            var_types[t_name] = cls_name
            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name) and isinstance(child.value, ast.Call):
                    if isinstance(child.value.func, ast.Name):
                        var_types[child.target.id] = child.value.func.id
                    elif isinstance(child.value.func, ast.Attribute):
                        var_types[child.target.id] = child.value.func.attr
        return var_types

    @classmethod
    def _evaluate_call_match(
        cls,
        call_node: ast.Call,
        cs: ChangedSymbol,
        imported_symbols: dict[str, dict[str, str]],
        rel_test_path: str,
        enclosing_class: Optional[str] = None,
        local_var_types: Optional[dict[str, str]] = None,
    ) -> Optional[tuple[str, str, float, float, bool]]:
        """
        Evaluates whether an AST Call matches the qualified identity of ChangedSymbol.
        Returns: (match_quality, description, confidence, uncertainty_score, is_ambiguous) or None.
        """
        var_types = local_var_types or {}
        norm_target_file = cs.file_path.replace("\\", "/").lstrip("./")
        norm_test_file = rel_test_path.replace("\\", "/").lstrip("./")
        is_same_file = bool(norm_target_file and (norm_test_file == norm_target_file or norm_test_file.endswith(norm_target_file)))

        # Case A: Constructor matching (cs.is_constructor, i.e. __init__ with class_name)
        if cs.is_constructor:
            target_class = cs.class_name
            # 1. Direct instantiation: TargetClass(...)
            if isinstance(call_node.func, ast.Name):
                called_id = call_node.func.id
                if called_id == target_class:
                    if called_id in imported_symbols:
                        mod = imported_symbols[called_id]["source_module"]
                        if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                            return (
                                "CONFIRMED_CONSTRUCTOR_CALL",
                                f"Directly instantiates changed class '{target_class}' ({cs.file_path})",
                                0.95,
                                0.05,
                                False,
                            )
                        else:
                            return None
                    elif is_same_file or LocalCodeGraphFallback._module_matches_path(norm_test_file, cs.file_path):
                        return (
                            "CONFIRMED_CONSTRUCTOR_CALL",
                            f"Instantiates changed class '{target_class}' in same module/file ({cs.file_path})",
                            0.95,
                            0.05,
                            False,
                        )
                    else:
                        # Bare class name in unimported, unrelated test
                        return None
                elif called_id in imported_symbols and imported_symbols[called_id].get("original_symbol") == target_class:
                    mod = imported_symbols[called_id]["source_module"]
                    if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                        return (
                            "CONFIRMED_CONSTRUCTOR_CALL",
                            f"Instantiates changed class '{target_class}' via alias '{called_id}' ({cs.file_path})",
                            0.95,
                            0.05,
                            False,
                        )

            # 2. Attribute instantiation or explicit method: module.TargetClass(...) or Class.__init__(...)
            elif isinstance(call_node.func, ast.Attribute):
                attr_name = call_node.func.attr
                # 2a. module.TargetClass(...)
                if attr_name == target_class:
                    rec_expr = EventRegistrationDetector._unparse_expr(call_node.func.value)
                    if rec_expr in imported_symbols:
                        mod = imported_symbols[rec_expr]["source_module"]
                        if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                            return (
                                "CONFIRMED_CONSTRUCTOR_CALL",
                                f"Instantiates '{target_class}' via module import '{rec_expr}' ({cs.file_path})",
                                0.95,
                                0.05,
                                False,
                            )
                    elif LocalCodeGraphFallback._module_matches_path(rec_expr, cs.file_path):
                        return (
                            "CONFIRMED_CONSTRUCTOR_CALL",
                            f"Instantiates '{target_class}' via module path '{rec_expr}' ({cs.file_path})",
                            0.95,
                            0.05,
                            False,
                        )
                    return None

                # 2b. Explicit __init__: TargetClass.__init__(self, ...) or instance.__init__()
                elif attr_name == "__init__":
                    rec_node = call_node.func.value
                    if isinstance(rec_node, ast.Name):
                        rec_id = rec_node.id
                        if rec_id == target_class:
                            if rec_id in imported_symbols:
                                mod = imported_symbols[rec_id]["source_module"]
                                if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                                    return (
                                        "CONFIRMED_CLASS_METHOD_CALL",
                                        f"Directly invokes '{target_class}.__init__' ({cs.file_path})",
                                        0.95,
                                        0.05,
                                        False,
                                    )
                            elif is_same_file:
                                return (
                                    "CONFIRMED_CLASS_METHOD_CALL",
                                    f"Directly invokes '{target_class}.__init__' in same file ({cs.file_path})",
                                    0.95,
                                    0.05,
                                    False,
                                )
                        elif rec_id in ("self", "cls"):
                            if enclosing_class == target_class:
                                return (
                                    "SAME_FILE_CLASS_CALL",
                                    f"Invokes '__init__' on enclosing class '{target_class}'",
                                    0.95,
                                    0.05,
                                    False,
                                )
                            else:
                                return None
                        elif rec_id in var_types and var_types[rec_id] == target_class:
                            return (
                                "CONFIRMED_CLASS_METHOD_CALL",
                                f"Invokes '__init__' on instance '{rec_id}' of '{target_class}'",
                                0.95,
                                0.05,
                                False,
                            )
                        else:
                            return None
                    elif isinstance(rec_node, ast.Call) and isinstance(rec_node.func, ast.Name) and rec_node.func.id == "super":
                        if enclosing_class == target_class:
                            return (
                                "SAME_FILE_CLASS_CALL",
                                f"Invokes 'super().__init__' within '{target_class}'",
                                0.90,
                                0.10,
                                False,
                            )
                        else:
                            return None
                    return None
            return None

        # Case B: Class method matching (cs.class_name and cs.name != "__init__")
        if cs.class_name:
            target_class = cs.class_name
            target_method = cs.name
            if isinstance(call_node.func, ast.Attribute):
                attr_name = call_node.func.attr
                if attr_name != target_method:
                    return None
                rec_node = call_node.func.value
                if isinstance(rec_node, ast.Name):
                    rec_id = rec_node.id
                    if rec_id == target_class:
                        if rec_id in imported_symbols:
                            mod = imported_symbols[rec_id]["source_module"]
                            if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                                return (
                                    "CONFIRMED_CLASS_METHOD_CALL",
                                    f"Calls method '{cs.qualified_name}' on class '{target_class}'",
                                    0.95,
                                    0.05,
                                    False,
                                )
                        elif is_same_file:
                            return (
                                "CONFIRMED_CLASS_METHOD_CALL",
                                f"Calls method '{cs.qualified_name}' on class '{target_class}' in same file",
                                0.95,
                                0.05,
                                False,
                            )
                    elif rec_id in ("self", "cls"):
                        if enclosing_class == target_class:
                            return (
                                "SAME_FILE_CLASS_CALL",
                                f"Calls method '{target_method}' on '{rec_id}' in '{target_class}'",
                                0.95,
                                0.05,
                                False,
                            )
                        else:
                            return None
                    elif rec_id in var_types and var_types[rec_id] == target_class:
                        return (
                            "CONFIRMED_METHOD_CALL",
                            f"Calls method '{target_method}' on instance '{rec_id}' of '{target_class}'",
                            0.95,
                            0.05,
                            False,
                        )
                    else:
                        if rec_id in ("app", "client", target_class.lower()) and (
                            target_class in imported_symbols
                            or any(m in imported_symbols for m in ("flask", "django", "testbed"))
                            or is_same_file
                        ):
                            return (
                                "CONFIRMED_METHOD_CALL",
                                f"Calls method '{target_method}' on fixture/instance '{rec_id}' of '{target_class}'",
                                0.95,
                                0.05,
                                False,
                            )
                        if target_method in cls.GENERIC_VERBS or target_method in LocalCodeGraphFallback.GENERIC_METHOD_NAMES:
                            return None
                        if target_class in imported_symbols:
                            return (
                                "AMBIGUOUS_SYMBOL_REFERENCE",
                                f"Calls method '{target_method}' on unanchored receiver '{rec_id}'",
                                0.60,
                                0.40,
                                True,
                            )
                        return None
                elif isinstance(rec_node, ast.Attribute):
                    rec_expr = EventRegistrationDetector._unparse_expr(rec_node)
                    if target_class in rec_expr:
                        return (
                            "CONFIRMED_CLASS_METHOD_CALL",
                            f"Calls method '{cs.qualified_name}' via '{rec_expr}'",
                            0.95,
                            0.05,
                            False,
                        )
                    return None
            elif isinstance(call_node.func, ast.Name):
                called_id = call_node.func.id
                if called_id in imported_symbols:
                    info = imported_symbols[called_id]
                    if info.get("original_symbol") == target_method:
                        if LocalCodeGraphFallback._module_matches_path(info["source_module"], cs.file_path):
                            return (
                                "CONFIRMED_IMPORT_CALL",
                                f"Directly calls imported method '{cs.qualified_name}'",
                                0.95,
                                0.05,
                                False,
                            )
                return None
            return None

        # Case C: Top-level function matching (not cs.class_name)
        target_fn = cs.name
        if isinstance(call_node.func, ast.Name):
            called_id = call_node.func.id
            if called_id in imported_symbols:
                info = imported_symbols[called_id]
                if info.get("original_symbol") == target_fn:
                    if LocalCodeGraphFallback._module_matches_path(info["source_module"], cs.file_path):
                        return (
                            "CONFIRMED_IMPORT_CALL",
                            f"Directly calls imported function '{target_fn}' ({cs.file_path})",
                            0.95,
                            0.05,
                            False,
                        )
                    else:
                        return None
            elif called_id == target_fn:
                if is_same_file or LocalCodeGraphFallback._module_matches_path(norm_test_file, cs.file_path):
                    return (
                        "SAME_FILE_AST_CALL",
                        f"Calls function '{target_fn}' in same module/file ({cs.file_path})",
                        0.95,
                        0.05,
                        False,
                    )
                elif target_fn in cls.GENERIC_VERBS or target_fn in LocalCodeGraphFallback.GENERIC_METHOD_NAMES:
                    return None
                else:
                    return (
                        "EXACT_AST_CALL",
                        f"Directly calls function '{target_fn}'",
                        0.90,
                        0.10,
                        False,
                    )
            return None

        elif isinstance(call_node.func, ast.Attribute):
            attr_name = call_node.func.attr
            if attr_name == target_fn:
                rec_expr = EventRegistrationDetector._unparse_expr(call_node.func.value)
                if rec_expr in imported_symbols:
                    mod = imported_symbols[rec_expr]["source_module"]
                    if LocalCodeGraphFallback._module_matches_path(mod, cs.file_path):
                        return (
                            "CONFIRMED_IMPORT_CALL",
                            f"Calls function '{target_fn}' via module import '{rec_expr}' ({cs.file_path})",
                            0.95,
                            0.05,
                            False,
                        )
                elif LocalCodeGraphFallback._module_matches_path(rec_expr, cs.file_path):
                    return (
                        "CONFIRMED_IMPORT_CALL",
                        f"Calls function '{target_fn}' via module path '{rec_expr}' ({cs.file_path})",
                        0.95,
                        0.05,
                        False,
                    )
                elif target_fn not in cls.GENERIC_VERBS and target_fn not in LocalCodeGraphFallback.GENERIC_METHOD_NAMES:
                    return (
                        "AMBIGUOUS_SYMBOL_REFERENCE",
                        f"Calls function '{target_fn}' on unanchored receiver '{rec_expr}'",
                        0.45,
                        0.55,
                        True,
                    )
            return None

        return None

    def prioritize_tests(
        self,
        changed_symbols: list[ChangedSymbol],
        direct_impacts: list[ImpactNode],
        indirect_impacts: list[ImpactNode],
        include_candidates: bool = True,
        changed_files: Optional[list[str]] = None,
    ) -> list[PrioritizedTest]:
        """
        Inspects existing tests across the repository and maps them against changed symbols,
        event registrations, and direct/indirect impacts to compute a deterministic prioritized ranking.
        Distinguishes explicit event dispatch, behavior-supported coverage, and trigger-only candidates.
        Uses qualified symbol identity and constructor/class resolution to eliminate false positives.
        """
        prioritized: list[PrioritizedTest] = []
        event_candidates: list[PrioritizedTest] = []
        seen_tests: set[tuple[str, str]] = set()  # (test_file, test_identifier)
        total_discovered = 0

        resolved_changed_files = (
            changed_files if changed_files is not None
            else list({cs.file_path for cs in changed_symbols})
        )

        def _is_test_path(fp: str) -> bool:
            return any(t_dir in fp for t_dir in ("tests/", "test_", "/tests", "_test.py"))

        # Filter out test symbols and setup fixtures from changed symbols
        valid_changed_symbols = [
            cs for cs in changed_symbols
            if cs.name not in self.NON_TEST_METHODS
            and not (_is_test_path(cs.file_path) and (cs.name.startswith("test_") or cs.name.endswith("_test")))
            and not _is_test_path(cs.file_path)
        ]
        direct_sym_map = {d.symbol_name: d for d in direct_impacts if d.impact_type == ImpactType.DIRECT}
        indirect_sym_map = {i.symbol_name: i for i in indirect_impacts if i.impact_type == ImpactType.INDIRECT}
        event_impacts = [d for d in direct_impacts if d.impact_type == ImpactType.EVENT_REGISTRATION]

        # Discover all test files in repo
        unique_test_files = self._discover_test_files()

        for test_path in unique_test_files:
            try:
                rel_test_path = str(test_path.relative_to(self.repo_root)).replace("\\", "/")
            except ValueError:
                rel_test_path = str(test_path).replace("\\", "/")

            try:
                source = test_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(test_path))
            except Exception:
                continue

            # Index imports in this test file to anchor qualified / generic symbol calls
            imported_symbols: dict[str, dict[str, str]] = {}
            for imp in ast.walk(tree):
                if isinstance(imp, ast.ImportFrom):
                    mod = imp.module or ""
                    for alias in imp.names:
                        imported_symbols[alias.asname or alias.name] = {
                            "source_module": mod,
                            "original_symbol": alias.name,
                        }
                elif isinstance(imp, ast.Import):
                    for alias in imp.names:
                        imported_symbols[alias.asname or alias.name] = {
                            "source_module": alias.name,
                            "original_symbol": alias.name.split(".")[-1],
                        }

            # Index helper functions and discover test items (functions and methods)
            local_helpers: dict[str, set[str]] = {}
            local_helpers_cmd_records: dict[str, list[CommandDispatchRecord]] = {}
            local_helpers_matched_symbols: dict[str, list[tuple[ChangedSymbol, str, float, float, bool]]] = {}
            test_items: list[tuple[str, Optional[str], ast.AST, int]] = []  # (func_name, class_name, node, lineno)

            for item in tree.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_") or item.name.endswith("_test"):
                        test_items.append((item.name, None, item, item.lineno))
                    elif item.name not in self.NON_TEST_METHODS:
                        local_helpers[item.name] = self._extract_call_names(item)
                        local_helpers_cmd_records[item.name] = self._extract_command_dispatch_records(item)
                        h_vars = self._extract_local_var_types(item)
                        h_matches = []
                        for child in ast.walk(item):
                            if isinstance(child, ast.Call):
                                for cs in valid_changed_symbols:
                                    m = self._evaluate_call_match(
                                        child, cs, imported_symbols, rel_test_path, None, h_vars
                                    )
                                    if m and not m[4]:
                                        h_matches.append((cs, m[0], m[2], m[3], m[4]))
                        local_helpers_matched_symbols[item.name] = h_matches
                elif isinstance(item, ast.ClassDef):
                    cls_name = item.name
                    for class_child in item.body:
                        if isinstance(class_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if class_child.name in self.NON_TEST_METHODS:
                                continue  # Skip setUp, tearDown, etc.
                            elif class_child.name.startswith("test_") or class_child.name.endswith("_test"):
                                test_items.append((class_child.name, cls_name, class_child, class_child.lineno))
                            else:
                                local_helpers[class_child.name] = self._extract_call_names(class_child)
                                local_helpers_cmd_records[class_child.name] = self._extract_command_dispatch_records(class_child)
                                h_vars = self._extract_local_var_types(class_child)
                                h_matches = []
                                for child in ast.walk(class_child):
                                    if isinstance(child, ast.Call):
                                        for cs in valid_changed_symbols:
                                            m = self._evaluate_call_match(
                                                child, cs, imported_symbols, rel_test_path, cls_name, h_vars
                                            )
                                            if m and not m[4]:
                                                h_matches.append((cs, m[0], m[2], m[3], m[4]))
                                local_helpers_matched_symbols[class_child.name] = h_matches

            # Compute transitive closure for local_helpers so nested helper invocations are fully expanded
            closure_changed = True
            while closure_changed:
                closure_changed = False
                for h_name, targets in list(local_helpers.items()):
                    expanded = set(targets)
                    cmd_recs = list(local_helpers_cmd_records.get(h_name, []))
                    matched_syms = list(local_helpers_matched_symbols.get(h_name, []))
                    seen_cmds = {(r.command_name, r.dispatch_api) for r in cmd_recs}
                    seen_matched = {m[0].canonical_id for m in matched_syms}
                    for t in targets:
                        if t in local_helpers:
                            expanded.update(local_helpers[t])
                        if t in local_helpers_cmd_records:
                            for r in local_helpers_cmd_records[t]:
                                if (r.command_name, r.dispatch_api) not in seen_cmds:
                                    cmd_recs.append(r)
                                    seen_cmds.add((r.command_name, r.dispatch_api))
                        if t in local_helpers_matched_symbols:
                            for m in local_helpers_matched_symbols[t]:
                                if m[0].canonical_id not in seen_matched:
                                    matched_syms.append(m)
                                    seen_matched.add(m[0].canonical_id)
                    if (
                        len(expanded) > len(targets)
                        or len(cmd_recs) > len(local_helpers_cmd_records.get(h_name, []))
                        or len(matched_syms) > len(local_helpers_matched_symbols.get(h_name, []))
                    ):
                        local_helpers[h_name] = expanded
                        local_helpers_cmd_records[h_name] = cmd_recs
                        local_helpers_matched_symbols[h_name] = matched_syms
                        closure_changed = True

            for func_name, class_name, node, lineno in test_items:
                total_discovered += 1
                test_identifier = f"{class_name}::{func_name}" if class_name else func_name
                test_key = (rel_test_path, test_identifier)
                if test_key in seen_tests:
                    continue

                local_var_types = self._extract_local_var_types(node)

                # Extract all call targets and expressions inside this test
                direct_called: set[str] = set()
                attribute_calls: set[str] = set()
                attribute_exprs: set[str] = set()
                helper_calls: set[str] = set()
                qualified_receivers: set[str] = set()
                command_records = self._extract_command_dispatch_records(node)

                direct_matches: list[tuple[ChangedSymbol, str, str, float, float, bool]] = []

                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            direct_called.add(child.func.id)
                            if child.func.id in local_helpers:
                                helper_calls.add(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            attr_name = child.func.attr
                            direct_called.add(attr_name)
                            attribute_calls.add(attr_name)
                            rec_name = EventRegistrationDetector._unparse_expr(child.func.value)
                            if rec_name:
                                qualified_receivers.add(rec_name)
                                attribute_exprs.add(f"{rec_name}.{attr_name}")
                            if rec_name in ("self", "cls") and attr_name in local_helpers:
                                helper_calls.add(attr_name)

                        for cs in valid_changed_symbols:
                            m = self._evaluate_call_match(
                                child, cs, imported_symbols, rel_test_path, class_name, local_var_types
                            )
                            if m:
                                direct_matches.append((cs, m[0], m[1], m[2], m[3], m[4]))

                # Expand called_symbols and command_records with targets called by invoked helpers
                called_symbols: set[str] = set(direct_called)
                for h_name in helper_calls:
                    called_symbols.update(local_helpers.get(h_name, set()))
                    for r in local_helpers_cmd_records.get(h_name, []):
                        if not any(cr.command_name == r.command_name and cr.dispatch_api == r.dispatch_api for cr in command_records):
                            command_records.append(r)

                # Tier 1: Test directly calls a changed symbol (confirmed match)
                confirmed_direct_matches = [m for m in direct_matches if not m[5]]
                if confirmed_direct_matches:
                    best_match = sorted(
                        confirmed_direct_matches,
                        key=lambda m: (
                            0 if m[0].change_type == ChangeType.MODIFIED else 1,
                            -m[3],  # confidence
                            m[0].canonical_id,
                        ),
                    )[0]
                    cs, match_qual, desc, conf, unc, is_ambig = best_match
                    target_sym_display = cs.qualified_name
                    seen_tests.add(test_key)
                    prioritized.append(
                        PrioritizedTest(
                            test_name=test_identifier,
                            test_file=rel_test_path,
                            line_number=lineno,
                            priority_tier=PriorityTier.CRITICAL,
                            priority_score=0.98,
                            confidence=conf,
                            reason=f"Direct test of changed symbol '{target_sym_display}' in {cs.file_path}",
                            selection_reason=f"Directly calls changed symbol '{target_sym_display}' in {cs.file_path}. Immediate regression test recommended.",
                            targeted_symbol=target_sym_display,
                            targeted_file=cs.file_path,
                            call_depth=1,
                            impact_distance="direct",
                            evidence_type="confirmed",
                            match_classification=EventMatchClassification.NO_RELEVANT_EVENT_EVIDENCE,
                            explanation=f"Directly calls changed symbol '{target_sym_display}' in {cs.file_path}. Immediate regression test recommended.",
                            execution_command=f"pytest {rel_test_path}::{test_identifier} -v",
                            evidence=EvidenceTrail(
                                call_chain=[test_identifier, target_sym_display],
                                caller_file=rel_test_path,
                                line_number=lineno,
                                resolution_engine="ast_call_match",
                                match_quality=match_qual,
                                evidence_source="AST Call Match",
                                call_path_description=f"{test_identifier} -> {target_sym_display}",
                                is_ambiguous=False,
                            ),
                            structured_evidence=[
                                EvidenceItem(
                                    type="CHANGED_SYMBOL_REFERENCE",
                                    description=desc,
                                    source_file=rel_test_path,
                                    line=lineno,
                                )
                            ],
                            uncertainty_score=unc,
                        )
                    )
                    continue

                # Tier 1b: Test calls changed symbol via local helper
                helper_matches: list[tuple[str, ChangedSymbol, str, float, float]] = []
                for h_name in helper_calls:
                    if h_name in local_helpers_matched_symbols:
                        for cs, m_qual, conf, unc, is_ambig in local_helpers_matched_symbols[h_name]:
                            if not is_ambig:
                                helper_matches.append((h_name, cs, m_qual, conf, unc))

                if helper_matches:
                    best_helper = sorted(
                        helper_matches,
                        key=lambda hm: (
                            0 if hm[1].change_type == ChangeType.MODIFIED else 1,
                            -hm[3],
                            hm[1].canonical_id,
                        ),
                    )[0]
                    h_name, cs, match_qual, conf, unc = best_helper
                    target_sym_display = cs.qualified_name
                    seen_tests.add(test_key)
                    prioritized.append(
                        PrioritizedTest(
                            test_name=test_identifier,
                            test_file=rel_test_path,
                            line_number=lineno,
                            priority_tier=PriorityTier.HIGH,
                            priority_score=0.85,
                            confidence=0.85,
                            reason=f"Tests changed symbol '{target_sym_display}' via helper in {rel_test_path}",
                            selection_reason=f"Invokes test helper that calls changed symbol '{target_sym_display}'.",
                            targeted_symbol=target_sym_display,
                            targeted_file=cs.file_path,
                            call_depth=2,
                            impact_distance="1 hop (helper)",
                            evidence_type="confirmed",
                            match_classification=EventMatchClassification.NO_RELEVANT_EVENT_EVIDENCE,
                            explanation=f"Invokes test helper that calls changed symbol '{target_sym_display}'.",
                            execution_command=f"pytest {rel_test_path}::{test_identifier} -v",
                            evidence=EvidenceTrail(
                                call_chain=[test_identifier, h_name, target_sym_display],
                                caller_file=rel_test_path,
                                line_number=lineno,
                                resolution_engine="ast_helper_call_match",
                                match_quality="HELPER_DEPENDENCY",
                                evidence_source="AST Helper Call Match",
                                call_path_description=f"{test_identifier} -> {h_name} -> {target_sym_display}",
                                is_ambiguous=False,
                            ),
                            structured_evidence=[
                                EvidenceItem(
                                    type="HELPER_CALL",
                                    description=f"Invokes test helper '{h_name}' which calls changed symbol '{target_sym_display}'.",
                                    source_file=rel_test_path,
                                    line=lineno,
                                )
                            ],
                            uncertainty_score=0.15,
                        )
                    )
                    continue

                # Tier 1c: Event-driven test match
                matched_event: Optional[tuple[ImpactNode, str, bool, list[CommandDispatchRecord]]] = None
                for ev_node in event_impacts:
                    ev_name = ev_node.symbol_name.replace("event:", "")
                    hints = self.event_detector.get_trigger_hints(ev_name)

                    # Check direct explicit event send (e.g. post_migrate.send(...), event.emit(...))
                    direct_event_send = (
                        ev_name in direct_called
                        or any(expr.startswith(f"{ev_name}.") or f"{ev_name}.send" in expr for expr in attribute_exprs)
                    )
                    if direct_event_send:
                        matched_event = (ev_node, f"{ev_name}.send", True, [])
                        break

                    # Check call-shape-aware trigger command heuristic (e.g. call_command("migrate", ...))
                    matching_recs = [r for r in command_records if r.command_name in hints]
                    if matching_recs:
                        matched_event = (ev_node, matching_recs[0].command_name, False, matching_recs)
                        break

                if matched_event:
                    ev_node, trigger_expr, is_direct_event, matched_cmd_recs = matched_event
                    ev_name = ev_node.symbol_name.replace("event:", "")
                    h_name = ev_node.root_changed_symbol
                    seen_tests.add(test_key)

                    behavioral_items, missing_ev = self._analyze_behavioral_evidence(
                        node=node,
                        test_identifier=test_identifier,
                        rel_test_path=rel_test_path,
                        ev_node=ev_node,
                        changed_symbols=changed_symbols,
                        changed_files=resolved_changed_files,
                        imported_symbols=imported_symbols,
                        direct_called=direct_called,
                        attribute_exprs=attribute_exprs,
                        command_records=command_records,
                        lineno=lineno,
                    )
                    has_behavioral = any(
                        item.type in ("BEHAVIORAL_ASSERTION", "CALL_GRAPH_REACHABILITY")
                        for item in behavioral_items
                    )

                    if is_direct_event:
                        if has_behavioral:
                            classification = EventMatchClassification.BEHAVIORAL_COVERAGE
                            prio_tier = PriorityTier.HIGH
                            prio_score = 0.88
                            ev_type = "confirmed"
                            expl = f"Explicitly dispatches '{ev_name}' with verified behavioral coverage of '{h_name}'."
                            reason = f"Explicitly dispatches '{ev_name}' with behavioral coverage of '{h_name}'"
                            match_qual = "BEHAVIOR_SUPPORTED_EVENT_DISPATCH"
                            ev_source = f"Event Registration + Behavioral Assertion ({ev_node.evidence.evidence_source})"
                            call_path = f"{test_identifier} -> {trigger_expr} -> {ev_name} -> {h_name} [behavior-verified]"
                            is_ambig = False
                            uncertainty = 0.12
                            struct_ev = [EvidenceItem(type="EXPLICIT_DISPATCH", description=f"Explicitly dispatches event '{ev_name}'.", source_file=rel_test_path, line=lineno)] + behavioral_items
                        else:
                            classification = EventMatchClassification.EXPLICIT_EVENT_DISPATCH
                            prio_tier = PriorityTier.MEDIUM
                            prio_score = 0.70
                            ev_type = "confirmed"
                            expl = f"Directly invokes event '{ev_name}' (explicit dispatch), but behavior-specific coverage of '{h_name}' is not statically verified."
                            reason = f"Directly invokes event '{ev_name}' registered to '{h_name}'"
                            match_qual = "CONFIRMED_EVENT_DISPATCH"
                            ev_source = f"Event Registration ({ev_node.evidence.evidence_source})"
                            call_path = f"{test_identifier} -> {trigger_expr} -> {h_name}"
                            is_ambig = not ev_node.is_confirmed
                            uncertainty = 0.25
                            struct_ev = [EvidenceItem(type="EXPLICIT_DISPATCH", description=f"Directly invokes event '{ev_name}' via send.", source_file=rel_test_path, line=lineno)]
                    else:
                        if has_behavioral:
                            classification = EventMatchClassification.BEHAVIORAL_COVERAGE
                            prio_tier = PriorityTier.HIGH
                            prio_score = 0.85
                            ev_type = "confirmed"
                            expl = f"Exercises event trigger '{trigger_expr}' with verified behavioral coverage of '{h_name}'."
                            reason = f"Exercises event trigger '{trigger_expr}' with behavioral coverage of '{h_name}'"
                            match_qual = "BEHAVIOR_SUPPORTED_TRIGGER"
                            ev_source = f"Event Trigger + Behavioral Assertion ({ev_node.evidence.evidence_source})"
                            helper_prefix = f" -> {' -> '.join(sorted(helper_calls))}" if helper_calls else ""
                            call_path = f"{test_identifier}{helper_prefix} -> {trigger_expr} -> {ev_name} -> {h_name} [behavior-verified]"
                            is_ambig = False
                            uncertainty = 0.15
                            struct_ev = [EvidenceItem(type="COMMAND_DISPATCH", description=f"Invokes trigger command '{trigger_expr}' associated with '{ev_name}'.", source_file=rel_test_path, line=lineno)] + behavioral_items
                            if helper_calls:
                                struct_ev.append(EvidenceItem(type="LOCAL_HELPER_CALL", description=f"Invokes trigger via helper: {', '.join(sorted(helper_calls))}", source_file=rel_test_path, line=lineno))
                        else:
                            classification = EventMatchClassification.EVENT_TRIGGER_CANDIDATE
                            prio_tier = PriorityTier.MEDIUM
                            prio_score = 0.45
                            ev_type = "heuristic"
                            expl = (
                                f"Exercises command/trigger candidate '{trigger_expr}' associated with event '{ev_name}' "
                                f"registered to handler '{h_name}' (event dispatch is not statically verified). Candidate match with unconfirmed behavioral coverage."
                            )
                            reason = (
                                f"Exercises candidate command trigger '{trigger_expr}' for '{ev_name}' registered to '{h_name}' "
                                f"(event dispatch not statically verified)"
                            )
                            match_qual = "HEURISTIC_COMMAND_TRIGGER"
                            ev_source = f"Event Registration Heuristic ({ev_node.evidence.evidence_source}) - event dispatch is not statically verified"
                            helper_prefix = f" -> {' -> '.join(sorted(helper_calls))}" if helper_calls else ""
                            call_path = f"{test_identifier}{helper_prefix} -> {trigger_expr} [candidate] -> {ev_name} (event dispatch not statically verified) -> {h_name}"
                            is_ambig = True
                            uncertainty = 0.35
                            struct_ev = [EvidenceItem(type="COMMAND_DISPATCH", description=f"The test invokes {trigger_expr}.", source_file=rel_test_path, line=lineno)]
                            if helper_calls:
                                struct_ev.append(EvidenceItem(type="LOCAL_HELPER_CALL", description=f"Invokes trigger via helper: {', '.join(sorted(helper_calls))}", source_file=rel_test_path, line=lineno))

                    chain = [test_identifier]
                    for h in sorted(helper_calls):
                        chain.append(f"helper:{h}")
                    chain.extend([f"trigger:{trigger_expr}", f"event:{ev_name}", h_name])

                    test_candidate = PrioritizedTest(
                        test_name=test_identifier,
                        test_file=rel_test_path,
                        line_number=lineno,
                        priority_tier=prio_tier,
                        priority_score=prio_score,
                        reason=reason,
                        selection_reason=expl,
                        targeted_symbol=h_name,
                        targeted_file=ev_node.root_changed_file,
                        call_depth=2,
                        impact_distance=f"event ({ev_name})",
                        evidence_type=ev_type,
                        explanation=expl,
                        execution_command=f"pytest {rel_test_path}::{test_identifier} -v",
                        evidence=EvidenceTrail(
                            call_chain=chain,
                            caller_file=rel_test_path,
                            line_number=lineno,
                            resolution_engine="event_trigger_matcher" if not is_direct_event else "event_dispatch_matcher",
                            match_quality=match_qual,
                            evidence_source=ev_source,
                            call_path_description=call_path,
                            is_ambiguous=is_ambig,
                        ),
                        uncertainty_score=uncertainty,
                        confidence=round(1.0 - uncertainty, 2),
                        match_classification=classification,
                        event_name=ev_name,
                        missing_evidence=missing_ev,
                        structured_evidence=struct_ev,
                    )

                    if classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE:
                        event_candidates.append(test_candidate)
                        if include_candidates:
                            prioritized.append(test_candidate)
                    else:
                        prioritized.append(test_candidate)
                    continue

                # Tier 2: Test calls a direct (1-hop) impacted caller
                direct_caller_matches = [
                    s for s in called_symbols
                    if s in direct_sym_map
                    and s not in self.GENERIC_VERBS
                    and s not in LocalCodeGraphFallback.GENERIC_METHOD_NAMES
                ]
                if direct_caller_matches:
                    matched_caller = sorted(direct_caller_matches)[0]
                    impact = direct_sym_map[matched_caller]
                    seen_tests.add(test_key)
                    ev_type = "confirmed" if impact.is_confirmed else "heuristic"
                    prioritized.append(
                        PrioritizedTest(
                            test_name=test_identifier,
                            test_file=rel_test_path,
                            line_number=lineno,
                            priority_tier=PriorityTier.HIGH,
                            priority_score=0.82,
                            confidence=round(1.0 - impact.uncertainty_score, 2),
                            reason=f"Tests direct caller '{matched_caller}' impacted by '{impact.root_changed_symbol}'",
                            selection_reason=f"Exercises direct caller '{matched_caller}' which depends on '{impact.root_changed_symbol}'.",
                            targeted_symbol=matched_caller,
                            targeted_file=impact.file_path,
                            call_depth=2,
                            impact_distance="1 hop",
                            evidence_type=ev_type,
                            match_classification=EventMatchClassification.NO_RELEVANT_EVENT_EVIDENCE,
                            explanation=f"Exercises direct caller '{matched_caller}' which depends on '{impact.root_changed_symbol}'.",
                            execution_command=f"pytest {rel_test_path}::{test_identifier} -v",
                            evidence=EvidenceTrail(
                                call_chain=[test_identifier, *impact.evidence.call_chain],
                                caller_file=rel_test_path,
                                line_number=lineno,
                                resolution_engine=impact.evidence.resolution_engine,
                                match_quality="CALLER_DEPENDENCY",
                                evidence_source=impact.evidence.evidence_source,
                                call_path_description=f"{test_identifier} -> {impact.evidence.call_path_description}",
                                is_ambiguous=impact.evidence.is_ambiguous,
                            ),
                            structured_evidence=[
                                EvidenceItem(
                                    type="CALLER_DEPENDENCY",
                                    description=f"Exercises direct caller '{matched_caller}' which depends on '{impact.root_changed_symbol}'.",
                                    source_file=rel_test_path,
                                    line=lineno,
                                )
                            ],
                            uncertainty_score=impact.uncertainty_score,
                        )
                    )
                    continue

                # Tier 3: Test calls an indirect (2+ hops) impacted caller
                indirect_caller_matches = [
                    s for s in called_symbols
                    if s in indirect_sym_map
                    and s not in self.GENERIC_VERBS
                    and s not in LocalCodeGraphFallback.GENERIC_METHOD_NAMES
                ]
                if indirect_caller_matches:
                    matched_indirect = sorted(indirect_caller_matches)[0]
                    impact = indirect_sym_map[matched_indirect]
                    seen_tests.add(test_key)
                    ev_type = "confirmed" if impact.is_confirmed else "heuristic"
                    prioritized.append(
                        PrioritizedTest(
                            test_name=test_identifier,
                            test_file=rel_test_path,
                            line_number=lineno,
                            priority_tier=PriorityTier.MEDIUM,
                            priority_score=0.60,
                            confidence=round(1.0 - impact.uncertainty_score, 2),
                            reason=f"Tests indirect caller '{matched_indirect}' ({impact.depth} hops from '{impact.root_changed_symbol}')",
                            selection_reason=f"Exercises transitive caller '{matched_indirect}' connected via {len(impact.evidence.call_chain)-1} hops.",
                            targeted_symbol=matched_indirect,
                            targeted_file=impact.file_path,
                            call_depth=impact.depth + 1,
                            impact_distance=f"{impact.depth} hops",
                            evidence_type=ev_type,
                            match_classification=EventMatchClassification.NO_RELEVANT_EVENT_EVIDENCE,
                            explanation=f"Exercises transitive caller '{matched_indirect}' connected via {len(impact.evidence.call_chain)-1} hops.",
                            execution_command=f"pytest {rel_test_path}::{test_identifier} -v",
                            evidence=EvidenceTrail(
                                call_chain=[test_identifier, *impact.evidence.call_chain],
                                caller_file=rel_test_path,
                                line_number=lineno,
                                resolution_engine=impact.evidence.resolution_engine,
                                match_quality="TRANSITIVE_DEPENDENCY",
                                evidence_source=impact.evidence.evidence_source,
                                call_path_description=f"{test_identifier} -> {impact.evidence.call_path_description}",
                                is_ambiguous=impact.evidence.is_ambiguous,
                            ),
                            structured_evidence=[
                                EvidenceItem(
                                    type="CALLER_DEPENDENCY",
                                    description=f"Exercises transitive caller '{matched_indirect}' connected via {impact.depth} hops.",
                                    source_file=rel_test_path,
                                    line=lineno,
                                )
                            ],
                            uncertainty_score=impact.uncertainty_score,
                        )
                    )
                    continue

        self.last_discovered_tests_count = total_discovered
        self.last_event_candidates = event_candidates

        # Sort prioritized tests deterministically by score descending, then file, then test name
        prioritized.sort(key=lambda t: (-t.priority_score, t.test_file, t.test_name))
        return prioritized

    def _compute_uncertainty_score(
        self,
        depth: int,
        symbol_name: str,
        engine: str,
        is_ambiguous: bool = False,
    ) -> float:
        """
        Calculates confidence uncertainty (0.0 = certain, 1.0 = highly uncertain).
        Factors:
        - Hop depth: each transitive hop degrades certainty by +0.12
        - Resolution engine: local AST name matching has slight ambiguity risk (+0.05) vs Sourcegraph
        - Common symbol name heuristic (e.g. validate, get, run): +0.15 uncertainty
        - Ambiguous/unconfirmed reference: +0.35 uncertainty
        """
        uncertainty = 0.05 + ((depth - 1) * 0.12)

        if engine == "local_ast_fallback":
            uncertainty += 0.05

        if is_ambiguous:
            uncertainty += 0.35

        generic_names = {"get", "set", "run", "execute", "validate", "process", "handle", "update", "create"}
        if symbol_name.lower() in generic_names:
            uncertainty += 0.15

        return round(min(1.0, max(0.0, uncertainty)), 2)

    def _compute_uncertainty_reason(
        self,
        depth: int,
        symbol_name: str,
        engine: str,
        is_ambiguous: bool = False,
    ) -> str:
        """Constructs human-readable justification for uncertainty score."""
        reasons: list[str] = []
        if depth == 1:
            reasons.append("Direct call invocation")
        else:
            reasons.append(f"Transitive distance degradation ({depth} hops from change)")
        if engine == "local_ast_fallback":
            reasons.append("Single-repository AST syntax inspection")
        else:
            reasons.append("Sourcegraph cross-repository index")
        if is_ambiguous:
            reasons.append("Unanchored receiver attribute call without confirmed module import")
        generic_names = {"get", "set", "run", "execute", "validate", "process", "handle", "update", "create"}
        if symbol_name.lower() in generic_names:
            reasons.append(f"Symbol '{symbol_name}' is a common verb subject to name collisions")
        return "; ".join(reasons)

    def analyze(self, request: Optional[EvolutionRequest] = None) -> EvolutionReport:
        """
        Orchestrates full evolution analysis workflow:
        1. Resolve git diff between base_ref and target_ref
        2. Extract changed Python symbols
        3. Build transitive blast-radius graph
        4. Prioritize existing test suite
        5. Quantify evidence, uncertainty, and latency
        """
        start_time = time.perf_counter()
        req = request or EvolutionRequest()
        stage_timings: dict[str, float] = {}

        # Step 1: Git Diff
        t_stage = time.perf_counter()
        diff_text, changed_files, diff_stat = self.resolve_git_diff(
            base_ref=req.base_ref,
            target_ref=req.target_ref,
        )
        stage_timings["git_diff_resolution_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Step 2: Extract changed symbols
        t_stage = time.perf_counter()
        changed_symbols = self.extract_changed_symbols(
            diff_text=diff_text,
            changed_files=changed_files,
            base_ref=req.base_ref,
            target_ref=req.target_ref,
            renamed_files=self.last_renamed_files,
            file_statuses=self.last_file_statuses,
        )
        stage_timings["changed_symbol_extraction_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Step 3: Build transitive impact graph
        t_stage = time.perf_counter()
        direct_impacts, indirect_impacts = self.build_transitive_impact_graph(
            changed_symbols=changed_symbols,
            max_depth=req.max_depth,
        )
        stage_timings["impact_graph_construction_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Step 3b: Evidence-Grounded Semantic Analysis & RAG
        semantic_impacts: list[SemanticImpactAssessment] = []
        if getattr(req, "enable_semantic_analysis", True):
            t_stage = time.perf_counter()
            try:
                from testpilot.evolution.semantic import SemanticImpactAnalyzer
                semantic_analyzer = SemanticImpactAnalyzer(
                    repo_root=str(self.repo_root),
                    sourcegraph_client=self.sourcegraph,
                )
                all_impact_nodes = direct_impacts + indirect_impacts
                semantic_impacts = semantic_analyzer.analyze_impacts(
                    changed_symbols=changed_symbols,
                    impact_nodes=all_impact_nodes,
                    enable_llm=getattr(req, "enable_llm_semantic", False),
                )
            except Exception as se:
                warnings_notice = f"Semantic impact analysis notice: {se}"
                logger.warning(warnings_notice)
            stage_timings["semantic_analysis_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Step 4: Prioritize tests
        t_stage = time.perf_counter()
        all_prioritized = self.prioritize_tests(
            changed_symbols=changed_symbols,
            direct_impacts=direct_impacts,
            indirect_impacts=indirect_impacts,
            include_candidates=True,
            changed_files=changed_files,
        )
        stage_timings["test_prioritization_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Distinguish prioritized regression tests from trigger-only candidates
        regression_tests = [
            t for t in all_prioritized
            if t.match_classification != EventMatchClassification.EVENT_TRIGGER_CANDIDATE
        ]
        candidate_tests = [
            t for t in all_prioritized
            if t.match_classification == EventMatchClassification.EVENT_TRIGGER_CANDIDATE
        ]

        t_stage = time.perf_counter()
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        total_impacts = len(direct_impacts) + len(indirect_impacts)

        # Step 5: Warnings & Limitations
        warnings: list[str] = []
        if not self.sourcegraph.is_alive():
            warnings.append(
                "Sourcegraph OSS daemon is offline; using local Tree-sitter AST call-graph fallback with single-repository scope."
            )

        non_py = [f for f in changed_files if not f.endswith(".py")]
        if non_py:
            warnings.append(
                f"{len(non_py)} non-Python file(s) modified in diff ({', '.join(non_py[:3])}); impact analysis currently models Python symbols."
            )

        deleted_files = [f for f in changed_files if f.endswith(".py") and self.last_file_statuses.get(f) == "DELETED"]
        if deleted_files:
            warnings.append(
                f"{len(deleted_files)} deleted Python file(s) detected ({', '.join(deleted_files[:3])}); callers may trigger ImportError."
            )

        if self.last_renamed_files:
            warnings.append(
                f"{len(self.last_renamed_files)} renamed file(s) tracked with historical paths preserved."
            )

        # Report unresolved event-driven coverage when handlers are registered but no tests trigger them
        event_nodes = [d for d in direct_impacts if d.impact_type == ImpactType.EVENT_REGISTRATION]
        for ev_node in event_nodes:
            ev_name = ev_node.symbol_name.replace("event:", "")
            h_name = ev_node.root_changed_symbol
            if not any(t.targeted_symbol == h_name for t in all_prioritized):
                warnings.append(
                    f"Event-driven coverage could not be resolved: handler '{h_name}' is registered to event '{ev_name}', but no existing tests were found exercising triggers for this event."
                )

        stage_timings["report_generation_ms"] = round((time.perf_counter() - t_stage) * 1000.0, 2)

        # Step 6: Metadata metrics
        metrics: dict[str, Any] = {
            "total_changed_files": len(changed_files),
            "total_changed_symbols": len(changed_symbols),
            "direct_impact_count": len(direct_impacts),
            "indirect_impact_count": len(indirect_impacts),
            "total_discovered_tests": self.last_discovered_tests_count,
            "prioritized_test_count": len(regression_tests),
            "event_trigger_candidate_count": len(candidate_tests),
            "directly_supported_matches": sum(1 for t in regression_tests if t.call_depth == 1 and t.evidence_type == "confirmed"),
            "behavior_supported_matches": sum(1 for t in regression_tests if t.match_classification == EventMatchClassification.BEHAVIORAL_COVERAGE),
            "explicit_dispatch_matches": sum(1 for t in regression_tests if t.match_classification == EventMatchClassification.EXPLICIT_EVENT_DISPATCH),
            "critical_tests": sum(1 for t in regression_tests if t.priority_tier == PriorityTier.CRITICAL),
            "high_tests": sum(1 for t in regression_tests if t.priority_tier == PriorityTier.HIGH),
            "medium_tests": sum(1 for t in regression_tests if t.priority_tier == PriorityTier.MEDIUM),
            "low_tests": sum(1 for t in regression_tests if t.priority_tier == PriorityTier.LOW),
            "stage_timings": stage_timings,
        }

        semantic_summary = {
            "total_semantic_assessments": len(semantic_impacts),
            "direct_behavioral_count": sum(
                1 for s in semantic_impacts
                if s.classification == SemanticImpactClassification.DIRECT_BEHAVIORAL_DEPENDENCY
            ),
            "indirect_behavioral_count": sum(
                1 for s in semantic_impacts
                if s.classification == SemanticImpactClassification.INDIRECT_BEHAVIORAL_DEPENDENCY
            ),
            "structural_only_count": sum(
                1 for s in semantic_impacts
                if s.classification == SemanticImpactClassification.STRUCTURAL_ONLY
            ),
            "uncertain_count": sum(
                1 for s in semantic_impacts
                if s.classification == SemanticImpactClassification.UNCERTAIN
            ),
            "verified_evidence_count": sum(1 for s in semantic_impacts if s.is_verified),
        }

        return EvolutionReport(
            base_ref=req.base_ref,
            target_ref=req.target_ref or "HEAD",
            diff_stat=diff_stat,
            changed_files=changed_files,
            renamed_files=self.last_renamed_files,
            changed_symbols=changed_symbols,
            direct_impacts=direct_impacts,
            indirect_impacts=indirect_impacts,
            total_impacted_symbols=total_impacts,
            prioritized_tests=regression_tests,
            event_trigger_candidates=candidate_tests,
            total_discovered_tests=self.last_discovered_tests_count,
            analysis_latency_ms=elapsed_ms,
            warnings=warnings,
            metrics=metrics,
            semantic_impacts=semantic_impacts,
            semantic_summary=semantic_summary,
        )
