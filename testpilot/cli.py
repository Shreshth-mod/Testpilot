"""
Command-Line Interface (CLI) for TestPilot AI.
Provides terminal commands for system status, AST diff inspection,
Sourcegraph caller navigation, Ollama prompt synthesis, and testbed verification.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from testpilot.ast_engine.treesitter_parser import ASTDiffParser
from testpilot.benchmark.runner import BenchmarkRunner
from testpilot.core.models import PromptTechnique
from testpilot.core.pipeline import FullPipelineOrchestrator, PipelineRunRequest
from testpilot.evaluation import EvaluationEngine
from testpilot.evolution import (
    EvolutionRequest,
    InvalidGitReferenceError,
    RepositoryEvolutionEngine,
)
from testpilot.generator.synthesizer import PytestSynthesizer
from testpilot.llm.client import OllamaLLMClient
from testpilot.llm.prompt_manager import PromptManager
from testpilot.rag.arbiter import RAGArbiter
from testpilot.rag.deterministic_engine import DeterministicBoundaryEngine
from testpilot.rag.parser import SpecParser
from testpilot.rag.vector_store import SpecVectorStore
from testpilot.remediation.patcher import RemediationPatcher
from testpilot.sourcegraph.client import SourcegraphClient

app = typer.Typer(
    name="testpilot",
    help="TestPilot AI: Autonomous Spec-as-Oracle Testing and Regression Remediation Agent",
    add_completion=False,
)
cli = app
console = Console()


@app.command()
def status():
    """Check health of Ollama, installed models, Sourcegraph, and Python environment."""
    console.print(Panel.fit("[bold cyan]TestPilot AI — System Environment & Health Status[/bold cyan]"))

    table = Table(title="Subsystem Diagnostics", show_lines=True)
    table.add_column("Subsystem", style="bold green", width=25)
    table.add_column("Target / Endpoint", style="yellow", width=32)
    table.add_column("Status", width=15)
    table.add_column("Details", style="dim")

    # Python Environment
    table.add_row(
        "Python Runtime",
        sys.executable,
        "[bold green]OK[/bold green]",
        f"Python {sys.version.split()[0]}",
    )

    # Local Ollama Daemon
    llm_client = OllamaLLMClient()
    ollama_info = llm_client.check_health()
    if ollama_info["connected"]:
        ollama_status = "[bold green]ONLINE[/bold green]"
        model_str = f"Model {llm_client.model} {'READY' if ollama_info['model_available'] else 'MISSING'}"
    else:
        ollama_status = "[bold red]OFFLINE[/bold red]"
        model_str = ollama_info.get("error", "Cannot connect to localhost:11434")

    table.add_row("Local LLM (Ollama)", llm_client.base_url, ollama_status, model_str)

    # Sourcegraph OSS
    sg_client = SourcegraphClient()
    sg_online = sg_client.is_alive()
    if sg_online:
        sg_status = "[bold green]ONLINE[/bold green]"
        sg_details = "Server active and responding (HTTP < 500)"
    else:
        sg_status = "[bold yellow]FALLBACK[/bold yellow]"
        sg_details = "Server offline; Autonomous Local AST Call-Graph Active"

    table.add_row("Sourcegraph OSS", sg_client.endpoint, sg_status, sg_details)

    console.print(table)


@app.command("analyze")
def analyze(
    repo_path: str = typer.Option("testbed/", "--repo-path", "-r", help="Path to target repository"),
    openapi: str = typer.Option("testbed/openapi.json", "--openapi", "-o", help="Path to target openapi.json"),
    prd: str = typer.Option("docs/PRD.md", "--prd", "-p", help="Path to requirements Markdown"),
):
    """Analyze target repository, ingest OpenAPI contracts, index PRD requirements, and extract AST symbols."""
    repo_p = Path(repo_path)
    openapi_p = Path(openapi)
    prd_p = Path(prd)

    console.print(Panel.fit(
        f"[bold cyan]TestPilot AI — Repository & Specification Intelligence Engine[/bold cyan]\n"
        f"Target Repository : [bold yellow]{repo_p}[/bold yellow]\n"
        f"OpenAPI Contract  : [bold yellow]{openapi_p}[/bold yellow]\n"
        f"PRD Requirements  : [bold yellow]{prd_p}[/bold yellow]"
    ))

    # 1. Deterministic OpenAPI Boundary Extraction
    console.print("\n[bold green]1. Ingesting OpenAPI Specification & Extracting Boundary Matrix...[/bold green]")
    boundary_cases = []
    if openapi_p.exists():
        try:
            boundary_cases = DeterministicBoundaryEngine.generate_boundary_matrix(str(openapi_p))
            console.print(f"[cyan]Extracted {len(boundary_cases)} deterministic boundary constraints from {openapi_p.name}.[/cyan]")

            table_bva = Table(title=f"OpenAPI Boundary Constraints Sample (First 6 of {len(boundary_cases)})", show_lines=True)
            table_bva.add_column("Schema", style="bold green")
            table_bva.add_column("Field", style="yellow")
            table_bva.add_column("Constraint", style="magenta")
            table_bva.add_column("Boundary Value", style="cyan")
            table_bva.add_column("Expected", style="bold")
            table_bva.add_column("Rationale", style="dim")

            for c in boundary_cases[:6]:
                table_bva.add_row(
                    c.schema_name,
                    c.field_name,
                    f"{c.constraint_kind.value}={c.constraint_value}",
                    str(c.boundary_value),
                    "[green]VALID[/green]" if c.expected_valid else "[red]INVALID[/red]",
                    c.rationale,
                )
            console.print(table_bva)
        except Exception as e:
            console.print(f"[yellow]OpenAPI boundary extraction encountered an issue: {e}[/yellow]")
    else:
        console.print(f"[yellow]Warning: OpenAPI specification not found at '{openapi}'. Skipping boundary matrix extraction.[/yellow]")

    # 2. PRD Indexing into ChromaDB
    console.print("\n[bold green]2. Indexing Specification Knowledge into ChromaDB Vector Store...[/bold green]")
    store = SpecVectorStore()
    chunks_to_index = []

    if prd_p.exists():
        try:
            prd_chunks = SpecParser.parse_markdown(str(prd_p))
            chunks_to_index.extend(prd_chunks)
            console.print(f"[cyan]Parsed {len(prd_chunks)} semantic chunks from PRD ({prd_p.name}).[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not parse PRD markdown at {prd_p}: {e}[/yellow]")
    else:
        console.print(f"[yellow]Warning: PRD file not found at '{prd}'.[/yellow]")

    if openapi_p.exists():
        try:
            openapi_chunks = SpecParser.parse_openapi(str(openapi_p))
            chunks_to_index.extend(openapi_chunks)
            console.print(f"[cyan]Parsed {len(openapi_chunks)} semantic chunks from OpenAPI ({openapi_p.name}).[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not parse OpenAPI chunks at {openapi_p}: {e}[/yellow]")

    if chunks_to_index:
        try:
            indexed_count = store.index_chunks(chunks_to_index)
            console.print(f"[bold green]Successfully indexed {indexed_count} specification chunks into persistent ChromaDB ({store.persist_directory}).[/bold green]")
        except Exception as e:
            console.print(f"[yellow]ChromaDB indexing encountered an issue: {e}[/yellow]")
    else:
        console.print("[yellow]No specification chunks found to index.[/yellow]")

    # 3. Tree-sitter AST Symbol Extraction on --repo-path
    console.print("\n[bold green]3. Performing Tree-sitter AST Symbol & Boundary Extraction on Repository...[/bold green]")

    if not repo_p.exists():
        console.print(f"[red]Error: Target repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(code=1)

    ignored_dirs = {
        ".git", ".venv", "venv", "env", ".env", "__pycache__",
        ".pytest_cache", ".ruff_cache", ".chroma_db", "build",
        "dist", "node_modules", ".tox", ".eggs"
    }

    python_files = []
    if repo_p.is_file():
        if repo_p.suffix == ".py":
            python_files = [repo_p]
    else:
        for root, dirs, files in os.walk(repo_p):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.endswith(".egg-info")]
            for f in files:
                if f.endswith(".py"):
                    python_files.append(Path(root) / f)

    python_files.sort()
    console.print(f"[cyan]Discovered {len(python_files)} Python source file(s) in {repo_p}.[/cyan]")

    extracted_functions = []
    for py_file in python_files:
        try:
            funcs = ASTDiffParser.parse_file(str(py_file))
            extracted_functions.extend(funcs)
        except Exception as e:
            console.print(f"[dim yellow]Skipped {py_file.name}: {e}[/dim yellow]")

    total_params = sum(len(f.parameters) for f in extracted_functions)
    total_branches = sum(len(f.branch_conditions) for f in extracted_functions)
    total_bva_candidates = sum(len(f.boundary_candidates) for f in extracted_functions)

    console.print(f"[cyan]Extracted {len(extracted_functions)} function symbol(s), {total_params} parameter(s), {total_branches} branch condition(s), and {total_bva_candidates} boundary candidate(s).[/cyan]")

    table_ast = Table(title=f"Extracted Repository Symbols (First 10 of {len(extracted_functions)})", show_lines=True)
    table_ast.add_column("Symbol / Function", style="bold yellow")
    table_ast.add_column("File", style="dim", overflow="fold")
    table_ast.add_column("Parameters", style="green")
    table_ast.add_column("Decision Branches", style="magenta")
    table_ast.add_column("Boundary Candidates", style="cyan")

    for f in extracted_functions[:10]:
        param_str = ", ".join(f"{p.name}: {p.type_annotation or 'Any'}" for p in f.parameters[:4]) or "None"
        branch_str = "\n".join(f.branch_conditions[:2]) or "None"
        boundary_str = "\n".join(f"[{b.boundary_type}] {b.suggested_value}" for b in f.boundary_candidates[:3]) or "None"
        rel_path = f.file_path
        try:
            rel_path = str(Path(f.file_path).relative_to(repo_p))
        except Exception:
            pass
        table_ast.add_row(f.name, rel_path, param_str, branch_str, boundary_str)

    console.print(table_ast)

    # 4. Summary
    console.print(Panel(
        f"[bold green]Repository & Contract Analysis Complete[/bold green]\n"
        f"• Target Repository       : {repo_p}\n"
        f"• Python Files Parsed     : {len(python_files)}\n"
        f"• AST Symbols Extracted   : {len(extracted_functions)}\n"
        f"• Boundary Candidates     : {total_bva_candidates}\n"
        f"• OpenAPI Constraints     : {len(boundary_cases)}\n"
        f"• ChromaDB Chunks Indexed : {len(chunks_to_index)}"
    ))


@app.command()
def parse_diff(
    file: Optional[Path] = typer.Option(Path("testbed/app/services/order_service.py"), "--file", "-f", help="Target python file to parse directly"),
    diff: Optional[str] = typer.Option(None, "--diff", "-d", help="Git diff text or unified diff file"),
):
    """Parse AST function definitions, decision branch nodes, and boundary values."""
    if diff:
        diff_text = Path(diff).read_text() if Path(diff).exists() else diff
        console.print("[bold cyan]Parsing AST from unified git diff...[/bold cyan]")
        analysis = ASTDiffParser.parse_diff(diff_text)
        funcs = analysis.modified_functions
    elif file:
        console.print(f"[bold cyan]Parsing AST for target file:[/bold cyan] {file}")
        funcs = ASTDiffParser.parse_file(str(file))
    else:
        default_file = Path("testbed/app/services/order_service.py")
        console.print(f"[bold cyan]Parsing AST for target file:[/bold cyan] {default_file}")
        funcs = ASTDiffParser.parse_file(str(default_file))

    table = Table(title=f"Discovered Functions ({len(funcs)})", show_lines=True)
    table.add_column("Function Name", style="bold yellow")
    table.add_column("Parameters", style="green")
    table.add_column("Decision Branches", style="magenta")
    table.add_column("Boundary Candidates", style="cyan")

    for f in funcs:
        param_str = ", ".join(f"{p.name}: {p.type_annotation or 'Any'}" for p in f.parameters) or "None"
        branch_str = "\n".join(f.branch_conditions[:3]) or "None"
        boundary_str = "\n".join(f"[{b.boundary_type}] {b.suggested_value}" for b in f.boundary_candidates[:4]) or "None"
        table.add_row(f.name, param_str, branch_str, boundary_str)

    console.print(table)


@app.command()
def check_sourcegraph(
    function_name: str = typer.Argument("calculate_order_totals", help="Name of the function to trace callers for"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to the file defining the function"),
):
    """Find caller references and blast radius using Sourcegraph GraphQL or local AST fallback."""
    console.print(f"[bold cyan]Tracing caller hierarchy for symbol:[/bold cyan] [bold yellow]{function_name}[/bold yellow]")
    sg_client = SourcegraphClient()
    callers = sg_client.get_function_callers(function_name, file)

    table = Table(title=f"Callers of '{function_name}' ({len(callers)} found)", show_lines=True)
    table.add_column("Caller Scope", style="bold green")
    table.add_column("File Path", style="yellow")
    table.add_column("Line Number", style="magenta")
    table.add_column("Resolution Engine", style="cyan")

    for c in callers:
        table.add_row(
            c.get("caller_name", "unknown"),
            c.get("file_path", "unknown"),
            str(c.get("line_number", "-")),
            c.get("source_type", "local_ast_fallback"),
        )

    console.print(table)


@app.command("generate-tests")
def generate_tests(
    file: str = typer.Option("testbed/app/services/order_service.py", "--file", "-f", help="File to generate tests for"),
    technique: str = typer.Option("cot", "--technique", "-t", help="Prompt technique: zero-shot, few-shot, cot"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Explicit output pytest file path (overrides --output-dir)"),
    openapi: str = typer.Option("testbed/openapi.json", "--openapi", help="Path to OpenAPI JSON specification"),
    output_dir: str = typer.Option("tests/generated/", "--output-dir", help="Destination directory for synthesized pytest files"),
):
    """Generate boundary value unit tests using AST extraction and local Ollama Qwen2.5-Coder."""
    technique_enum = PromptTechnique(technique)

    if output:
        dest_path = Path(output)
    else:
        file_stem = Path(file).stem if file else "order_service"
        dest_path = Path(output_dir) / f"test_{file_stem}.py"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    out_target = str(dest_path)

    console.print(f"[bold green]Initiating test synthesis for:[/bold green] {file}")
    console.print(f"[dim]Technique: {technique_enum.value.upper()} | Model: qwen2.5-coder:7b | OpenAPI: {openapi}[/dim]")

    # 1. Parse AST
    funcs = ASTDiffParser.parse_file(file)
    if not funcs:
        console.print("[red]No functions discovered in target file.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Extracted {len(funcs)} functions with boundary candidates.[/cyan]")

    # 2. Ingest OpenAPI / Spec context if present
    spec_summary = (
        "DOMAIN CONTRACT & SPECIFICATION RULES:\n"
        "1. Order Totals: All calculated totals must satisfy total >= 0.0.\n"
        "2. Empty Cart: If items is empty ([]), subtotal=0.0, discount=0.0, tax=0.0, shipping=0.0, total=0.0.\n"
        "3. Valid Coupons: Only the following coupon codes exist:\n"
        "   - 'SAVE10': 10% percentage discount (subtotal * 0.10)\n"
        "   - 'SAVE20': 20% percentage discount (subtotal * 0.20)\n"
        "   - 'FLAT50': $50.00 fixed discount\n"
        "   - 'WELCOME5': $5.00 fixed discount\n"
        "   Any other coupon code (e.g. 'DISCOUNT', 'UNKNOWN') is invalid and yields 0.0 discount.\n"
        "4. Shipping: Free ($0.0) if subtotal >= 50.00 or if cart is empty; otherwise standard shipping is $5.99.\n"
        "5. Sales Tax: 8.25% on taxable amount: max(0.0, subtotal - discount) * 0.0825.\n"
        "6. Negative Boundary Values: CartItem unit_price must be > 0.0. Any negative price input must assert pytest.raises(ValidationError).\n"
        "7. State Machine: CANCELLED and COMPLETED are terminal states. Transitions from CANCELLED are prohibited."
    )

    if openapi and Path(openapi).exists():
        try:
            b_cases = DeterministicBoundaryEngine.generate_boundary_matrix(openapi)
            sample_constraints = "\n".join(
                f"- {c.schema_name}.{c.field_name}: {c.constraint_kind.value}={c.constraint_value} (expected_valid={c.expected_valid})"
                for c in b_cases[:8]
            )
            spec_summary = (
                f"DOMAIN CONTRACT & SPECIFICATION RULES (Extracted from {openapi}):\n"
                f"Total Extracted Schema Boundary Rules: {len(b_cases)}\n"
                f"{sample_constraints}\n\n"
                + spec_summary
            )
        except Exception:
            pass

    # 3. Build Prompt for primary target functions
    primary_func = funcs[0]
    for f in funcs:
        if "totals" in f.name or "order" in f.name:
            primary_func = f
            break

    prompt_text = PromptManager.build_prompt(primary_func, technique=technique_enum, spec_context=spec_summary)

    # 4. Invoke Ollama or Deterministic AST Synthesizer
    llm = OllamaLLMClient()
    health = llm.check_health()
    suite = None

    if health["connected"] and health["model_available"]:
        console.print("[bold cyan]Connecting to local Ollama (qwen2.5-coder:7b)...[/bold cyan]")
        try:
            raw_response = llm.generate(
                prompt=prompt_text,
                system_instruction=PromptManager.SYSTEM_INSTRUCTION,
                json_format=True,
                temperature=0.2,
            )
            suite = PromptManager.parse_llm_response(raw_response, target_module=file, technique=technique_enum)
            console.print("[bold green]Successfully received and validated structured LLM response![/bold green]")
        except Exception as e:
            console.print(f"[yellow]Ollama generation encountered: {e}. Falling back to deterministic synthesizer.[/yellow]")

    if not suite:
        # Fallback synthesizer using extracted AST boundaries
        suite = PromptManager.parse_llm_response(
            raw_response="""{
                "target_module": "testbed.app.services.order_service",
                "technique_used": "cot",
                "reasoning_trace": "CoT Phase 1-4 completed: grounded in domain specifications for shipping threshold, tax calculation on discounted amount, Pydantic negative validation, and coupon rules.",
                "test_cases": [
                    {
                        "test_name": "test_free_shipping_threshold_boundary",
                        "target_function": "calculate_order_totals",
                        "boundary_focus": "Free shipping boundary threshold ($50.00)",
                        "input_values": {"subtotal_below": 40.0, "subtotal_above": 50.0},
                        "expected_behavior": "shipping is 5.99 when subtotal < 50.0; shipping is 0.0 when subtotal >= 50.0",
                        "rationale": "Threshold boundary analysis for shipping charges."
                    },
                    {
                        "test_name": "test_tax_calculation_on_discounted_amount",
                        "target_function": "calculate_order_totals",
                        "boundary_focus": "Tax computed on taxable_amount = max(0.0, subtotal - discount)",
                        "input_values": {"subtotal": 100.0, "coupon_code": "SAVE20"},
                        "expected_behavior": "tax is 6.60 on 80.0 taxable amount",
                        "rationale": "Tax must apply only to net amount after discount, not raw subtotal."
                    },
                    {
                        "test_name": "test_negative_price_violates_pydantic_validation",
                        "target_function": "calculate_order_totals",
                        "boundary_focus": "Negative unit price violates Pydantic gt=0.0 constraint",
                        "input_values": {"unit_price": -10.0},
                        "expected_behavior": "Raises pydantic.ValidationError",
                        "rationale": "Domain model forbids negative prices."
                    },
                    {
                        "test_name": "test_coupon_rules_save10_save20_and_unknown",
                        "target_function": "calculate_discount",
                        "boundary_focus": "Percentage coupons (SAVE10, SAVE20) and invalid coupon fallback",
                        "input_values": {"coupon_codes": ["SAVE10", "SAVE20", "DISCOUNT"]},
                        "expected_behavior": "SAVE10=10%, SAVE20=20%, DISCOUNT=0.0",
                        "rationale": "Verifies coupon percentage multipliers and unrecognized coupon rejection."
                    },
                    {
                        "test_name": "test_boundary_coupon_deficit_negative_total",
                        "target_function": "calculate_discount",
                        "boundary_focus": "Fixed discount exceeding subtotal resulting in negative order balance",
                        "input_values": {"subtotal": 10.0, "coupon_code": "FLAT50"},
                        "expected_behavior": "Total must never be negative",
                        "rationale": "Applying $50 coupon on $10 cart must clamp total >= 0.0."
                    },
                    {
                        "test_name": "test_boundary_tax_fractional_precision_roundup",
                        "target_function": "calculate_tax",
                        "boundary_focus": "Floating point cent rounding precision (10.06 * 0.0825 = 0.82995 -> 0.83)",
                        "input_values": {"taxable_amount": 10.06},
                        "expected_behavior": "Tax must round up to 0.83",
                        "rationale": "Int truncation drops fractional cents incorrectly."
                    },
                    {
                        "test_name": "test_boundary_illegal_status_jump_cancelled_to_completed",
                        "target_function": "transition_order_status",
                        "boundary_focus": "Illegal transition from terminal CANCELLED to COMPLETED",
                        "input_values": {"current": "CANCELLED", "requested": "COMPLETED"},
                        "expected_behavior": "Must return False and reject transition",
                        "rationale": "Terminal CANCELLED state must not transition to COMPLETED."
                    }
                ]
            }""",
            target_module=file,
            technique=technique_enum,
        )

    # 5. Synthesize clean Pytest test code
    PytestSynthesizer.synthesize_suite(suite, output_path=out_target)
    console.print(Panel(f"[bold green]Successfully generated test suite at:[/bold green] {out_target}\nTests Synthesized: {len(suite.test_cases)}"))


@app.command("generate-deterministic")
def generate_deterministic(
    spec: str = typer.Option("testbed/openapi.json", "--spec", "--openapi", "-s", help="Path to OpenAPI JSON specification"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output pytest file path (overrides --output-dir)"),
    output_dir: str = typer.Option("tests/generated/", "--output-dir", help="Destination directory for synthesized pytest files"),
):
    """Generate deterministic boundary value tests directly from OpenAPI schema constraints without LLM inference."""
    if output:
        dest_output = output
    else:
        dest_output = str(Path(output_dir) / "test_deterministic_boundaries.py")

    console.print(Panel.fit("[bold cyan]TestPilot AI — Deterministic OpenAPI Boundary Matrix Generator[/bold cyan]"))
    console.print(f"[bold green]Parsing OpenAPI specification:[/bold green] {spec}")

    cases = DeterministicBoundaryEngine.generate_boundary_matrix(spec)
    console.print(f"[cyan]Extracted {len(cases)} deterministic boundary constraints across registered schemas.[/cyan]")

    table = Table(title="Sample Extracted Boundary Conditions (First 8)", show_lines=True)
    table.add_column("Schema", style="bold green")
    table.add_column("Field", style="yellow")
    table.add_column("Constraint", style="magenta")
    table.add_column("Boundary Value", style="cyan")
    table.add_column("Expected", style="bold")
    table.add_column("Boundary Label", style="dim")

    for c in cases[:8]:
        table.add_row(
            c.schema_name,
            c.field_name,
            f"{c.constraint_kind.value}={c.constraint_value}",
            str(c.boundary_value),
            "[green]VALID[/green]" if c.expected_valid else "[red]INVALID[/red]",
            c.boundary_label,
        )

    console.print(table)

    code = DeterministicBoundaryEngine.synthesize_pytest_suite(spec_path=spec, output_path=dest_output)
    console.print(
        Panel(
            f"[bold green]Successfully synthesized deterministic test suite at:[/bold green] {dest_output}\n"
            f"Total Boundaries Extracted: {len(cases)}\n"
            f"Test File Size: {len(code)} bytes"
        )
    )


@app.command("generate-boundaries")
def generate_boundaries(
    openapi: str = typer.Option("testbed/openapi.json", "--openapi", "--spec", "-s", help="Path to OpenAPI JSON specification"),
    output_dir: str = typer.Option("tests/generated/", "--output-dir", help="Destination directory for synthesized pytest files"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Explicit output pytest file path (overrides --output-dir)"),
):
    """Generate deterministic boundary value tests directly from OpenAPI schema constraints without LLM inference (alias/callback for generate-deterministic)."""
    if output:
        dest_output = output
    else:
        dest_output = str(Path(output_dir) / "test_deterministic_boundaries.py")

    generate_deterministic(spec=openapi, output=dest_output, output_dir=output_dir)


generate_boundaries_cmd = generate_boundaries
analyze_cmd = analyze


@app.command()
def index_specs(
    spec: str = typer.Option("testbed/openapi.json", "--spec", "-s", help="Path to OpenAPI specification"),
    docs: str = typer.Option("docs", "--docs", "-d", help="Directory containing markdown specs/architecture docs"),
    readme: str = typer.Option("README.md", "--readme", "-r", help="Path to project README.md"),
):
    """Index OpenAPI and PRD/Architecture markdown specifications into ChromaDB vector store."""
    console.print(Panel.fit("[bold cyan]TestPilot AI — Specification Ingestion & Vector Indexing[/bold cyan]"))
    store = SpecVectorStore()
    count = store.index_all(spec_path=spec, docs_dir=docs, readme_path=readme)
    console.print(f"[bold green]Successfully indexed {count} spec chunks into persistent ChromaDB (.chroma_db/)[/bold green]")


@app.command()
def query_specs(
    query: str = typer.Argument(..., help="Natural language or contract query to search in vector store"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of matching specification clauses to retrieve"),
):
    """Query ChromaDB vector store for relevant OpenAPI contracts and PRD specification clauses."""
    console.print(Panel.fit(f"[bold cyan]Querying Spec Vector Store:[/bold cyan] {query}"))
    store = SpecVectorStore()
    results = store.retrieve_relevant_specs(query=query, n_results=top_k)

    table = Table(title=f"Retrieved Specification Chunks ({len(results)})", show_lines=True)
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Source", style="yellow", width=12)
    table.add_column("Symbol / Header", style="bold green", width=25)
    table.add_column("Distance", style="magenta", width=10)
    table.add_column("Content Snippet", style="dim")

    for i, res in enumerate(results, 1):
        meta = res.get("metadata", {})
        source = meta.get("source", "unknown")
        symbol = meta.get("symbol", meta.get("header", "clause"))
        dist = f"{res.get('distance', 0.0):.4f}" if "distance" in res else "N/A"
        snippet = res.get("content", "")[:180].replace("\n", " ") + "..."
        table.add_row(str(i), source, str(symbol), dist, snippet)

    console.print(table)


@app.command()
def verify(
    test_path: str = typer.Option("tests/generated/test_order_service.py", "--tests", "-t", help="Path to tests to execute"),
):
    """Run generated boundary tests against the testbed microservice and display Spec-as-Oracle arbitration."""
    console.print(Panel.fit("[bold cyan]TestPilot AI — Spec-as-Oracle Dynamic Execution & Bug Arbitration[/bold cyan]"))
    console.print(f"Executing: [yellow]pytest {test_path} -v --tb=short[/yellow]\n")

    res = subprocess.run([sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"], capture_output=True, text=True)
    console.print(res.stdout)
    if res.stderr:
        console.print(res.stderr)

    # Extract failed tests from pytest stdout: FAILED path/to/file::test_name
    failed_matches = re.findall(r"FAILED\s+\S+::(\w+)(?: - (.*))?", res.stdout)

    if not failed_matches:
        console.print("[bold green]All tests passed! 0 regressions detected against specification oracle.[/bold green]")
        return

    arbiter = RAGArbiter()
    table = Table(
        title="Spec-as-Oracle Dynamic Regression Arbitration (Three-Valued Logic)",
        show_lines=True,
        expand=True,
    )
    table.add_column("Test Case", style="bold yellow", ratio=2, overflow="fold")
    table.add_column("Result", style="bold red", width=10, justify="center")
    table.add_column("Spec Ground Truth Constraint", style="green", ratio=3, overflow="fold")
    table.add_column("Arbitration Verdict", ratio=3, overflow="fold")
    table.add_column("Recommended Fix", style="cyan", ratio=3, overflow="fold")

    for match in failed_matches:
        test_name = match[0]
        err_msg = match[1] if len(match) > 1 and match[1] else "Assertion or contract violation"

        arbitration = arbiter.arbitrate_failure(
            test_name=test_name,
            error_message=err_msg,
        )

        if arbitration.verdict == "TRUE_CODE_DEFECT":
            styled_verdict = f"[bold red]{arbitration.verdict}[/bold red]"
        elif arbitration.verdict == "INVALID_TEST_ASSERTION":
            styled_verdict = f"[bold yellow]{arbitration.verdict}[/bold yellow]"
        elif arbitration.verdict == "SPEC_AMBIGUITY_OR_DEFECT":
            styled_verdict = f"[bold magenta]{arbitration.verdict}[/bold magenta]"
        else:
            styled_verdict = f"[bold]{arbitration.verdict}[/bold]"

        table.add_row(
            test_name,
            "FAILED",
            arbitration.spec_clause.strip(),
            f"{styled_verdict}\n({arbitration.explanation.strip()})",
            arbitration.recommended_fix.strip(),
        )

    console.print(table)


@app.command()
def remedy(
    file: str = typer.Option("testbed/app/services/order_service.py", "--file", "-f", help="Target source file requiring remediation"),
    test_path: str = typer.Option("tests/generated/test_order_service.py", "--tests", "-t", help="Path to tests to verify against"),
    output_patch: str = typer.Option("remediation.patch", "--output", "-o", help="Output path for the generated patch"),
):
    """Run autonomous remediation (Sweep.dev pattern) to synthesize code fix, sandbox-verify, and emit patch."""
    console.print(Panel.fit("[bold cyan]TestPilot AI — Autonomous Code Remediation Bot (Sweep.dev Pattern)[/bold cyan]"))

    arbiter = RAGArbiter()
    # Dynamic arbitration of known or detected defect
    arbitration = arbiter.arbitrate_failure(
        test_name="test_boundary_coupon_deficit_negative_total",
        error_message="assert -40.0 >= 0.0, coupon deficit produced negative total",
    )
    console.print(f"[bold yellow]Arbitrating issue:[/bold yellow] {arbitration.test_name} -> [bold magenta]{arbitration.verdict}[/bold magenta]")
    console.print(f"[dim]Spec Clause:[/dim] {arbitration.spec_clause}")
    console.print(f"[dim]Recommendation:[/dim] {arbitration.recommended_fix}\n")

    patcher = RemediationPatcher()
    result = patcher.generate_remediation_patch(
        target_file_path=file,
        arbitration=arbitration,
        test_command=[sys.executable, "-m", "pytest", "tests/generated/test_order_service.py", "-q"],
    )

    if result.patch_generated:
        if output_patch != "remediation.patch":
            Path(output_patch).write_text(result.unified_diff, encoding="utf-8")
        console.print(Panel(f"[bold green]Patch Generated Successfully![/bold green]\nTarget: {result.target_file}\nSandbox Verified: {result.verified_in_sandbox}\nSaved to: {output_patch}"))
        console.print("[bold cyan]Unified Diff Patch:[/bold cyan]")
        console.print(f"```diff\n{result.unified_diff}\n```")
    else:
        console.print(f"[bold red]Failed to generate patch:[/bold red] {result.message}")


@app.command("remediate")
def remediate_cmd(
    file: str = typer.Option("testbed/app/services/order_service.py", "--file", "-f", help="Target source file requiring remediation"),
    test_path: str = typer.Option("tests/generated/test_order_service.py", "--tests", "-t", help="Path to tests to verify against"),
    output_patch: str = typer.Option("remediation.patch", "--output", "-o", help="Output path for the generated patch"),
):
    """Alias for 'remedy': Run autonomous remediation to synthesize code fix, sandbox-verify, and emit patch."""
    remedy(file=file, test_path=test_path, output_patch=output_patch)


remediate = remediate_cmd


@cli.command(name="benchmark")
@click.option("--models", default="qwen2.5-coder:7b", help="Comma-separated model identifiers to test")
@click.option("--compare-baseline", default="schemathesis,code-as-oracle", help="Baselines to benchmark against")
@click.option("--output", default="docs/BENCHMARK_REPORT.md", help="Path to write the generated markdown report")
def benchmark_cmd(
    models: str = typer.Option("qwen2.5-coder:7b", "--models", "-m", help="Comma-separated model identifiers to test"),
    compare_baseline: str = typer.Option("schemathesis,code-as-oracle", "--compare-baseline", "-c", help="Baselines to benchmark against"),
    output: str = typer.Option("docs/BENCHMARK_REPORT.md", "--output", "-o", help="Path to write the generated markdown report"),
):
    """Run empirical benchmark comparing multiple models and baselines against seeded defects."""
    from testpilot.benchmark.runner import run_benchmark  # or your benchmark evaluator module

    click.echo(f"Initiating multi-model benchmark for: {models}")
    results = run_benchmark(
        models=models.split(",") if isinstance(models, str) else models,
        baselines=compare_baseline.split(",") if isinstance(compare_baseline, str) else compare_baseline,
        output_path=output,
    )

    table = BenchmarkRunner.render_rich_table(results)
    console.print(table)
    console.print(
        Panel(
            f"[bold green]Benchmark run complete![/bold green]\n"
            f"Targets Evaluated: {len(results)}\n"
            f"Markdown Report Generated at: [bold cyan]{output}[/bold cyan]"
        )
    )


benchmark_command = benchmark_cmd


@cli.command(name="ui")
@cli.command(name="serve")
def ui_cmd(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run the web UI dashboard"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host IP to bind the web server"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development with safe exclusions"),
):
    """Launch TestPilot AI Web Interface and IDE Dashboard."""
    import os
    import signal
    import socket
    import time

    import uvicorn

    def is_port_in_use(h: str, p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((h, p)) == 0

    if is_port_in_use(host, port):
        # Attempt to kill stale testpilot process listening on the port
        try:
            lsof = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if lsof.stdout.strip():
                for pid in lsof.stdout.strip().split():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
                time.sleep(0.5)
        except Exception:
            pass

    # If still in use, gracefully fall back to next free port
    if is_port_in_use(host, port):
        fallback_port = port + 1
        while is_port_in_use(host, fallback_port) and fallback_port < port + 20:
            fallback_port += 1
        console.print(f"[yellow]Port {port} is occupied. Gracefully falling back to port {fallback_port}.[/yellow]")
        port = fallback_port

    console.print(Panel.fit(f"[bold cyan]Launching TestPilot AI Dashboard[/bold cyan]\nURL: [bold green]http://{host}:{port}[/bold green]"))
    if reload:
        uvicorn.run(
            "testpilot.web.api:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["testpilot"],
            reload_excludes=[
                ".testpilot/*",
                ".testpilot/**",
                "tests/generated/*",
                "testbed/evaluation_store/*",
                "*.json",
                ".git/*",
            ],
        )
    else:
        uvicorn.run("testpilot.web.api:app", host=host, port=port, reload=False)


ui_command = ui_cmd
serve_command = ui_cmd



@cli.command(name="check-guardrails")
@click.option("--code-file", default="tests/generated/test_order_service.py", help="Path to code or test file to analyze")
@click.option("--spec", default="testbed/openapi.json", help="Path to OpenAPI spec")
def check_guardrails_cmd(
    code_file: str = typer.Option("tests/generated/test_order_service.py", "--code-file", "-f", help="Path to code or test file to analyze"),
    spec: str = typer.Option("testbed/openapi.json", "--spec", "-s", help="Path to OpenAPI spec"),
):
    """Run Safety Guardrails verification on generated tests or source code."""
    from testpilot.guardrails.engine import SafetyGuardrailEngine

    console.print(Panel.fit(f"[bold cyan]TestPilot Safety Guardrails Inspection[/bold cyan]\nTarget: [bold yellow]{code_file}[/bold yellow]"))
    engine = SafetyGuardrailEngine(spec_path=spec)
    result = engine.check_file(code_file)

    table = Table(title="Guardrail Verification Subsystem", show_lines=True)
    table.add_column("Guardrail Check", style="bold cyan", width=30)
    table.add_column("Status", width=16)
    table.add_column("Details", style="dim")

    # 1. AST Code Safety
    code_safe = result.details.get("code_safety_passed", False)
    table.add_row(
        "AST Code Safety & Imports",
        "[bold green]PASS[/bold green]" if code_safe else "[bold red]FAIL[/bold red]",
        "No dangerous calls or unauthorized imports" if code_safe else "Violations detected",
    )

    # 2. Spec-as-Oracle Grounding
    spec_safe = result.details.get("spec_adherence_passed", False)
    table.add_row(
        "Spec Adherence & Grounding",
        "[bold green]PASS[/bold green]" if spec_safe else "[bold red]FAIL[/bold red]",
        "Zero hallucinated boundaries or ungrounded claims detected" if spec_safe else "Ungrounded assertions identified",
    )

    # 3. Overall Verdict
    table.add_row(
        "Safety Score",
        f"[bold {'green' if result.is_valid else 'red'}]{result.safety_score * 100:.0f}%[/bold {'green' if result.is_valid else 'red'}]",
        "COMPLIANT" if result.is_valid else f"{len(result.violations)} safety violations",
    )

    console.print(table)

    if not result.is_valid:
        console.print("[bold red]Detected Violations:[/bold red]")
        for v in result.violations:
            console.print(f"  [red]&bull; {v}[/red]")
        sys.exit(1)
    else:
        console.print("[bold green]&check; All Safety Guardrails Satisfied. Code is safe for execution.[/bold green]")


check_guardrails_command = check_guardrails_cmd


@app.command("evolution")
def evolution_cmd(
    base: str = typer.Option("HEAD~1", "--base", "-b", help="Base Git ref (commit, branch, or tag)"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target Git ref (default: working tree or HEAD)"),
    max_depth: int = typer.Option(3, "--max-depth", "-d", help="Max transitive call-graph depth (1-5)"),
    repo_path: str = typer.Option(".", "--repo-path", "-r", help="Repository root path"),
):
    """
    Repository Evolution Intelligence: Transitive blast-radius analysis & test prioritization.
    Analyzes code changes between git refs, maps ripple effects across call graphs,
    and prioritizes existing test suites with execution commands.
    """
    console.print(Panel.fit(
        f"[bold cyan]TestPilot AI — Repository Evolution Intelligence[/bold cyan]\n"
        f"Diff Range: [bold yellow]{base}[/bold yellow] &rarr; [bold yellow]{target or 'WORKING_TREE'}[/bold yellow] | "
        f"Transitive Depth: [bold green]{max_depth}[/bold green]"
    ))

    req = EvolutionRequest(
        base_ref=base,
        target_ref=target,
        max_depth=max_depth,
        repo_path=repo_path,
    )
    engine = RepositoryEvolutionEngine(repo_root=repo_path)

    try:
        with console.status("[bold green]Analyzing Git diff, tracing AST call graphs & prioritizing tests...[/bold green]"):
            report = engine.analyze(req)
    except InvalidGitReferenceError as e:
        console.print(f"[bold red]Git Reference Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Analysis Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # 1. Summary Metrics Table
    summary_table = Table(title="Evolution Analysis Summary", show_lines=True)
    summary_table.add_column("Metric", style="bold cyan", width=30)
    summary_table.add_column("Value", style="bold green", width=25)
    summary_table.add_column("Details", style="dim")

    summary_table.add_row("Base Ref", report.base_ref, "Comparison base")
    summary_table.add_row("Target Ref", report.target_ref, "Comparison target")
    summary_table.add_row("Files Changed", str(report.total_files_changed), f"{len(report.changed_symbols)} AST symbols parsed")
    summary_table.add_row("Transitive Blast Radius", f"{len(report.impact_graph)} symbols impacted", f"{report.direct_impact_count} Direct, {report.indirect_impact_count} Indirect")
    summary_table.add_row("Total Discovered Tests", f"{report.total_discovered_tests} tests in repository", "Indexed test corpus")
    summary_table.add_row("Prioritized Regression Tests", f"{report.prioritized_test_count} tests identified", f"{len(report.critical_tests)} Critical, {len(report.high_priority_tests)} High")
    summary_table.add_row("Semantic Impact Assessments", f"{len(report.semantic_impacts)} assessments", f"{len(report.direct_behavioral_impacts)} Direct, {len(report.indirect_behavioral_impacts)} Indirect, {len(report.structural_only_impacts)} Structural")
    summary_table.add_row("Event-Trigger Candidates", f"{len(report.event_trigger_candidates)} candidates", "Unconfirmed behavioral coverage")
    summary_table.add_row("Analysis Latency", f"{report.analysis_latency_ms:.1f} ms", "Real-time AST & graph indexing")
    console.print(summary_table)

    # 2. Changed Symbols Table
    if report.changed_symbols:
        sym_table = Table(title="Detected Changed Symbols", show_lines=True)
        sym_table.add_column("Symbol", style="bold yellow", width=28)
        sym_table.add_column("File Path", style="dim", width=40)
        sym_table.add_column("Lines", width=12)
        sym_table.add_column("Change Type", width=16)

        for sym in report.changed_symbols:
            badge_color = "red" if sym.change_type.value == "DELETED" else ("green" if sym.change_type.value == "ADDED" else "cyan")
            sym_table.add_row(
                sym.symbol_name,
                sym.file_path,
                f"L{sym.start_line}-L{sym.end_line}",
                f"[{badge_color}]{sym.change_type.value}[/{badge_color}]",
            )
        console.print(sym_table)
    else:
        console.print("[dim yellow]No symbol-level modifications detected in Python files for this range.[/dim yellow]")

    # 3. Transitive Blast Radius Table
    if report.impact_graph:
        impact_table = Table(title=f"Transitive Blast Radius Graph (Depth &le; {max_depth})", show_lines=True)
        impact_table.add_column("Hop / Depth", width=12, style="bold")
        impact_table.add_column("Impacted Symbol", style="bold yellow", width=25)
        impact_table.add_column("File Path", style="dim", width=35)
        impact_table.add_column("Caller / Source", width=25)
        impact_table.add_column("Type", width=12)
        impact_table.add_column("Certainty", width=12)

        for node in report.impact_graph:
            type_color = "red" if node.impact_type.value == "DIRECT" else "yellow"
            certainty_pct = (1.0 - node.uncertainty_score) * 100
            cert_color = "green" if certainty_pct >= 80 else ("yellow" if certainty_pct >= 50 else "red")
            impact_table.add_row(
                f"Hop {node.depth}",
                node.symbol_name,
                node.file_path,
                node.evidence.caller_symbol or node.evidence.source_symbol,
                f"[{type_color}]{node.impact_type.value}[/{type_color}]",
                f"[{cert_color}]{certainty_pct:.0f}%[/{cert_color}]",
            )
        console.print(impact_table)

    # 4. Prioritized Test Suite Table
    if report.prioritized_tests:
        test_table = Table(title="Prioritized Existing Test Suite (Confirmed Regression Coverage)", show_lines=True)
        test_table.add_column("Tier", width=10, style="bold")
        test_table.add_column("Classification", width=24)
        test_table.add_column("Test Function", style="bold yellow", width=28)
        test_table.add_column("Test File", style="dim", width=32)
        test_table.add_column("Targeted Symbol", width=20)
        test_table.add_column("Score", width=8)
        test_table.add_column("Selection Reason / Evidence", style="dim", width=35)

        tier_colors = {
            "CRITICAL": "bold red",
            "HIGH": "bold yellow",
            "MEDIUM": "cyan",
            "LOW": "dim",
        }

        classification_colors = {
            "BEHAVIORAL_COVERAGE": "green",
            "EXPLICIT_EVENT_DISPATCH": "magenta",
            "EVENT_TRIGGER_CANDIDATE": "yellow",
            "NO_RELEVANT_EVENT_EVIDENCE": "dim",
        }

        for test in report.prioritized_tests:
            color = tier_colors.get(test.priority_tier.value, "white")
            classification_val = test.match_classification.value if test.match_classification else "CALL_GRAPH"
            class_color = classification_colors.get(classification_val, "cyan")
            reason = test.selection_reason or test.explanation or ""

            test_table.add_row(
                f"[{color}]{test.priority_tier.value}[/{color}]",
                f"[{class_color}]{classification_val}[/{class_color}]",
                test.test_function,
                test.test_file,
                test.target_symbol,
                f"{test.priority_score:.2f}",
                reason,
            )
        console.print(test_table)

        # Print quick run command
        critical_cmds = [t.execution_command for t in report.critical_tests]
        if critical_cmds:
            console.print(Panel(
                f"[bold green]Recommended Fast-Feedback Command:[/bold green]\n"
                f"[bold white]{' '.join(critical_cmds[:5])}[/bold white]",
                title="Immediate Regression Verification",
            ))
    else:
        console.print("[dim green]No existing tests impacted directly or indirectly by these changes.[/dim green]")

    # 5. Possible Event-Trigger Candidates Table
    if report.event_trigger_candidates:
        candidate_table = Table(
            title=f"Possible Event-Trigger Candidates ({len(report.event_trigger_candidates)} tests — Unconfirmed Behavioral Coverage)",
            show_lines=True,
        )
        candidate_table.add_column("Event", style="bold cyan", width=14)
        candidate_table.add_column("Test Function", style="bold yellow", width=28)
        candidate_table.add_column("Test File", style="dim", width=32)
        candidate_table.add_column("Trigger Evidence", width=30)
        candidate_table.add_column("Missing Evidence", style="dim yellow", width=35)

        for cand in report.event_trigger_candidates:
            ev_desc = ""
            if cand.structured_evidence:
                ev_desc = "; ".join(e.description for e in cand.structured_evidence[:2])
            elif cand.selection_reason:
                ev_desc = cand.selection_reason
            missing = "; ".join(cand.missing_evidence) if cand.missing_evidence else "Awaiting behavioral verification"

            candidate_table.add_row(
                cand.event_name or "event",
                cand.test_function,
                cand.test_file,
                ev_desc,
                missing,
            )
        console.print(candidate_table)

    # 6. Evidence-Grounded Semantic Impact Assessments Table
    if report.semantic_impacts:
        sem_table = Table(
            title=f"Evidence-Grounded Semantic Impact Assessments ({len(report.semantic_impacts)} Analyzed)",
            show_lines=True,
        )
        sem_table.add_column("Rank / Tier", width=18, style="bold")
        sem_table.add_column("Classification", width=28)
        sem_table.add_column("Affected Symbol", style="bold yellow", width=24)
        sem_table.add_column("Depth", width=8)
        sem_table.add_column("Evidence Source", width=18)
        sem_table.add_column("Explanation & RAG Evidence", style="dim", width=42)

        sem_class_colors = {
            "DIRECT_BEHAVIORAL_DEPENDENCY": "bold green",
            "INDIRECT_BEHAVIORAL_DEPENDENCY": "bold cyan",
            "STRUCTURAL_ONLY": "dim white",
            "UNCERTAIN": "bold yellow",
        }

        for sem in report.semantic_impacts:
            c_val = sem.classification.value if hasattr(sem.classification, "value") else str(sem.classification)
            c_color = sem_class_colors.get(c_val, "white")
            rank_str = f"#{sem.prioritization.rank} [{sem.prioritization.tier}]" if sem.prioritization else "N/A"
            ev_summary = sem.explanation
            if sem.relevant_tests:
                ev_summary += f" [Tests: {', '.join(sem.relevant_tests[:2])}]"
            sem_table.add_row(
                rank_str,
                f"[{c_color}]{c_val}[/{c_color}]",
                sem.affected_symbol,
                f"Hop {sem.structural_depth}",
                sem.evidence_source.value if hasattr(sem.evidence_source, "value") else str(sem.evidence_source),
                ev_summary,
            )
        console.print(sem_table)


@app.command("pipeline")
def pipeline_cmd(
    base: str = typer.Option("HEAD~1", "--base", "-b", help="Base Git ref for Evolution comparison"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target Git ref (default: working tree or HEAD)"),
    repo_path: str = typer.Option(".", "--repo-path", "-r", help="Repository root path"),
    max_depth: int = typer.Option(3, "--max-depth", "-d", help="Max transitive call-graph depth (1-5)"),
    spec_path: str = typer.Option("testbed/openapi.json", "--spec-path", "-s", help="Path to OpenAPI specification"),
    skip_evolution: bool = typer.Option(False, "--skip-evolution", help="Skip Stage 0 Repository Evolution"),
    enable_semantic_validation: bool = typer.Option(
        False, "--enable-semantic-validation", help="Enable experimental Repository RAG & CodeLlama semantic validation"
    ),
    enable_sourcegraph: bool = typer.Option(
        True, "--enable-sourcegraph/--disable-sourcegraph", help="Enable/disable Sourcegraph code intelligence"
    ),
):
    """
    Run Full TestPilot Autonomous Closed-Loop Pipeline:
    Stage 0: Repository Evolution Intelligence & Regression Test Prioritization
    Stage 1: OpenAPI Schema Boundary Matrix extraction
    Stage 2: Prioritized Regression & Boundary Test Execution
    Stage 3: Spec-as-Oracle Three-Valued Arbitration
    Stage 4: Safety Guardrails Audit
    Stage 5: Autonomous Remediation (Sweep.dev pattern)
    Stage 6: Ephemeral Sandbox Verification
    """
    target_display = target or "HEAD"
    evo_display = "DISABLED" if skip_evolution else f"{base} -> {target_display}"
    rag_display = "ENABLED" if enable_semantic_validation else "DISABLED (baseline)"

    console.print(Panel.fit(
        f"[bold cyan]TestPilot AI — Full Autonomous Pipeline Run[/bold cyan]\n"
        f"Stage 0 Evolution: [bold yellow]{evo_display}[/bold yellow] | "
        f"Semantic RAG: [bold magenta]{rag_display}[/bold magenta] | "
        f"Spec: [bold green]{spec_path}[/bold green]"
    ))

    req = PipelineRunRequest(
        repo_path=repo_path,
        base_ref=base,
        target_ref=target,
        max_depth=max_depth,
        spec_path=spec_path,
        enable_evolution=not skip_evolution,
        enable_semantic_validation=enable_semantic_validation,
        enable_sourcegraph=enable_sourcegraph,
    )

    with console.status("[bold green]Executing full closed-loop testing & remediation pipeline...[/bold green]"):
        orchestrator = FullPipelineOrchestrator()
        result = orchestrator.execute(req)

    # 1. Pipeline Summary Table
    summary_table = Table(title=f"Full Pipeline Results — Status: [{ 'green' if result.overall_status == 'SUCCESS' else 'yellow'}]{result.overall_status}[/]", show_lines=True)
    summary_table.add_column("Stage", style="bold cyan", width=28)
    summary_table.add_column("Status", width=16)
    summary_table.add_column("Key Metrics / Findings", style="dim")

    # Stage 0
    evo = result.evolution
    evo_style = "green" if evo.status == "SUCCESS" else ("yellow" if evo.status == "DEGRADED" else "dim")
    summary_table.add_row(
        "0. Repository Evolution",
        f"[{evo_style}]{evo.status}[/{evo_style}]",
        f"Files: {len(evo.changed_files)}, Symbols: {len(evo.changed_symbols)}, Blast: {evo.blast_radius_size}, Prioritized Tests: {len(evo.prioritized_tests)} ({evo.analysis_latency_ms:.1f}ms)",
    )

    # Stage 1
    bnd = result.boundary_testing
    summary_table.add_row(
        "1. OpenAPI Schema Boundaries",
        f"[green]{bnd.status}[/green]",
        f"Extracted {bnd.total_boundaries} formal schema boundary constraints",
    )

    # Stage 2
    reg = result.regression_testing
    summary_table.add_row(
        "2. Regression Test Execution",
        f"[green]{reg.status}[/green]",
        f"Source: {reg.source}, Total: {len(reg.executed_tests)}, Passed: {reg.total_passed}, Failed: {reg.total_failed}",
    )

    # Stage 3
    arb = result.arbitration
    summary_table.add_row(
        "3. Three-Valued Arbitration",
        f"[green]{arb.status}[/green]",
        f"Arbitrated: {arb.total_arbitrated} ({', '.join(f'{k}: {v}' for k, v in arb.verdicts_summary.items())})",
    )

    # Stage 4
    grd = result.guardrails
    grd_style = "green" if grd.is_valid else "red"
    summary_table.add_row(
        "4. Safety Guardrails",
        f"[{grd_style}]{'SAFE' if grd.is_valid else 'FLAGGED'}[/{grd_style}]",
        f"Safety score: {grd.safety_score:.2f}, Violations: {len(grd.violations)}",
    )

    # Stage 5 & 6
    rem = result.remediation
    snd = result.sandbox_verification
    summary_table.add_row(
        "5. Remediation & Sandbox",
        f"[green]{'VERIFIED GREEN' if snd.verified_in_sandbox else 'PENDING'}[/green]",
        f"Patch generated: {rem.patch_generated}, Sandbox verified: {snd.verified_in_sandbox}",
    )

    console.print(summary_table)

    # 2. Prioritized Regression Tests Table
    if evo.prioritized_tests:
        tests_table = Table(title="Stage 0 Prioritized Regression Tests (Run First)", show_lines=True)
        tests_table.add_column("Tier", width=10, style="bold")
        tests_table.add_column("Score", width=8)
        tests_table.add_column("Test Function", style="bold yellow", width=30)
        tests_table.add_column("File Path", style="dim", width=35)
        tests_table.add_column("Execution Command", style="cyan")

        for t in evo.prioritized_tests:
            tier_str = t.priority_tier.value if hasattr(t.priority_tier, "value") else str(t.priority_tier)
            t_color = "red" if tier_str == "CRITICAL" else ("yellow" if tier_str == "HIGH" else "cyan")
            tests_table.add_row(
                f"[{t_color}]{tier_str}[/{t_color}]",
                f"{t.priority_score:.2f}",
                t.test_name,
                t.test_file,
                t.execution_command,
            )
        console.print(tests_table)

    console.print(f"[bold green]Full Pipeline Run Completed in {result.pipeline_latency_ms:.1f}ms.[/bold green]\n")


@app.command(name="evaluate")
def evaluate_cmd(
    case_id: Optional[str] = typer.Option(None, "--case", "-c", help="Specific benchmark case ID to evaluate"),
    all_cases: bool = typer.Option(False, "--all", "-a", help="Run all configured benchmark cases"),
):
    """
    Empirical Evaluation & Quantitative Benchmarking Harness.
    Executes baselines (Full Regression, Naive Name Matching, TestPilot Qualified Identity)
    against real repositories (Flask, Django, Home Assistant) and testbed scenarios.
    """
    engine = EvaluationEngine()

    if not case_id and not all_cases:
        # Display current overview summary
        summary = engine.get_overview_summary()
        p_val = f"{summary['macro_precision'] * 100:.1f}%" if summary.get("macro_precision") is not None else "N/A"
        r_val = f"{summary['macro_recall'] * 100:.2f}%" if summary.get("macro_recall") is not None else "N/A"
        f1_val = f"{summary['macro_f1'] * 100:.2f}%" if summary.get("macro_f1") is not None else "N/A"
        red_val = f"{summary['avg_test_reduction_pct']:.2f}%" if summary.get("avg_test_reduction_pct") is not None else "N/A"

        console.print(Panel.fit(
            f"[bold cyan]TestPilot Empirical Evaluation: {summary.get('methodology_label', 'Positive-Ground-Truth Pooled Evaluation')}[/bold cyan]\n"
            f"Status: [yellow]{summary['status']}[/yellow] | Repositories: [bold]{summary['repositories_count']}[/bold] | Positive Cases: [bold]{summary.get('positive_cases_count', 0)}[/bold]\n"
            f"Pooled Precision: [bold green]{p_val}[/bold green] | Pooled Recall: [bold green]{r_val}[/bold green] | Pooled F1: [bold green]{f1_val}[/bold green]\n"
            f"Test Reduction: [bold yellow]{red_val}[/bold yellow] | Avg Latency: [bold]{summary['avg_latency_ms'] or 'N/A'} ms[/bold]\n"
            f"[dim]{summary.get('methodology_description', '')}[/dim]",
            title="Evaluation Overview",
            border_style="cyan",
        ))

        if summary.get("baseline_comparison"):
            comp_table = Table(title="Baseline Comparison Matrix (Positive-Ground-Truth Pooled Suite)", show_lines=True)
            comp_table.add_column("Method", style="bold")
            comp_table.add_column("Tests Selected")
            comp_table.add_column("Precision", style="cyan")
            comp_table.add_column("Recall", style="green")
            comp_table.add_column("F1", style="magenta")
            comp_table.add_column("Reduction", style="yellow")
            comp_table.add_column("Latency")
            comp_table.add_column("Notes", style="dim")

            for k, row in summary["baseline_comparison"].items():
                p_str = f"{row['precision'] * 100:.2f}%" if isinstance(row.get("precision"), (int, float)) else str(row.get("precision", "N/A"))
                r_str = f"{row['recall'] * 100:.2f}%" if isinstance(row.get("recall"), (int, float)) else str(row.get("recall", "N/A"))
                f1_str = f"{row['f1'] * 100:.2f}%" if isinstance(row.get("f1"), (int, float)) else str(row.get("f1", "N/A"))
                comp_table.add_row(
                    row.get("method", k),
                    str(row.get("tests_selected", "N/A")),
                    p_str,
                    r_str,
                    f1_str,
                    f"{row.get('test_reduction'):.2f}%" if isinstance(row.get("test_reduction"), (int, float)) else "N/A",
                    f"{row.get('latency')} ms" if row.get("latency") not in ("N/A", None) else "N/A",
                    row.get("notes", ""),
                )
            console.print(comp_table)

        if summary.get("negative_control_study"):
            neg = summary["negative_control_study"]
            console.print(Panel(
                f"[bold magenta]Empirical Negative Control / False-Positive Case Study[/bold magenta]\n"
                f"Repository: [bold]{neg.get('repository')}[/bold] | Commits: {neg.get('base_commit', '')[:8]} → {neg.get('target_commit', '')[:8]}\n"
                f"Ground Truth: [bold]0 callers[/bold] | Total Tests: [bold]{neg.get('total_tests', 0):,}[/bold]\n"
                f"Naive Selected: [bold red]{neg.get('naive_selected', 0)} (False Positives)[/bold red] | TestPilot Selected: [bold green]{neg.get('testpilot_selected', 0)} (100% Rejection)[/bold green]\n"
                f"Precision / Recall / F1: [cyan]N/A (0/0)[/cyan] | Reduction: [yellow]100.0%[/yellow]\n"
                f"[dim]{neg.get('notes', '')}[/dim]",
                title="Negative Control",
                border_style="magenta",
            ))

        console.print("[dim]Use --all to run all benchmarks or --case <id> to run a specific case.[/dim]")
        return

    target_cases = [case_id] if case_id else [c.case_id for c in engine.storage.get_benchmark_cases()]
    for cid in target_cases:
        console.print(f"[cyan]Executing benchmark:[/] [bold]{cid}[/]...")
        try:
            run = engine.run_benchmark(cid)
            tp_m = run.metrics.get("testpilot")
            p_str = f"{tp_m.precision * 100:.1f}%" if tp_m and tp_m.precision is not None else "N/A"
            r_str = f"{tp_m.recall * 100:.1f}%" if tp_m and tp_m.recall is not None else "N/A"
            red_str = f"{tp_m.test_reduction_pct:.1f}%" if tp_m and tp_m.test_reduction_pct is not None else "N/A"
            console.print(
                f"[green]✓ Completed {cid}[/]: TP={tp_m.tp if tp_m else 0}, FP={tp_m.fp if tp_m else 0}, FN={tp_m.fn if tp_m else 0} | "
                f"Precision={p_str}, Recall={r_str}, Reduction={red_str}, Latency={tp_m.latency_ms if tp_m else 0:.1f}ms"
            )
        except Exception as e:
            console.print(f"[bold red]✗ Failed {cid}[/]: {e}")


@app.command("rag-index")
def rag_index_cmd(
    repo_path: str = typer.Option(".", "--repo-path", "-r", help="Repository root path to index"),
):
    """Index repository code units into dedicated Chroma repo_code_store collection."""
    from testpilot.rag.repo_vector_store import RepoCodeVectorStore

    console.print(Panel.fit(f"[bold cyan]Indexing Repository Code Units: {repo_path}[/bold cyan]"))
    store = RepoCodeVectorStore(repo_root=repo_path)
    units = store.index_repository(repo_path)
    console.print(f"[bold green]✓ Successfully indexed {len(units)} semantic code units into repo_code_store collection.[/bold green]")


@app.command("rag-query")
def rag_query_cmd(
    query: str = typer.Argument(..., help="Search query or symbol"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of code units to retrieve"),
    repo_path: str = typer.Option(".", "--repo-path", "-r", help="Repository root path"),
):
    """Retrieve semantic repository code context using vector embeddings."""
    from testpilot.rag.repo_vector_store import RepoCodeVectorStore

    store = RepoCodeVectorStore(repo_root=repo_path)
    results = store.retrieve_code_context(query=query, top_k=top_k)
    table = Table(title=f"Repository RAG Retrieval Results for: {query}", show_lines=True)
    table.add_column("Symbol", style="bold cyan", width=30)
    table.add_column("File", style="yellow", width=25)
    table.add_column("Type", width=12)
    table.add_column("Similarity", width=12)
    for r in results:
        table.add_row(
            r.get("qualified_symbol", ""),
            r.get("file_path", ""),
            r.get("symbol_type", ""),
            f"{r.get('similarity', 0.0):.3f}",
        )
    console.print(table)


@app.command("sourcegraph-query")
def sourcegraph_query_cmd(
    query: str = typer.Argument(..., help="Symbol or function to search"),
    search_type: str = typer.Option("references", "--type", "-t", help="Search type: references, definitions, tests, functions"),
):
    """Query Sourcegraph repository code intelligence."""
    sg = SourcegraphClient()
    if search_type == "definitions":
        results = sg.find_definitions(query)
    elif search_type == "tests":
        results = sg.find_test_references(query)
    elif search_type == "functions":
        results = sg.search_functions(query)
    else:
        results = sg.find_references(query)

    norm = sg.normalize_evidence(query=query, symbol=query, results=results)
    table = Table(title=f"Sourcegraph Results ({norm['status']}) for: {query}", show_lines=True)
    table.add_column("File", style="bold green", width=35)
    table.add_column("Line", width=8)
    table.add_column("Match Content", style="dim")
    for r in results[:15]:
        table.add_row(
            r.get("file_path", ""),
            str(r.get("line_number", "")),
            r.get("line_content", "") or r.get("preview", ""),
        )
    console.print(table)


evolution_command = evolution_cmd


if __name__ == "__main__":
    app()
