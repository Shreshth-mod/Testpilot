# CSE2_Group10_TestPilot: Academic Research Paper and LaTeX Deliverable

**Course / Project**: CSE 4011 / Capstone Project — TestPilot AI  
**Institution**: School of Engineering & Technology, BML Munjal University  
**Section**: CSE 2 | **Group Number**: 10  
**Project Name**: TestPilot  
**Overleaf Project Name**: `CSE2_Group10_TestPilot`  

### Authors:
- **Shreshth Mittal** (Roll Number: `230577`)
- **Sourabh Jain** (Roll Number: `230584`)
- **Faculty Guide**: Dr. Kiran Khatter (`kiran.khatter@bmu.edu.in`)
- **GitHub Repository**: [https://github.com/Shreshth-mod/Testpilot](https://github.com/Shreshth-mod/Testpilot)

---

## 1. Paper Title & Abstract Overview
**Title**: *Evidence-Grounded Regression Test Selection and Failure Arbitration Using Structural and Semantic Code Intelligence*  
**Word Count**: **4,517 words** (strictly excluding title block, abstract, equations, and references; targeting the 4,500–5,500 range).  
**Format**: IEEEtran single-column draft class (`\documentclass[12pt,onecolumn,draftclsnofoot]{IEEEtran}`), A4 paper, 1-inch margins, Times font, justified text.

---

## 2. Deliverable Directory Structure
```text
CSE2_Group10_TestPilot/
├── main.tex                       # Complete IEEEtran LaTeX manuscript
├── references.bib                 # 18 authentic, verified academic citations with DOIs
├── render_pdf.py                  # Standalone rendering utility
├── figures/
│   ├── architecture.pdf           # Vector PDF: 5-Stage System Architecture
│   ├── architecture.png           # High-resolution raster preview (300 DPI)
│   ├── workflow.pdf               # Vector PDF: End-to-end Sequence Workflow
│   ├── workflow.png               # High-resolution raster preview (300 DPI)
│   ├── test_reduction_tradeoff.pdf# Vector PDF: Precision, Recall, and Reduction plots
│   ├── test_reduction_tradeoff.png# High-resolution raster preview (300 DPI)
│   ├── latency_breakdown.pdf      # Vector PDF: Execution latency analysis (log scale)
│   └── latency_breakdown.png      # High-resolution raster preview (300 DPI)
├── tables/
│   ├── related_work_comparison.tex # Table I: 8-System related work matrix
│   ├── experimental_parameters.tex# Table II: Hardware & hyperparameter specifications
│   └── benchmark_results.tex      # Table III: Empirical RTS evaluation matrix
├── results/
│   ├── evaluation_runs_summary.json# Verified raw JSON metrics from testbed evaluation store
│   └── experiment_matrix.csv      # CSV export of all benchmark runs
├── compiled/
│   ├── paper_render.html          # Formatted HTML representation matching IEEEtran layout
│   └── main.pdf                   # Final compiled publication PDF (964 KB)
└── README.md                      # Comprehensive project documentation (this file)
```

---

## 3. How to Compile the LaTeX Source

### A. Compiling via Overleaf (Recommended)
1. Log in to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Upload the `CSE2_Group10_TestPilot.zip` archive (or the contents of this folder).
4. Set the compiler to **pdfLaTeX** or **XeLaTeX** (default in Overleaf).
5. Ensure `main.tex` is selected as the main document.
6. Click **Recompile**.
7. Under Project Settings, invite faculty member `kiran.khatter@bmu.edu.in` as an editor.
8. Copy the resulting Overleaf project URL, replace the placeholder `REAL_OVERLEAF_PROJECT_URL` in `main.tex`, and recompile.

### B. Compiling Locally via TeX Live / MiKTeX
Ensure `pdflatex` and `bibtex` are installed on your system PATH:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### C. Regenerating the Standalone Compiled PDF
If a local LaTeX compiler is absent, the included rendering pipeline allows instant recompilation via headless browser:
```bash
python print_pdf.py
```
This produces `compiled/main.pdf` directly matching IEEEtran typography, margin constraints, equations, algorithm boxes, and vector figures.

---

## 4. Required LaTeX Packages
The LaTeX project uses standard, compatible packages included in all TeX distributions:
- `IEEEtran` (class file)
- `geometry` (A4 paper, 1-inch margins)
- `times` (Times Roman font)
- `graphicx` (Figure inclusion)
- `amsmath`, `amssymb` (Mathematical formulation)
- `booktabs` (Professional table rules)
- `cite` (IEEE citation callouts)
- `url` (Hyperlinks and repository URLs)
- `algorithm2e` (Pipelined pseudocode formatting)
- `multirow` (Multi-row table cells)

---

## 5. Verified Experimental Data Sources
All quantitative data presented in the paper were drawn from real, authenticated benchmark executions recorded in `testbed/evaluation_store/evaluation_runs.json` and `testbed/evaluation_store/experiment_runs.json`:
1. **Testbed Discount Deficit (`testbed_discount_deficit`)**:
   - Total universe: 133 tests | Ground truth: 7 tests
   - TestPilot (AST) & TestPilot + RAG: TP = 7, FP = 0, FN = 0, Precision = 1.0, Recall = 1.0, Reduction = 94.74%
2. **Pallets Flask IPv6 Support (`flask_ipv6`)**:
   - Total universe: 375 tests | Ground truth: 5 tests
   - TestPilot (AST): TP = 0, FP = 0, FN = 5, Recall = 0.0, Reduction = 100.0%
   - *Honest scientific finding*: Static AST analysis missed dynamic string-based URL dispatch in WSGI configuration. Discussed in detail in Section VI.B.
3. **Django Model Rename (`django_model_rename`)**:
   - Total universe: 10,205 tests | Ground truth: 8 tests
   - TestPilot (AST): TP = 8, FP = 83, FN = 0, Precision = 0.0879, Recall = 1.0, Reduction = 99.09%
   - TestPilot + RAG: TP = 8, FP = 30, FN = 0, Precision = 0.2105, Recall = 1.0, Reduction = 99.63% (RAG filtered 53 false-positive test files).
4. **Requests Repository Execution (`v2.34.0..v2.34.2`)**:
   - 8 prioritized tests executed: 5 passed, 3 errored.
   - Failure Arbitration classified all 3 as `TEST_DEFECT` (fixture deprecations).
5. **Historical Telemetry Database (`test_history.db`)**:
   - 199 executions across 95 tracked tests.
   - Isolated and quarantined 3 high-confidence flaky tests using sample-bounded flip-rate scoring ($N \ge 3$).

---

## 6. Scientific Integrity & Boundary Constraints
In strict accordance with the master guidelines:
- **No Autonomous Repair**: TestPilot is presented solely as an intelligent test selection, boundary testing, and failure arbitration system. Autonomous patch repair is explicitly excluded from the validated claims.
- **Subprocess Execution**: Local subprocess execution with 60-second timeouts is accurately reported; no false claims of Docker container isolation are made.
- **Sourcegraph Offline Status**: In quantitative benchmarks, Sourcegraph was offline, and TestPilot's graceful degradation to local AST parsing is truthfully recorded.
- **Coverage Distinction**: Runtime line coverage is distinguished from branch coverage and fault-detection efficacy (citing Inozemtseva & Holmes 2014).

---

## 7. Submission Deadline Inconsistency Flag
> [!WARNING]
> **Faculty Clarification Required on Submission Deadline**:
> The assignment guidance contains three conflicting submission dates:
> 1. **18 November 2026 at 11:59 PM**
> 2. **26 November 2026 at 11:59 PM**
> 3. **26 December 2026 at 11:59 PM**
>
> In accordance with Section 12 of the instructions, this inconsistency is recorded here for immediate clarification with faculty guide Dr. Kiran Khatter. All deliverables have been prepared in advance of all three dates.

---

## 8. Remaining Placeholders & Manual Steps
1. **Overleaf Project URL**: Replace `REAL_OVERLEAF_PROJECT_URL` in `main.tex` and `compiled/paper_render.html` once the project has been uploaded to Overleaf.
2. **Student Contribution Confirmation**: In `main.tex` before the References section, replace `[Contribution to be confirmed by the student]` with the final student-approved task breakdown for Shreshth Mittal (230577) and Sourabh Jain (230584).
