"""Static gates for the standalone Colab workflow."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "online_sdft_bandit_demo.ipynb"


def _code_sources() -> list[str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


def test_colab_setup_removes_unused_incompatible_torchao():
    setup = next(
        source
        for source in _code_sources()
        if "Removing unused torchao" in source
    )
    assert '[sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]' in setup
    assert '"transformers==5.13.1"' in setup
    assert '"peft==0.19.1"' in setup
    assert 'if importlib.util.find_spec("torch") is None' in setup


def test_colab_setup_requires_and_reports_the_selected_gpu():
    setup = next(
        source
        for source in _code_sources()
        if "Runtime ready" in source
    )
    assert 'os.environ.get("COLAB_RELEASE_TAG")' in setup
    assert "Runtime > Change runtime type > T4 GPU" in setup
    assert "torch.cuda.get_device_name(0)" in setup


def test_colab_runner_reports_progress_and_releases_cuda_cache():
    runner = next(
        source
        for source in _code_sources()
        if "run_experiment_in_memory" in source
    )
    assert "seed {seed + 1}/{seeds}" in runner
    assert "torch.cuda.empty_cache()" in runner
    assert "N_SEEDS = 3" in runner


def test_colab_results_cell_is_self_checking():
    results = next(
        source
        for source in _code_sources()
        if "Reproduction check passed" in source
    )
    assert 'summary["Online-SDFT"]' in results
    assert 'row["online_accuracy"]["mean"] for row in baselines' in results
    assert 'row["cum_regret"]["mean"] for row in baselines' in results


def test_notebook_contains_complete_trace_backed_icl_and_rag_prompts():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    assert "Complete ICL user prompt" in markdown
    assert "Complete RAG user prompt" in markdown
    assert "P(A/B/C) = (0.5127, 0.4041, 0.0832)" in markdown
    assert "P(A/B/C) = (0.1182, 0.8136, 0.0682)" in markdown
    assert "decision 148" in markdown
