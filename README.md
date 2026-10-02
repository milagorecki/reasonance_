# Reasonance

Code for *Does Reasoning Make Language Models Agree?*

Do reasoning models agree more with each other, or does reasoning make their predictions more diverse? The paper separates two views of reasoning and measures agreement against what model accuracy alone would produce:

- **reasoning as an intervention**: the same models, individuals and prompts with reasoning disabled and enabled. Enabling reasoning *increases* agreement between models.
- **reasoning as an observed behavior**: individuals sorted by how long the models reasoned about them. Individuals who elicit longer reasoning receive *less* consistent predictions across models.

Both hold across nine models from two developers and three classification tasks (ACSIncome, ACSEmployment, SIPP), and after accounting for differences in  accuracy. Every figure of the paper is drawn in one of the notebooks, which are saved with their outputs:

- `notebooks/0-figure-1.ipynb`: Figure 1
- `notebooks/1-general-performance.ipynb`: coverage, accuracy, risk scores, effort vs. reasoning length
- `notebooks/2-reasoning-effort.ipynb`: agreement with reasoning off vs. on
- `notebooks/3-reasoning-length.ipynb`: agreement vs. realized reasoning length

The figure functions are in `reasonance/paper_figures.py`; each notebook section names the paper figure it draws.

## Installation

```bash
conda env create -f environment.yml && conda activate reason-py311
uv pip install -e ".[notebooks]"
```

`requirements/main.txt` pins the versions the figures were made with. Drawing figures needs LaTeX.

## Data

The notebooks read `results/{task}-aggregated-0-bullet-is.csv` (one row per `folktexts` benchmark
run) and the per-run predictions under `$REASONANCE_RESULTS_ROOT`, the directory that contains the
benchmark `results/` folder. The XGBoost baselines are read from `results/baselines/`. To rebuild an
aggregate:

```bash
python -m reasonance.cli.aggregate_results --results-dir $REASONANCE_RESULTS_ROOT/results \
    --tasks ACSIncome --save-path results/ACSIncome-aggregated-0-bullet-is.csv
```

## Development

```bash
uv pip install -e ".[tests]"
python -m pytest tests/ && ruff check reasonance tests notebooks && mypy reasonance tests
```
