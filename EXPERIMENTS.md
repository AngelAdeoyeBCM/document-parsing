# Running the Stage 1.1 experiments

How to build an annotated dataset, parse it under every experiment configuration, and review the results in MLflow.
The reasoning behind the experiments (what each Docling lever does, why these metrics) is in
[docling_experiment_overview.md](docling_experiment_overview.md). The experiments themselves are defined in
[configs/experiments/stage_1_1.yaml](configs/experiments/stage_1_1.yaml).

## The workflow at a glance

```
corpus (storage/) ──► inventory ──► dataset of N documents ──┬──► parse runs (GPU box) ──► score ──► MLflow
                                    (tail-truncated PDFs)    │                              ▲
                                                             └──► CVAT annotation ──► gold ─┘  (re-scored as gold grows)
```

Parsing does not need the gold. Every experiment changes one setting on the B0 baseline, so all of them can be
parsed as soon as the dataset exists. Annotation runs alongside, and each time new gold is collected, the next
`experiments log` scores every run again against it. Metrics that need no gold (garble rate, dropped-text rate, time
per page) are available straight away; the gold metrics fill in as annotation progresses.

| Id | Experiment | What changes on B0 |
|:---|:-----------|:-------------------|
| E0 | Determinism check | Nothing: a second B0 run, giving the noise floor |
| E1 | PDF backend | `pypdfium2` instead of `docling_parse` |
| E2 | Layout model | `layout_heron_101`, `layout_egret_medium`, `layout_egret_large`, `layout_egret_xlarge` (B0 is `layout_heron_default`) |
| E3 | Table structure | Table structure off |
| E4 | Reading-order separators | Separators off |
| E5 | Combine | Filled in after reviewing E1 to E4 |
| E6 | Validate | Filled in after E5, scored on the held-out documents |

Every run is also shown next to two fixed rows: **B0** ([configs/docling_b0.yaml](configs/docling_b0.yaml): Docling
defaults, OCR off) and **P0** ([configs/pypdf_default.yaml](configs/pypdf_default.yaml): plain pypdf text).

## Where things live

```
storage/                       the corpus of PDFs, and storage/inventory.db
data/annotation/               the dataset and its annotation workspace
  pdfs/<hash_id>.pdf             sampled PDFs without their disclosure tail
  documents.json                 per document: source path, kept pages, stratum, truncated file's hash
  manifest.txt                   the originals' paths: the input of every parse run
  cvat/                          CVAT tasks; exports go in cvat/cvat_annotations/zips/
  gold/                          the ground truth collected from CVAT
configs/stage_1_1/             run configs generated for each variant
outputs/docling_b0/            B0 run          (each run directory holds run.jsonl, settings.json,
outputs/pypdf_default/         P0 run           one <doc>.json per document and evaluation/)
outputs/stage_1_1/<id>_<variant>_<hash>/   variant runs
outputs/stage_1_1/logs/        one log per run config
mlflow.db, mlruns/             the MLflow store
```

`data/`, `outputs/`, `storage/` and the MLflow store are git-ignored.

## 1. Setup

Parsing runs on the Linux GPU box; annotation and review can run anywhere that has the dataset folder. Every
command below runs from the repository root.

### First time on a Linux machine

uv installs itself to `~/.local/bin`, which may not be on your `PATH`, and it needs the system certificates to
trust the internal Nexus server. Set both once, for every future shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && echo 'export UV_SYSTEM_CERTS=true' >> ~/.bashrc
```

```bash
source ~/.bashrc
```

### The project

Install the project with every extra: Docling for parsing, pyarrow for reading the gold, MLflow for tracking.

```bash
uv sync --all-extras
```

### docling-eval, for preparing and collecting annotation

`annotation prepare` and `annotation collect` run the `docling-eval` command. It is installed as a separate uv
tool, with its own environment, because its CVAT tools need Pillow below 12, and the project uses Pillow 12 or
later:

```bash
uv tool install "docling-eval[cvat_tools]"
```

> **Blocked on Nexus for now.** Nexus quarantines every Pillow below 12, so this install fails on machines that
> use it, until Pillow 11.3.0 is released from quarantine or docling-cvat-tools accepts Pillow 12. Only these two
> annotation commands need docling-eval: parsing, scoring and MLflow are unaffected, and CVAT itself runs
> elsewhere.

## 2. Build the inventory

The inventory profiles every PDF in the corpus once (pages, text-layer size, images) and runs triage, which marks the
disclosure tail of each document. The dataset is drawn from it, and scoring reads its stored pypdf text. It must be
at `storage/inventory.db`, where the parse runs and `experiments log` look for it.

```bash
uv run python -m pipeline inventory --directory storage --out storage/inventory.db --config configs/docling_standard.yaml
```

`--config` supplies the triage settings (the `triage:` block of that config). Re-running only profiles new or changed
files.

## 3. Build the dataset: N stratified, truncated documents

```bash
uv run python -m pipeline dataset --directory storage --database storage/inventory.db --number 100
```

What it does:

- **Stratified sample.** Documents are spread evenly over three axes, all measured on the pages triage keeps:
  length (1-3, 4-10, 11-30, 31+ pages), text density (under 2,000, 2,000-3,200, over 3,200 characters per page)
  and imagery (text, illustrated, image-heavy). Each non-empty combination gets an equal share of N. The log prints
  how many documents fall in each level of each axis.
- **Embedded text only.** By default it samples only documents with an embedded text layer and leaves out scans
  (`--text-quality`, `--ocr` change this).
- **Truncated.** Each PDF is written to `data/annotation/pdfs/` without its triaged disclosure tail, so only the
  pages that matter are annotated and scored.
- **Deterministic.** The same `--seed` (default 0) and inventory give the same sample.

The parse runs still read the original PDFs (listed in `manifest.txt`), but every run is scored only on the pages
the dataset keeps, the same pages the gold covers.

**Choose N for your annotation budget.** Every kept page of every document becomes one image to annotate. The
command logs the total, e.g. `dataset: 100 documents, 640 pages, in data/annotation`.

**A dataset is never overwritten.** The command refuses a folder that already holds one, because the runs and the
gold are tied to it. To start again, move or delete `data/annotation/` first (and `outputs/`, whose runs belong to
the old sample).

**Screening and held-out documents.** Each document is assigned to a split by its content hash: about 80% for
**screen** (E0 to E5) and about 20% for **holdout** (E6 only). The split is fixed, so it never depends on which
documents have been annotated.

## 4. Annotate, on a rolling basis

### Prepare the CVAT tasks (once)

```bash
uv run python -m annotation prepare
```

This runs Docling over the dataset to pre-annotate every page, then writes the CVAT project into
`data/annotation/cvat/`:

- `cvat_description.json`: the label set for the project
- `cvat_tasks/task_00/`, `task_01/`, ...: up to 20 page images per task
- `cvat_tasks/task_00_preannotate.xml`, ...: each task's pre-annotations

### Annotate the screening documents first

E1 to E5 only use the screening documents, so annotate those first and the held-out ones last:

```bash
uv run python -m annotation tasks
```

```
task_00  screen           20 pages from 3 documents
task_01  holdout+screen   20 pages from 4 documents
...
```

### In CVAT

1. Create a project and paste the contents of `cvat/cvat_description.json` as its raw labels (once).
2. For each task: create a task from the images in `cvat/cvat_tasks/task_XX/`, then upload
   `cvat/cvat_tasks/task_XX_preannotate.xml` as its annotations (CVAT for images 1.1 format).
3. Correct the boxes, labels and the reading-order line:
   - **Correct every box,** not only the obvious errors. The pre-annotations come from Docling's default layout
     model, so any box left as it was favours that model in E2.
   - **Draw text boxes tight to the text.** Docling shrinks each text box to its lines, and boxes are matched at
     50% overlap.
   - **Titles** can be labelled `title`. Docling itself never outputs `title` (it relabels titles as section
     headers), so layout F1 counts both as `section_header`.
4. **Export a task only once it is fully corrected.** Any box left uncorrected in an export counts as gold.
   Export the task's annotations as **CVAT for images 1.1** and save the zip in
   `data/annotation/cvat/cvat_annotations/zips/`.

### Collect the gold (after each batch of exports)

```bash
uv run python -m annotation collect
```

This rebuilds `data/annotation/gold/` from every zip so far. Then run `experiments log` (section 6): it notices the
gold has changed and scores every run again, so all rows are always compared against the same gold.

> Check the first partial collect: confirm that `annotation collect` succeeds when only some tasks have been
> exported, and that the log reports gold for only the documents you exported.

## 5. Run the experiments (GPU box)

### All experiments

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml
```

This writes a run config for each variant to `configs/stage_1_1/`, then parses B0, P0 and every variant, one run per
free GPU at a time. Experiments with no variants yet (E5 and E6 until you fill them in) are skipped, with a log
line saying so.

Start with E0 alone. It is only three runs (B0, P0 and the B0 re-run), so it tests the setup end to end before the
full batch, and gives you the noise floor:

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml E0
```

### One or several experiments

List their ids:

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml E2
```

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml E1 E3 E4
```

### Choosing GPUs

By default each run goes to a GPU with at least half its memory free, most free first. To choose:

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml --gpus 0,2
```

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml --min-free 0.3
```

GPU numbers are as `nvidia-smi` shows them. Each run's output goes to `outputs/stage_1_1/logs/<config>.log`:

```bash
tail -f outputs/stage_1_1/logs/docling_b0.log
```

### Interrupted runs and re-runs

Re-running the same command is always safe. A run only parses the documents its directory is missing, so finished
runs return at once and interrupted ones carry on.

### See what would run without parsing

```bash
uv run python -m experiments configs configs/experiments/stage_1_1.yaml
```

This writes the run configs and prints each one's state (`parsed` or `to parse`) and its `pipeline run` command.

### Parse a single config by hand

`CUDA_VISIBLE_DEVICES` picks the GPU; without it Docling takes the first GPU, busy or not. `--shard i/n` splits a
run over several processes, each parsing every n-th document:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m pipeline run configs/stage_1_1/e2_egret_xlarge_bbcbe9.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m pipeline run configs/docling_b0.yaml --shard 0/2
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m pipeline run configs/docling_b0.yaml --shard 1/2
```

## 6. Score and log to MLflow

```bash
uv run python -m experiments log configs/experiments/stage_1_1.yaml
```

Or for some experiments only:

```bash
uv run python -m experiments log configs/experiments/stage_1_1.yaml E1 E2
```

For each experiment this:

1. scores every run that has been parsed since it was last scored, or not scored since the gold last changed,
   writing `<run>/evaluation/` (`pages.csv`, `metrics.json`, `params.json`);
2. compares every row with B0 on the same pages, with a 95% bootstrap interval that resamples documents;
3. logs one MLflow experiment, `stage-1.1/<id> <title>`, with one MLflow run per row. Logging again replaces the
   earlier runs, so nothing is duplicated.

Run it whenever there are new runs or new gold. Before any gold is collected it logs the gold-free metrics only.

The defaults are `--inventory storage/inventory.db` and `--gold data/annotation`; pass these if yours are elsewhere.

## 7. Review in MLflow

Start the UI from the repository root, where `mlflow.db` is:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open <http://localhost:5000>. If the UI runs on the GPU box, forward the port from your machine first:

```bash
ssh -L 5000:localhost:5000 <gpu-box>
```

What you will find:

- **`stage-1.1/00 overview`**: the description holds the stage's aim, every experiment with its winner, and the
  metric definitions.
- **`stage-1.1/E<n> ...`**, one per experiment. Its description holds the question, what each row changes, the
  settings held fixed, a table of means and a table of differences from B0.
- **Each row** (MLflow run) has tags `role` (`reference`, `control`, `variant`) and `row`. It logs:
  - the metric means (e.g. `layout.f1`, `body.loss`);
  - the difference from B0 (`diff/<metric>`, with `diff_low/` and `diff_high/` for the interval);
  - page counts (`better/`, `worse/`, `tie/`);
  - its full evaluation folder and config as artifacts, and the per-stratum comparison as `comparison.json`.

Reading the differences: ✓ marks an interval entirely on the better side, ✗ one entirely on the worse side. Each
metric's direction is taken into account, so for `body.loss` a negative difference is better. Compare against E0: a
difference no bigger than B0's difference from its own re-run is noise.

**Do not look at the held-out results before E6.** E0 to E5 show screening results only, but each row's
`evaluation/metrics.json` artifact also holds `holdout/...` means. Leave those unopened until the final choice is
made, or they stop being a fair check.

## 8. Decide, combine and validate

### Record a winner

When an experiment is reviewed, set `winner:` on it in
[stage_1_1.yaml](configs/experiments/stage_1_1.yaml): a variant name, or `control` if B0 stays best. It records the
decision in the overview and changes no other experiment's runs.

### E5: combine

List the combinations to test as variants, each with every setting it combines, on B0. Test every combination if
3 or fewer settings beat the noise floor; otherwise add them one at a time and keep each one that still helps.

```yaml
  - id: E5
    variants:
      backend_model: {backend: pypdfium2, layout_model: layout_egret_large}
      backend_model_tables: {backend: pypdfium2, layout_model: layout_egret_large, overrides: {do_table_structure: false}}
```

Then parse and log it:

```bash
uv run python -m experiments run configs/experiments/stage_1_1.yaml E5
```

```bash
uv run python -m experiments log configs/experiments/stage_1_1.yaml E5
```

### E6: validate on the held-out documents

List the winner and runner-up as runs that already exist. E6 parses nothing; it scores those runs on the held-out
documents against B0:

```yaml
  - id: E6
    variants:
      winner: {from: E5.backend_model_tables}
      runner_up: {from: E2.egret_large}
```

```bash
uv run python -m experiments log configs/experiments/stage_1_1.yaml E6
```

If the winner's gain over B0 holds here, it is real. If it shrinks to nothing, or the runner-up overtakes it, prefer
the simpler or cheaper option.

## 9. Inspecting runs

An interactive page per document, with every box drawn over the page:

```bash
uv run python -m pipeline visualize outputs/docling_b0 --labels --reading-order
```

One document only (repeat `--doc` for more); doc ids are the original PDF's file name without `.pdf`:

```bash
uv run python -m pipeline visualize outputs/docling_b0 --doc <doc_id> --labels
```

The body text a run outputs, one file per document:

```bash
uv run python -m pipeline serialize outputs/docling_b0 --serialization body
```

Score or compare runs outside an experiment:

```bash
uv run python -m evaluation score outputs/docling_b0 outputs/pypdf_default
```

```bash
uv run python -m evaluation compare outputs/docling_b0 outputs/stage_1_1/e1_pypdfium2_9fcc88 --metric extracted.garble_rate --metric extracted.dropped_text_rate
```

## 10. Optional: page tags

The spec's `strata: data/strata.csv` adds one comparison per tag to every row's `comparison.json`, so a regression on
one kind of page is not hidden by the mean. It is a hand-made CSV; tags are space-separated, and an empty page tags
the whole document:

```csv
doc_id,page,tags
GS_Copper_Outlook_2026,3,multi_column table_heavy
BNP_EM_Weekly_2026_11,,footnote_heavy
```

Without the file, comparisons are over all pages only, and `experiments log` says so.

## 11. Changing the spec safely

- **Adding a variant or experiment** is always safe: it gets its own run directory.
- **Changing a variant's settings** gives it a new directory automatically: the name ends in a hash of the settings.
- **Changing B0 or P0** ([docling_b0.yaml](configs/docling_b0.yaml), [pypdf_default.yaml](configs/pypdf_default.yaml)):
  their output directories are bound to their settings, and `pipeline run` refuses a directory holding another
  config's run (`holds a run with other settings`). Give the changed config a new `name` and `output`. The variants
  inherit B0's settings, so they all re-parse into new directories too.
- **Building on another experiment's winner:** set `control: <id>` on an experiment instead of `control: base`. It
  then changes settings on that winner, and is skipped until the winner is set.

## Troubleshooting

| Message | Fix |
|:--------|:----|
| `no inventory at storage/inventory.db` | Build it (section 2), or pass `--inventory` |
| `... already holds a dataset` | Use a new `--out` folder, or move the old dataset (section 3) |
| `no dataset in data/annotation` | Run `pipeline dataset` before `annotation prepare` |
| `uv: command not found` | `~/.local/bin` is not on your `PATH` (section 1) |
| Certificate error downloading from Nexus | Set `UV_SYSTEM_CERTS=true`, or pass `--system-certs` (section 1) |
| `docling-eval is not installed` | `uv tool install "docling-eval[cvat_tools]"` (section 1) |
| `no CVAT export in .../zips` | Export at least one task's zip into `cvat/cvat_annotations/zips/` |
| `document ... appears twice` (scoring) | An earlier collect left files behind: delete `data/annotation/gold/` and run `annotation collect` again |
| `no GPU has 50% of its memory free` | Wait, lower `--min-free`, or pass `--gpus` |
| `holds a run with other settings` | A config changed under an existing directory; see section 11 |
| `... has no option ...` (at parse time) | A wrong option path under `overrides:` in the spec |
| `skipped: E5 has no variants yet` | Expected until you fill in E5 (section 8) |
