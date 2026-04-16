# Small ModelNet Experiment Design

## Context

The repository already supports:

- Real `ModelNet40` loading from local `OFF` meshes
- Surface point sampling from meshes into point clouds
- Point-cloud classification with `CAMENet`
- Standard training and validation loops

The current gap is not model capability, but experiment ergonomics. There is no lightweight, reproducible experiment path for quickly demonstrating that the method can train on a small subset of `ModelNet40`, report accuracy, and generate visual outputs without running a full benchmark.

## Goal

Add a lightweight experiment path that can run a small but structured `ModelNet40` classification experiment within roughly 15 minutes on the user's `RTX 4060 Laptop GPU`, while preserving the existing CAME-Net method and training code paths.

The experiment should:

- Use a fixed 5-class subset from `ModelNet40`
- Train a lightweight `CAMENet` configuration on small sampled subsets
- Report train and validation trends
- Save final accuracy metrics and confusion matrix
- Save a small set of point-cloud prediction visualizations

## Non-Goals

This work will not:

- Change the paper method, model mathematics, or geometric design
- Turn the small experiment into a benchmark-quality full `ModelNet40` run
- Replace the main training entrypoint in `train.py`
- Add heavy interactive visualization or web serving
- Introduce large preprocessing pipelines or new dataset formats

## Recommended Experiment Configuration

The default small experiment will use:

- Classes: `airplane`, `chair`, `lamp`, `sofa`, `toilet`
- Train samples per class: `100`
- Test samples per class: `30`
- Points per sample: `256`
- Model hidden dimension: `32`
- Number of CAME layers: `2`
- Attention heads: `4`
- Batch size: `8`
- Epochs: `10`
- Learning rate: `1e-3`
- Equivariance regularization: enabled with default weight `0.1`

These defaults are chosen to keep runtime modest while still producing a visible learning curve and non-trivial classification results.

## Approach

The implementation will add a dedicated small-experiment script instead of overloading the existing training entrypoint with many experiment-specific branches.

The experiment flow is:

1. Resolve the local `ModelNet40` root
2. Filter the dataset down to the fixed 5 classes
3. Cap the number of train and test samples per class
4. Train a lightweight `CAMENet`
5. Evaluate on the held-out subset
6. Save metrics, plots, and qualitative prediction figures

This keeps the existing training code largely intact and introduces a narrow experiment layer on top of the current dataset and model infrastructure.

## Components

### 1. Small experiment entrypoint

Add a new script:

- `run_small_modelnet_experiment.py`

Responsibilities:

- Build the filtered train and test datasets
- Construct a lightweight `CAMENet`
- Run training and validation
- Evaluate the final model
- Save all outputs into a timestamped artifact directory
- Print a short terminal summary at the end

This script is the main user-facing entrypoint for the lightweight demo experiment.

The default user command should be:

- `& 'D:\\projectcreating\\anaconda\\envs\\pytorch\\python.exe' run_small_modelnet_experiment.py`

### 2. Dataset filtering support

Add a thin wrapper dataset around `ModelNetDataset` that filters by class and caps per-class sample counts.

Required support:

- `allowed_classes`
- `max_samples_per_class`
- deterministic subset selection based on the existing sample order when augmentation is disabled

This layer must not change how point clouds are generated. Meshes must still be converted to point clouds by area-weighted surface sampling from triangle faces.

### 3. Experiment visualization and reporting helpers

Add lightweight helpers for:

- plotting training curves
- plotting a confusion matrix
- plotting a small set of point-cloud test examples with predicted and true labels
- saving metrics and history as JSON

These helpers should remain simple and local to the experiment workflow.

## Data Flow

The experiment data flow is:

1. Read local `OFF` meshes from `ModelNet40`
2. Keep only the five selected classes
3. Select a small fixed-size subset per class for train and test
4. Sample each mesh into a 256-point cloud using the existing surface sampling logic
5. Normalize the point cloud using the existing point-cloud normalization path
6. Feed the points into the existing point-cloud branch of `CAMENet`
7. Collect epoch-by-epoch loss and accuracy
8. Run final evaluation on the filtered test split
9. Save metrics and figures to disk

No new data representation is introduced. The model still receives point coordinates as before.

## Artifacts

Each experiment run will write to:

- `artifacts/small_modelnet_experiment/<timestamp>/`

Required files:

- `config.json`
- `history.json`
- `metrics.json`
- `summary.md`
- `training_curves.png`
- `confusion_matrix.png`
- `sample_predictions.png`

`summary.md` should contain:

- selected classes
- dataset sizes
- runtime summary
- final validation accuracy
- per-class accuracy
- artifact file list

## Error Handling

The experiment should fail clearly when:

- the local `ModelNet40` root does not exist
- one of the required classes is missing
- the filtered split becomes empty
- there are too few samples to satisfy the requested subset limits

The failure mode should be a direct, readable exception or terminal message that tells the user what is missing or misconfigured.

## Testing

Testing should stay lightweight and focused on the new experiment path.

Required checks:

- filtered dataset construction works for a chosen subset of classes
- per-class sample caps are respected
- experiment artifact directory and required files are created
- a tiny smoke experiment can run end-to-end on a very small subset

Tests should avoid long training. The smoke path should use a much smaller subset and fewer epochs than the default experiment configuration.

## Success Criteria

This task is successful if:

- the user can run one small command and start a 5-class demo experiment
- the experiment completes in roughly 15 minutes or less on the user's machine under the intended lightweight configuration
- the run produces interpretable accuracy outputs and saved visual artifacts
- the existing method and main training path remain unchanged in purpose

## Implementation Notes

- Prefer reusing the existing `train_came_net`, `validate`, and evaluation utilities when practical
- Keep experiment-only logic out of the core model implementation unless a small shared helper clearly improves the design
- The default class set should be fixed, not randomly sampled, so results are reproducible and the experiment remains stable across runs
- The experiment should prefer GPU when available and otherwise fall back to CPU with a clear warning that runtime may exceed the intended budget
