# ModelNet40 Data Adapter Design

## Goal

Enable the repository to train on the local `ModelNet40` dataset without changing the purpose of the paper method. The adaptation must only bridge `OFF` mesh data into the existing point-cloud training interface already used by CAME-Net.

## Scope

This work covers:

- Loading `ModelNet40/ModelNet40/<class>/<split>/*.off`
- Parsing standard `OFF` mesh files
- Converting each mesh into a fixed-size point cloud by area-weighted triangle sampling
- Normalizing sampled points with the existing point-cloud normalization utility
- Reusing the existing point-cloud batch format: `{'point_coords', 'labels'}`
- Switching the example training entry point from synthetic data to `ModelNetDataset`
- Adding tests for dataset loading and basic training-path compatibility

This work does not cover:

- Any change to CAME-Net geometric modules, equivariance objective, or paper claims
- Any new modality, annotation, or supervision signal
- Any offline preprocessing pipeline that creates a second permanent dataset format
- Any benchmark-tuned training recipe beyond making the local dataset usable

## Constraints

- The paper method purpose must remain unchanged. Only the data ingestion path may be adapted.
- The training code should continue to consume point clouds as raw XYZ coordinates.
- The adaptation must not introduce assumptions that require image, text, pose, or mesh-topology inputs at model level.
- Test-time behavior should be reproducible.
- The implementation should stay lightweight and local to the current repository.

## Dataset Observations

The local dataset is arranged as:

`ModelNet40/ModelNet40/<class_name>/<train|test>/*.off`

Each sample is a standard `OFF` mesh file:

- line 1: `OFF`
- line 2: `<num_vertices> <num_faces> <num_edges>`
- next `num_vertices` lines: vertex coordinates `(x, y, z)`
- next `num_faces` lines: polygon indices, with triangles already common in the dataset

The class directory names define the classification labels.

## Recommended Approach

Implement a real `ModelNetDataset` that reads meshes on demand and samples point clouds from triangle surfaces online.

This is preferred over directly using mesh vertices because surface sampling produces a point distribution closer to the intended point-cloud setting. It also avoids introducing a separate preprocessing pipeline, which would add operational complexity without improving alignment to the method.

## Design

### 1. Dataset indexing

`ModelNetDataset` will:

- scan all class directories under the dataset root
- sort class names to build stable `class_to_idx`
- collect `(off_path, label)` pairs for the requested split
- expose `__len__` based on indexed files

The default root expected by the example training entry point will be `ModelNet40/ModelNet40`.

### 2. OFF parsing

The loader will parse a standard `OFF` file into:

- `vertices: float32[N, 3]`
- `faces: int64[F, K]`

If a face has more than three vertices, it will be triangulated using fan triangulation. This keeps the downstream sampler simple while preserving the mesh surface.

Malformed files should raise a clear `ValueError` that includes the file path.

### 3. Triangle-surface point sampling

Each mesh will be converted to `num_points` XYZ samples by:

1. computing each triangle area
2. sampling triangles proportionally to area
3. sampling barycentric coordinates inside each selected triangle
4. forming 3D points from the sampled barycentric combination

Degenerate triangles with zero area will be ignored. If a mesh has no valid triangle area after parsing, the loader will fall back to sampling from vertices so the training path remains usable instead of crashing on one bad sample.

### 4. Normalization and augmentation

After sampling, the dataset will call `PointCloudProcessor.normalize`.

Split behavior:

- `train`: random surface sampling each access, optional existing point-cloud augmentation
- `test`: deterministic surface sampling using a seed derived from the sample index, no augmentation

This keeps evaluation stable while preserving training diversity.

### 5. Batch contract

The dataset output remains:

```python
{
    "point_coords": torch.FloatTensor[num_points, 3],
    "labels": torch.LongTensor[]
}
```

`collate_fn` does not need an interface change because the shape contract stays fixed.

### 6. Training entry point

The example block in `train.py` will:

- import `ModelNetDataset` instead of `RandomPointCloudDataset`
- point to the local `ModelNet40/ModelNet40` root
- create train and validation loaders from the real dataset

The core training loop stays unchanged because it already consumes `point_coords` and `labels`.

## Error Handling

- Missing dataset root: raise `FileNotFoundError` with the expected path
- Missing split directory contents: raise `ValueError` with split and root details
- Invalid OFF header or malformed counts: raise `ValueError` with file path
- Mesh with no valid faces: fall back to vertex sampling instead of aborting the run

The fallback is limited to data loading and does not alter the model or method objective.

## Testing Strategy

Tests will cover:

1. `ModelNetDataset` indexes local ModelNet40 files and exposes non-zero length
2. one fetched sample has shape `(num_points, 3)` and a valid integer class label
3. `collate_fn` produces batch tensors with expected dimensions
4. a small real-data batch can pass through the model forward path

Tests will avoid long training runs. The target is compatibility verification, not benchmark evaluation.

## Non-Goals and Method Preservation

This design preserves the paper method purpose because the model still receives point clouds and operates through the same Clifford-algebra point-cloud branch. The only change is how local mesh files are transformed into point coordinates before entering the existing network.

No part of the design:

- changes MPE/GCA/GLN semantics
- changes the approximate equivariance objective
- changes the multimodal theory or claims
- redefines the point-cloud representation used by the model

## Implementation Boundary

The expected code touch points are:

- `data_utils.py`
- `train.py`
- dataset-focused tests in `test_came_net.py` or a new targeted test file

No model-architecture files should be modified unless a compatibility bug is discovered during testing.
