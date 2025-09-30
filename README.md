## Photo Mesh Pipeline

Extract a face contour from a photo, generate a 2D triangular mesh with Netgen/NGSolve, and overlay a synthetic PDE solution on the face region.

### Features
- Face parsing with BiSeNet (pretrained weights)
- Midline detection with MediaPipe FaceMesh
- 2D mesh generation from the face contour (Netgen/NGSolve)
- Synthetic PDE visualization and image-colored mesh overlays
- Clean input/output separation (`inputs/` → `output/`)

### Requirements
- Python 3.11 (recommended)
- System packages: none required beyond a working Python toolchain

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Note: The `ngsolve` wheel bundles Netgen and provides the Python API on macOS (CPU).

### Model Weights
Download BiSeNet weights and place at `79999_iter.pth` in the project root. This file is ignored by git.

### Project Layout
- `main.py` — pipeline entrypoint and CLI
- `model.py`, `resnet.py` — BiSeNet model and backbone
- `writer_dat.py` — writes subdivided Tecplot `.dat` from the coarse mesh and image
- `inputs/` — put your input images here (e.g., `input_01.jpg`)
- `output/` — all generated artifacts per input image basename

### Usage
Run the pipeline on an input image:
```bash
source venv/bin/activate
python main.py inputs/input_01.jpg
```

Outputs are written to `output/` with the input basename prefix, for example:
- `<name>_face_contour_final_clean_no_magic.jpg`
- `<name>_face_contour_final_clean_no_magic_points.txt`
- `<name>_mesh_normalized_subsampled.in2d`
- `<name>_out.mesh.vol.gz`
- `<name>_solution.txt`
- `<name>_subdivided_image.dat`
- `<name>_mesh_pde_contour_overlay.png`
- `<name>_overlay_dat_coarse_mesh.png`
- `<name>_overlay_solution_coarse_mesh.png`

### Implementation Notes
- Mesh generation uses Netgen via the Python API from `ngsolve`; if unavailable it attempts a fallback.
- Mesh files are saved as compressed `.vol.gz` and parsed directly in Python.
- The PDE solution is synthetic (normalized product of sines). Replace with your solver as needed.

### Troubleshooting
- If MediaPipe cannot detect a face, ensure the image has a clear, frontal face.
- If `ngsolve` install fails, upgrade pip and retry. On Apple Silicon, ensure you use the venv’s Python.

