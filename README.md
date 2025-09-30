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

Note: The `ngsolve` wheel bundles Netgen and provides the Python API.

### Model and Weights
This project uses an open-source BiSeNet implementation and pretrained weights:

- `model.py` and `resnet.py` are adapted from the Face Parsing BiSeNet implementation (CelebAMask-HQ):
  - Repository: `https://github.com/zllrunning/face-parsing.PyTorch`
  - Original backbone and segmentation modules were refactored locally for this pipeline.

- Pretrained parsing weights (BiSeNet on face parsing) can be downloaded from the same repository releases or mirrors, for example:
  - `79999_iter.pth` (checkpoint used here): `https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view` (linked from the repo README)

Place the downloaded `79999_iter.pth` in the project root before running.

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
- `<name>_mesh_pde_contour_overlay.png`
- `<name>_overlay_solution_coarse_mesh.png`

### Implementation Notes
- Mesh generation uses Netgen via the Python API from `ngsolve`; if unavailable it attempts a fallback.
- Mesh files are saved as compressed `.vol.gz` and parsed directly in Python.
- The PDE solution is synthetic (normalized product of sines). Replace with your solver as needed.

### Troubleshooting
- If MediaPipe cannot detect a face, ensure the image has a clear, frontal face.
- If `ngsolve` install fails, upgrade pip and retry. On Apple Silicon, ensure you use the venv’s Python.

