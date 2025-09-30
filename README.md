## photo_mesh

Pipeline to extract a face contour, generate a 2D triangular mesh (Netgen/NGSolve), and overlay synthetic PDE solutions.

### Setup
- Python 3.11 recommended
- Create venv and install deps:



Note:  installs Netgen binaries. On macOS, this works in CPU mode.

### Model weights
Place BiSeNet weights at  in project root (ignored by git).

### Inputs
Place images in  as .

### Run


Outputs are written to  with the input basename prefix.

### Files
- : pipeline
- , : BiSeNet model
- : subdivided Tecplot .dat writer

### Notes
- Netgen meshes are saved as  and parsed in Python.
- The PDE solution here is synthetic; replace with your solver as needed.
