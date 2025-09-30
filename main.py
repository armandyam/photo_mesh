import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import mediapipe as mp
from model import BiSeNet  # from official repo
import subprocess
import argparse

# === CONFIG ===
WEIGHTS_PATH = "79999_iter.pth"
LABELS_TO_KEEP = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
MODEL_INPUT_SIZE = 512
SUBSAMPLE_N = 20

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")


def load_bisenet():
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    net.to(device)
    net.eval()
    return net


def find_midline(img):
    H, W = img.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise Exception("❌ No face detected with Mediapipe")
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye_x = int(landmarks[33].x * W)
        right_eye_x = int(landmarks[263].x * W)
        midline_x = (left_eye_x + right_eye_x) // 2
    print(f"ℹ️ Midline at x = {midline_x}")
    return midline_x


def segment_face(net, img):
    H, W = img.shape[:2]
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    inp = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(inp)[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    parsing = cv2.resize(parsing, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = np.isin(parsing, LABELS_TO_KEEP).astype(np.uint8) * 255
    return mask


def extract_contour(mask, midline_x):
    mask_left = np.zeros_like(mask)
    mask_left[:, :midline_x] = mask[:, :midline_x]
    contours, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    contour = largest[:, 0, :]

    x_vals, y_vals = contour[:, 0], contour[:, 1]
    bump_mask = (x_vals >= midline_x)

    if np.any(bump_mask):
        y_top_bump = np.min(y_vals[bump_mask])
        y_bottom_bump = np.max(y_vals[bump_mask])
    else:
        y_top_bump, y_bottom_bump = np.min(y_vals), np.max(y_vals)

    new_contour = []
    for x, y in contour:
        if y_top_bump <= y <= y_bottom_bump and x >= midline_x:
            new_contour.append([midline_x, y])
        else:
            new_contour.append([x, y])

    return np.array(new_contour, dtype=int)


def save_contour_and_image(img, contour, base_name):
    cv2.polylines(img, [contour.reshape(-1, 1, 2)], isClosed=False, color=(255, 255, 255), thickness=2)
    contour_img = os.path.join(OUTPUT_DIR, f"{base_name}_face_contour_final_clean_no_magic.jpg")
    contour_points = os.path.join(OUTPUT_DIR, f"{base_name}_face_contour_final_clean_no_magic_points.txt")
    cv2.imwrite(contour_img, img)
    np.savetxt(contour_points, contour, fmt="%d")
    print(f"✅ Saved: {contour_img} & {contour_points}")


def write_in2d(contour, H, W, subsample_n, in2d_file):
    max_dim = max(H, W)
    normalized = np.zeros_like(contour, dtype=float)
    normalized[:, 0] = contour[:, 0] / max_dim
    normalized[:, 1] = (H - contour[:, 1]) / max_dim

    normalized_sub = normalized[::subsample_n]
    print(f"ℹ️ Writing {len(normalized_sub)} points out of {len(normalized)}")

    with open(in2d_file, 'w') as f:
        f.write("splinecurves2dv2\n1\n\npoints\n")
        for idx, (x, y) in enumerate(normalized_sub, 1):
            f.write(f"{idx}\t{x:.6f}\t{y:.6f}\n")

        f.write("\nsegments\n")
        for idx in range(1, len(normalized_sub)):
            f.write(f"1 0 2 {idx} {idx+1} -bc=1\n")
        f.write(f"1 0 2 {len(normalized_sub)} 1 -bc=1\n")

        f.write("\nmaterials\n1 face -maxh=1000\n")

    print(f"✅ Written .in2d: {in2d_file}")


def run_netgen(in2d_file, mesh_file):
    print("🚀 Running Netgen...")
    try:
        from netgen.geom2d import SplineGeometry
        from netgen.meshing import MeshingParameters
        
        # Read the .in2d file and create mesh using Python API
        geo = SplineGeometry()
        geo.Load(in2d_file)
        
        # Set meshing parameters - try different attribute names
        mp = MeshingParameters()
        if hasattr(mp, 'maxh'):
            mp.maxh = 1000
        elif hasattr(mp, 'max_element_size'):
            mp.max_element_size = 1000
        
        # Generate mesh
        mesh = geo.GenerateMesh(mp=mp)
        
        # Save mesh
        mesh.Save(mesh_file)
        print(f"✅ Netgen finished. Mesh saved as {mesh_file}")
        
    except Exception as e:
        print(f"❌ Netgen Python API failed: {e}")
        print("🔄 Trying alternative approach...")
        try:
            # Try using ngsolve instead
            from ngsolve import Mesh
            from netgen.geom2d import SplineGeometry
            
            geo = SplineGeometry()
            geo.Load(in2d_file)
            mesh = geo.GenerateMesh()
            mesh.Save(mesh_file)
            print(f"✅ NGSolve mesh generation finished. Mesh saved as {mesh_file}")
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            print("🔄 Skipping mesh generation - will use existing mesh if available")
            # Don't fail the entire script, just skip mesh generation


def read_mesh(vol_path):
    import gzip
    if vol_path.endswith('.gz'):
        with gzip.open(vol_path, 'rt') as f:
            lines = f.readlines()
    else:
        with open(vol_path) as f:
            lines = f.readlines()

    # Points
    for i, line in enumerate(lines):
        if line.strip().lower() == "points":
            num_points = int(lines[i+1].strip())
            points = [list(map(float, l.strip().split()[:2])) for l in lines[i+2:i+2+num_points]]
            break

    # Triangles
    for i, line in enumerate(lines):
        if line.strip().lower() == "surfaceelements":
            num_elements = int(lines[i+1].strip())
            triangles = []
            for l in lines[i+2:i+2+num_elements]:
                parts = l.strip().split()
                triangles.append([int(parts[-3])-1, int(parts[-2])-1, int(parts[-1])-1])
            break

    return np.array(points), np.array(triangles)

def parse_dat(filepath, scalar_col=2):  # 0-based: 0=x,1=y,2=W1…
    points = []
    triangles = []
    solution = []
    node_offset = 0

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ZONE"):
            # extract N, E
            tokens = line.split(',')
            N = int([t.split('=')[1] for t in tokens if 'N=' in t][0])
            E = int([t.split('=')[1] for t in tokens if 'E=' in t][0])

            # read N nodes
            local_nodes = []
            local_sol = []
            for j in range(i+1, i+1+N):
                vals = list(map(float, lines[j].split()))
                local_nodes.append([vals[0], vals[1]])
                local_sol.append(vals[scalar_col])
            points.extend(local_nodes)
            solution.extend(local_sol)

            # read E triangles
            for j in range(i+1+N, i+1+N+E):
                idxs = [int(v)-1 + node_offset for v in lines[j].split()]
                triangles.append(idxs)

            node_offset += N
            i = i+1+N+E
        else:
            i +=1

    points = np.array(points)
    triangles = np.array(triangles)
    solution = np.array(solution)

    print(f"✅ Parsed {len(points)} nodes, {len(triangles)} triangles")
    return points, triangles, solution


def draw_overlay_with_pde(img, points, triangles, contour, solution, base_name, alpha=0.3):
    H, W = img.shape[:2]
    midline_x = W // 2
    max_dim = max(W, H)

    # Rescale points to image
    points_img = points * max_dim
    points_img[:,1] = H - points_img[:,1]

    sol_norm = (solution - np.min(solution)) / (np.max(solution)-np.min(solution)+1e-8)

    # Start with a copy of the original photo
    overlay = img.copy()

    # Create a transparent layer just for the triangles
    triangle_layer = np.zeros_like(img, dtype=np.uint8)

    # === Fill triangles in separate layer ===
    for tri in triangles:
        pts = points_img[tri].astype(int)
        color_val = np.mean(sol_norm[tri])
        color = cv2.applyColorMap(np.uint8([[color_val*255]]), cv2.COLORMAP_JET)[0,0,:].tolist()

        # Only draw left of midline
        if np.all(pts[:,0] <= midline_x+10):
            cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    # Blend only the triangle_layer with original photo
    mask = (triangle_layer > 0).any(axis=2)
    overlay[mask] = (alpha * triangle_layer[mask] + (1-alpha) * img[mask]).astype(np.uint8)

    # === Then draw mesh ===
    for tri in triangles:
        pts = points_img[tri].astype(int).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], True, (0,0,0), 1)

    # === Draw nodes ===
    for pt in points_img:
        cv2.circle(overlay, tuple(pt.astype(int)), 1, (0,0,0), -1)

    # === Draw contour ===
    cv2.polylines(overlay, [contour.reshape((-1,1,2))], False, (0,0,0), 1)

    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_mesh_pde_contour_overlay.png")
    cv2.imwrite(output_file, overlay)
    print(f"✅ Saved: {output_file}")

def read_dat_with_rgb(dat_path):
    fine_points, fine_triangles, colors = [], [], []
    coarse_zones = []
    with open(dat_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ZONE"):
            tokens = line.split(',')
            N = int([t.split('=')[1] for t in tokens if 'N=' in t][0])
            E = int([t.split('=')[1] for t in tokens if 'E=' in t][0])
            p0 = len(fine_points)
            for j in range(i+1, i+1+N):
                vals = lines[j].split()
                x, y = float(vals[0]), float(vals[1])
                r, g, b = map(int, vals[2:5])
                fine_points.append([x,y])
                colors.append([b,g,r])
            tris = []
            for j in range(i+1+N, i+1+N+E):
                idxs = [int(v)-1 + p0 for v in lines[j].split()]
                tris.append(idxs)
            coarse_zones.append((p0, N, tris))
            i = i+1+N+E
        else:
            i+=1

    fine_points = np.array(fine_points)
    colors = np.array(colors)
    return fine_points, colors, coarse_zones

def generate_solution(points, solution_path):
    """Generate a synthetic PDE solution for the mesh points"""
    x_max = np.max(points[:,0])
    y_max = np.max(points[:,1])
    
    # Synthetic solution: sin(πx/x_max) * sin(πy/y_max)
    solution = np.sin(np.pi * points[:,0] / x_max) * np.sin(np.pi * points[:,1] / y_max)
    
    # Normalize to [0,1]
    solution -= solution.min()
    solution /= solution.max()
    
    np.savetxt(solution_path, solution, fmt="%.6f")
    print(f"✅ Generated synthetic solution: {solution_path}")
    return solution

def read_solution_txt(solution_path):
    return np.loadtxt(solution_path)

def overlay_on_image(
    img,
    coarse_points,
    coarse_triangles,
    fine_points=None,
    fine_colors=None,
    coarse_zones=None,
    solution=None,
    contour=None,
    alpha=1.,
    mode="dat",
    base_name="output"
):
    H,W = img.shape[:2]
    midline_x = W//2
    max_dim = max(W,H)

    coarse_points_img = coarse_points * max_dim
    coarse_points_img[:,1] = H - coarse_points_img[:,1]

    if fine_points is not None:
        fine_points_img = fine_points * max_dim
        fine_points_img[:,1] = H - fine_points_img[:,1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    triangle_layer = np.zeros_like(img, dtype=np.uint8)

    if mode == "dat":
        for zone in coarse_zones:
            p0, N, fine_tris = zone
            for tri in fine_tris:
                pts = fine_points_img[tri].astype(int)
                color = np.mean(fine_colors[tri], axis=0).astype(int).tolist()
                if np.any(pts[:,0]<=midline_x+10):
                    cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    elif mode == "solution":
        sol_norm = (solution - np.min(solution)) / (np.max(solution)-np.min(solution)+1e-8)
        for tri in coarse_triangles:
            pts = coarse_points_img[tri].astype(int)
            color_val = np.mean(sol_norm[tri])
            color = cv2.applyColorMap(np.uint8([[color_val*255]]), cv2.COLORMAP_JET)[0,0,:].tolist()
            if np.any(pts[:,0]<=midline_x+10):
                cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    # Blend
    mask = (triangle_layer>0).any(axis=2)
    overlay[mask] = (alpha * triangle_layer[mask] + (1-alpha) * overlay[mask]).astype(np.uint8)

    print(f"Painted triangles: {np.sum(mask)} pixels")

    # Draw coarse mesh
    for tri in coarse_triangles:
        pts = coarse_points_img[tri].astype(int).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], True, (0,0,0), 1)

    for pt in coarse_points_img:
        cv2.circle(overlay, tuple(pt.astype(int)), 1, (0,0,0), -1)

    if contour is not None:
        cv2.polylines(overlay, [contour.reshape((-1,1,2))], False, (0,0,0), 1)

    outname = os.path.join(OUTPUT_DIR, f"{base_name}_overlay_{mode}_coarse_mesh.png")
    cv2.imwrite(outname, overlay)
    print(f"✅ Saved: {outname}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Photo to Mesh Pipeline')
    parser.add_argument('input_image', help='Path to input image file')
    args = parser.parse_args()
    
    IMAGE_PATH = args.input_image
    
    # Generate unique filenames based on input image
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    IN2D_FILE = os.path.join(OUTPUT_DIR, f"{base_name}_mesh_normalized_subsampled.in2d")
    MESH_FILE = os.path.join(OUTPUT_DIR, f"{base_name}_out.mesh.vol.gz")
    SOLUTION_PATH = os.path.join(OUTPUT_DIR, f"{base_name}_solution.txt")
    DAT_PATH = os.path.join(OUTPUT_DIR, f"{base_name}_subdivided_image.dat")
    
    print(f"🚀 Processing image: {IMAGE_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # === Step 1: Face segmentation & contour ===
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ Error: Could not load image {IMAGE_PATH}")
        return
    
    H, W = img.shape[:2]
    print(f"📐 Image dimensions: {W}x{H}")

    net = load_bisenet()
    midline_x = find_midline(img)
    mask = segment_face(net, img)
    contour = extract_contour(mask, midline_x)
    save_contour_and_image(img.copy(), contour, base_name)

    # === Step 2: Write .in2d & run Netgen ===
    write_in2d(contour, H, W, SUBSAMPLE_N, IN2D_FILE)
    run_netgen(IN2D_FILE, MESH_FILE)

    # === Step 3: Read mesh & overlay ===
    points, triangles = read_mesh(MESH_FILE)
    # draw_overlay(cv2.imread(IMAGE_PATH), points, triangles, contour)

    # === GENERATE SOLUTION ===
    solution = generate_solution(points, SOLUTION_PATH)

    # === GENERATE SUBDIVIDED DAT FILE ===
    from writer_dat import write_subdivided_dat
    write_subdivided_dat(IMAGE_PATH, points, triangles, n_sub=10, outpath=DAT_PATH)

    draw_overlay_with_pde(cv2.imread(IMAGE_PATH), points, triangles, contour, solution, base_name)

    coarse_points = points
    coarse_triangles = triangles
     # Load image & contour
    img = cv2.imread(IMAGE_PATH)
    contour_points_file = os.path.join(OUTPUT_DIR, f"{base_name}_face_contour_final_clean_no_magic_points.txt")
    try:
        contour = np.loadtxt(contour_points_file, dtype=int)
    except:
        contour = None

    # Option 1: .dat
    fine_points, fine_colors, coarse_zones = read_dat_with_rgb(DAT_PATH)
    overlay_on_image(
        img, coarse_points, coarse_triangles,
        fine_points=fine_points, fine_colors=fine_colors, coarse_zones=coarse_zones,
        contour=contour, alpha = 1., mode="dat", base_name=base_name
    )

    # Option 2: .solution.txt
    solution = read_solution_txt(SOLUTION_PATH)
    overlay_on_image(
        img, coarse_points, coarse_triangles,
        solution=solution,
        contour=contour, alpha=0.5, mode="solution", base_name=base_name
    )

if __name__ == "__main__":
    main()