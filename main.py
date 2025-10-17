import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import mediapipe as mp
from model import BiSeNet  # from official repo
import subprocess
import argparse
import gmsh

# === CONFIG ===
WEIGHTS_PATH = "79999_iter.pth"
# Face parsing labels (BiSeNet 19-class model - CelebAMask-HQ dataset):
# 0: Background, 1: Skin, 2: Left Brow, 3: Right Brow, 4: Left Eye, 5: Right Eye
# 6: Glasses, 7: Left Ear, 8: Right Ear, 9: Ear Ring, 10: Nose, 11: Mouth
# 12: Upper Lip, 13: Lower Lip, 14: Neck, 15: Necklace, 16: Cloth, 17: Hair, 18: Hat
# Keeping facial regions + right ear + hair + lips, excluding: left ear (7), neck (14), necklace (15), cloth (16)
LABELS_TO_KEEP = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 18]
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
    # Intentionally do not write intermediate contour artifacts to disk
    # Keep drawing in-memory only for downstream use if needed
    pass


def write_in2d(contour, H, W, subsample_n, in2d_file):
    # No-op retained for compatibility; Netgen path removed
    print(f"✅ Prepared geometry for meshing (skipped file write)")


def run_netgen(in2d_file, mesh_file):
    # Netgen removed – function kept for compatibility
    print("ℹ️ Netgen step skipped (pure-Python meshing)")


def read_mesh(vol_path):
    # Kept for historical compatibility; not used in pure-Python path
    raise RuntimeError("read_mesh is not used in the pure-Python meshing path")

def generate_mesh_from_contour(contour: np.ndarray, H: int, W: int):
    """
    High-quality 2D meshing with Gmsh:
    - Build a spline from the face contour
    - Create a plane surface and generate a triangular mesh
    Returns (points_normalized, triangles_indices)
    """
    max_dim = float(max(W, H))
    gmsh.initialize()
    gmsh.model.add("face")
    try:
        # Add points (in pixel coordinates)
        point_tags = []
        for x, y in contour.astype(float):
            tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0)
            point_tags.append(tag)

        # Close the loop
        if point_tags[0] != point_tags[-1]:
            point_tags.append(point_tags[0])

        # Spline through points, curve loop, and surface
        spline = gmsh.model.geo.addSpline(point_tags)
        loop = gmsh.model.geo.addCurveLoop([spline])
        surf = gmsh.model.geo.addPlaneSurface([loop])

        # Mesh options: quality and size adapted to image size - higher resolution
        char_len = max_dim / 90.0  # Increased resolution (was 80.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len * 0.3)  # Smaller min size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len * 1.2)  # Smaller max size
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # Extract nodes and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        pts = np.array(node_coords, dtype=float).reshape(-1, 3)[:, :2]

        tris = []
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        for etype, nodes in zip(elem_types, elem_node_tags):
            if etype == 2:  # 3-node triangle
                conn = np.array(nodes, dtype=int).reshape(-1, 3) - 1
                tris.append(conn)
        if not tris:
            raise RuntimeError("No triangle elements generated by Gmsh")
        triangles = np.vstack(tris)

        # Normalize to [0,1] with y-up
        points_norm = pts.copy()
        points_norm[:, 0] = pts[:, 0] / max_dim
        points_norm[:, 1] = (H - pts[:, 1]) / max_dim
        return points_norm, triangles
    finally:
        gmsh.finalize()

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

    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_overlay_color_alpha{int(alpha*100)}.png")
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

def generate_solution(points):
    """Generate a synthetic PDE solution for the mesh points (in-memory only)."""
    x_max = np.max(points[:,0])
    y_max = np.max(points[:,1])
    
    # Synthetic solution: sin(πx/x_max) * sin(πy/y_max)
    solution = np.sin(np.pi * points[:,0] / x_max) * np.sin(np.pi * points[:,1] / y_max)
    
    # Normalize to [0,1]
    solution -= solution.min()
    solution /= solution.max()
    
    return solution

def extract_image_colors_at_points(points, img, H, W):
    """Extract image colors at mesh points and convert to grayscale for JET colormap."""
    max_dim = max(W, H)
    
    # Convert points back to image coordinates
    points_img = points * max_dim
    points_img[:, 1] = H - points_img[:, 1]  # Flip y-coordinate
    
    # Clamp coordinates to image bounds
    points_img[:, 0] = np.clip(points_img[:, 0], 0, W-1)
    points_img[:, 1] = np.clip(points_img[:, 1], 0, H-1)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sample colors at mesh points with bilinear interpolation for smoother results
    colors = []
    for pt in points_img:
        x, y = pt[0], pt[1]
        
        # Bilinear interpolation for smoother color sampling
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, W-1), min(y1 + 1, H-1)
        
        # Get the four surrounding pixels
        c11 = gray_img[y1, x1]
        c12 = gray_img[y2, x1] if y2 < H else c11
        c21 = gray_img[y1, x2] if x2 < W else c11
        c22 = gray_img[y2, x2] if (x2 < W and y2 < H) else c11
        
        # Bilinear interpolation weights
        wx = x - x1
        wy = y - y1
        
        # Interpolate
        c = (1-wx)*(1-wy)*c11 + wx*(1-wy)*c21 + (1-wx)*wy*c12 + wx*wy*c22
        colors.append(c)
    
    colors = np.array(colors, dtype=np.float32)
    
    # Normalize to [0,1] for JET colormap
    colors = colors / 255.0
    
    return colors

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

    if mode == "solution":
        outname = os.path.join(OUTPUT_DIR, f"{base_name}_overlay_gray_alpha{int(alpha*100)}.png")
    else:
        outname = os.path.join(OUTPUT_DIR, f"{base_name}_dat_overlay_alpha{int(alpha*100)}.png")
    cv2.imwrite(outname, overlay)
    print(f"✅ Saved: {outname}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Photo to Mesh Pipeline')
    parser.add_argument('input_image', help='Path to input image file')
    parser.add_argument('--alpha', type=float, default=0.3, help='Overlay alpha in [0,1] (default: 0.3)')
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

    # === Step 2: Pure-Python meshing from contour ===
    points, triangles = generate_mesh_from_contour(contour, H, W)
    # draw_overlay(cv2.imread(IMAGE_PATH), points, triangles, contour)

    # === EXTRACT IMAGE COLORS AT MESH POINTS ===
    solution = extract_image_colors_at_points(points, img, H, W)

    # Write only the two requested visualizations
    draw_overlay_with_pde(cv2.imread(IMAGE_PATH), points, triangles, contour, solution, base_name, alpha=max(0.0, min(1.0, args.alpha)))

    # Second visualization (solution on coarse mesh)
    coarse_points = points
    coarse_triangles = triangles
    img = cv2.imread(IMAGE_PATH)
    overlay_on_image(
        img, coarse_points, coarse_triangles,
        solution=solution,
        contour=contour, alpha=max(0.0, min(1.0, args.alpha)), mode="solution", base_name=base_name
    )

if __name__ == "__main__":
    main()