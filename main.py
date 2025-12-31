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
# Full body extraction: including neck, clothing, and all facial features
LABELS_TO_KEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# Presets:
# - full: includes neck and clothing (current default)
# - face: excludes neck (14), necklace (15), and clothing (16)
LABELS_PRESET_FULL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
LABELS_PRESET_FACE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]
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


def find_midline_from_mask(mask):
    """
    Find the face midline from the segmentation mask by finding the center of mass
    at different vertical positions (top, middle, bottom).
    Returns a line as two points (top, bottom).
    """
    H, W = mask.shape
    
    # Find the bounding box of the face
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        raise Exception("❌ No face region found in mask")
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Sample at three vertical positions: top (25%), middle (50%), bottom (75%)
    y_top = int(y_min + (y_max - y_min) * 0.25)
    y_mid = int(y_min + (y_max - y_min) * 0.50)
    y_bottom = int(y_min + (y_max - y_min) * 0.75)
    
    # Find center of mass at each vertical position
    def get_center_x_at_y(y):
        row = mask[y, :]
        x_coords = np.where(row > 0)[0]
        if len(x_coords) > 0:
            return int(np.mean(x_coords))
        return W // 2
    
    face_mid_x = get_center_x_at_y(y_top)
    face_mid_y = y_top
    
    nose_mid_x = get_center_x_at_y(y_mid)
    nose_mid_y = y_mid
    
    chin_mid_x = get_center_x_at_y(y_bottom)
    chin_mid_y = y_bottom
    
    return (face_mid_x, face_mid_y), (nose_mid_x, nose_mid_y), (chin_mid_x, chin_mid_y)

def find_midline(img, mask=None):
    """
    Find the face midline as an angled line that follows the face orientation.
    Uses three key points: face mid (forehead), nose mid, and chin mid.
    Returns a line as two points (top, bottom) that can extend beyond image bounds.
    
    If mask is provided, uses the mask to calculate midline. Otherwise tries MediaPipe.
    """
    H, W = img.shape[:2]
    
    # If mask is provided, use it to calculate midline
    if mask is not None:
        try:
            face_mid, nose_mid, chin_mid = find_midline_from_mask(mask)
            face_mid_x, face_mid_y = face_mid
            nose_mid_x, nose_mid_y = nose_mid
            chin_mid_x, chin_mid_y = chin_mid
        except Exception as e:
            raise Exception(f"❌ Failed to find midline from mask: {e}")
    else:
        # Try MediaPipe (may not work with new API)
        try:
            from mediapipe.tasks.python import vision
            from mediapipe import tasks
            
            base_options = tasks.BaseOptions()
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            detector = vision.FaceLandmarker.create_from_options(options)
            
            mp_image = vision.TaskRunner.Image.create_from_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)
            
            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                raise Exception("❌ No face detected with Mediapipe")
            
            landmarks = detection_result.face_landmarks[0]
            face_mid_x = landmarks[10].x * W
            face_mid_y = landmarks[10].y * H
            nose_mid_x = landmarks[1].x * W
            nose_mid_y = landmarks[1].y * H
            chin_mid_x = landmarks[175].x * W
            chin_mid_y = landmarks[175].y * H
        except Exception as e:
            raise Exception(f"❌ MediaPipe face detection failed: {e}")
    
    # Fit a line through these three points using least squares
    # We'll use the top (face_mid) and bottom (chin_mid) points and extend
    top_point = np.array([face_mid_x, face_mid_y])
    bottom_point = np.array([chin_mid_x, chin_mid_y])
    
    # Calculate line direction
    direction = bottom_point - top_point
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    else:
        # Fallback: vertical line
        direction = np.array([0.0, 1.0])
    
    # Extend line to image boundaries
    # We want to extend from top_point upward and from bottom_point downward
    # Calculate parameter t where y = top_point[1] + t * direction[1]
    
    # Extend upward from top_point
    if direction[1] != 0:
        t_top = (0 - top_point[1]) / direction[1]  # y = 0
    else:
        t_top = 0
    
    # Extend downward from bottom_point
    if direction[1] != 0:
        t_bottom = (H - bottom_point[1]) / direction[1]  # y = H
    else:
        t_bottom = H - bottom_point[1]
    
    # Calculate extended points
    line_top = top_point + direction * t_top
    line_bottom = bottom_point + direction * t_bottom
    
    # Ensure points are within reasonable bounds (allow extension beyond image)
    line_top[0] = np.clip(line_top[0], -W, 2*W)
    line_top[1] = np.clip(line_top[1], -H, 2*H)
    line_bottom[0] = np.clip(line_bottom[0], -W, 2*W)
    line_bottom[1] = np.clip(line_bottom[1], -H, 2*H)
    
    print(f"ℹ️ Face midline calculated:")
    print(f"   Face mid (top): ({face_mid_x:.1f}, {face_mid_y:.1f})")
    print(f"   Nose mid: ({nose_mid_x:.1f}, {nose_mid_y:.1f})")
    print(f"   Chin mid (bottom): ({chin_mid_x:.1f}, {chin_mid_y:.1f})")
    print(f"   Line: ({line_top[0]:.1f}, {line_top[1]:.1f}) -> ({line_bottom[0]:.1f}, {line_bottom[1]:.1f})")
    
    return (line_top, line_bottom)


def segment_face(net, img, labels_to_keep):
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
    mask = np.isin(parsing, labels_to_keep).astype(np.uint8) * 255
    return mask


def extract_contour(mask, cutoff_line=None, cutoff_x=None):
    """
    Extract contour with configurable cutoff position.
    
    Args:
        mask: Segmentation mask
        cutoff_line: Tuple of (line_top, line_bottom) points defining an angled cutoff line
        cutoff_x: X coordinate for vertical cutoff line (for backward compatibility)
    """
    H, W = mask.shape
    
    # Determine which side of the line to keep (left side)
    if cutoff_line is not None:
        line_top, line_bottom = cutoff_line
        # Create a function to determine which side of the line a point is on
        # Using cross product: if cross > 0, point is on left side
        line_vec = line_bottom - line_top
        
        def is_left_of_line(point):
            """Check if point is on the left side of the line"""
            point_vec = point - line_top
            cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
            return cross > 0
        
        # Create mask for everything on the left side of the line
        mask_left = np.zeros_like(mask)
        for y in range(H):
            for x in range(W):
                if mask[y, x] > 0:
                    if is_left_of_line(np.array([x, y])):
                        mask_left[y, x] = mask[y, x]
    else:
        # Fallback to vertical line
        cutoff_x = cutoff_x if cutoff_x is not None else W // 2
        mask_left = np.zeros_like(mask)
        mask_left[:, :cutoff_x] = mask[:, :cutoff_x]
    
    contours, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.array([], dtype=int).reshape(0, 2)
        
    largest = max(contours, key=cv2.contourArea)
    contour = largest[:, 0, :]

    # Clip contour points that are on the right side of the line to the line
    if cutoff_line is not None:
        line_top, line_bottom = cutoff_line
        line_vec = line_bottom - line_top
        
        def is_left_of_line(point):
            point_vec = point - line_top
            cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
            return cross > 0
        
        def project_to_line(point):
            """Project point onto the line and return the closest point on the line"""
            point_vec = point - line_top
            t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
            t = np.clip(t, 0, 1)
            return line_top + t * line_vec
        
        new_contour = []
        for pt in contour:
            if is_left_of_line(pt):
                new_contour.append(pt)
            else:
                # Project to line
                projected = project_to_line(pt)
                new_contour.append(projected.astype(int))
        
        return np.array(new_contour, dtype=int)
    else:
        # Original vertical line logic
        x_vals, y_vals = contour[:, 0], contour[:, 1]
        bump_mask = (x_vals >= cutoff_x)

        if np.any(bump_mask):
            y_top_bump = np.min(y_vals[bump_mask])
            y_bottom_bump = np.max(y_vals[bump_mask])
        else:
            y_top_bump, y_bottom_bump = np.min(y_vals), np.max(y_vals)

        new_contour = []
        for x, y in contour:
            if y_top_bump <= y <= y_bottom_bump and x >= cutoff_x:
                new_contour.append([cutoff_x, y])
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


def generate_contour_mesh(contour: np.ndarray, H: int, W: int, mesh_density_factor=0.5):
    """
    Generate a different mesh for contour visualization.
    Uses coarser mesh density by default.
    """
    max_dim = float(max(W, H))
    gmsh.initialize()
    gmsh.model.add("contour_mesh")
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

        # Mesh options: coarser mesh for contour visualization
        char_len = max_dim / (80.0 * mesh_density_factor)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len * 0.8)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len * 2.0)
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
        points = pts.copy()
        points[:, 0] = points[:, 0] / max_dim
        points[:, 1] = 1.0 - points[:, 1] / max_dim

    finally:
        gmsh.finalize()

    return points, triangles


def generate_voronoi_diagram(points, H, W, contour):
    """
    Generate Voronoi diagram from mesh points (excluding boundary points), constrained to person boundary.
    Returns list of Voronoi edges.
    """
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, Polygon, LineString
    import numpy as np
    
    # Convert normalized points to image coordinates
    max_dim = max(W, H)
    points_img = points * max_dim
    points_img[:, 1] = H - points_img[:, 1]
    
    # Use all mesh points - the mesher already gives us proper mesh points
    # Only filter out points that are exactly on the contour (distance == 0)
    filtered_points = []
    if contour is not None and len(contour) > 0:
        for pt in points_img:
            is_exactly_on_contour = False
            for contour_pt in contour:
                dist = np.sqrt((pt[0] - contour_pt[0])**2 + (pt[1] - contour_pt[1])**2)
                if dist == 0:  # Exact match - only contour points
                    is_exactly_on_contour = True
                    break
            
            if not is_exactly_on_contour:
                filtered_points.append(pt)
    else:
        filtered_points = points_img.tolist()
    
    if len(filtered_points) < 3:
        return []
    
    # Generate Voronoi diagram from filtered points
    voronoi_points = np.array(filtered_points)
    vor = Voronoi(voronoi_points)
    
    # Create person boundary polygon
    person_poly = None
    if contour is not None and len(contour) > 0:
        try:
            person_poly = Polygon(contour)
            if not person_poly.is_valid:
                person_poly = person_poly.buffer(0)
        except:
            person_poly = None
    
    # Extract and clip finite edges
    voronoi_edges = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # Only finite edges
            edge = [vor.vertices[ridge[0]], vor.vertices[ridge[1]]]
            
            # Clip edge to person boundary
            if person_poly is not None:
                try:
                    line = LineString(edge)
                    clipped_line = line.intersection(person_poly)
                    
                    if not clipped_line.is_empty:
                        if hasattr(clipped_line, 'coords'):
                            coords = list(clipped_line.coords)
                            if len(coords) >= 2:
                                voronoi_edges.append([coords[0], coords[-1]])
                        elif hasattr(clipped_line, 'geoms'):
                            for geom in clipped_line.geoms:
                                if hasattr(geom, 'coords'):
                                    coords = list(geom.coords)
                                    if len(coords) >= 2:
                                        voronoi_edges.append([coords[0], coords[-1]])
                except:
                    continue
            else:
                voronoi_edges.append(edge)
    
    return voronoi_edges


def is_point_left_of_line(point, line_top, line_bottom):
    """Check if a point is on the left side of a line"""
    line_vec = line_bottom - line_top
    point_vec = point - line_top
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    return cross > 0

def draw_overlay_with_pde(img, points, triangles, contour, solution, base_name, alpha=0.3, cutoff_x=None, cutoff_line=None, separate_contour_mesh=False, contour_mesh_points=None, contour_mesh_triangles=None, show_mesh_nodes=False, show_voronoi=False, hide_contour_outline=False):
    H, W = img.shape[:2]
    if cutoff_x is None and cutoff_line is None:
        cutoff_x = W // 2
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

        # Only draw left of cutoff line
        if cutoff_line is not None:
            # Check if triangle center is on left side of line
            tri_center = np.mean(pts, axis=0)
            line_top, line_bottom = cutoff_line
            if is_point_left_of_line(tri_center, line_top, line_bottom):
                cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)
        else:
            if np.all(pts[:,0] <= cutoff_x+10):
                cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    # Blend only the triangle_layer with original photo
    mask = (triangle_layer > 0).any(axis=2)
    overlay[mask] = (alpha * triangle_layer[mask] + (1-alpha) * img[mask]).astype(np.uint8)

    # === Then draw mesh (only if not using different contour mesh) ===
    if contour_mesh_points is None:
        for tri in triangles:
            pts = points_img[tri].astype(int).reshape((-1,1,2))
            cv2.polylines(overlay, [pts], True, (0,0,0), 1)

    # === Draw nodes (only if not using different contour mesh) ===
    # Removed small black dots that were creating artifacts
    
    # === Draw mesh nodes as bigger dots if requested ===
    if show_mesh_nodes:
        # ONLY draw nodes from the contour mesh (Mesh B) - exclude boundary points
        if contour_mesh_points is not None:
            contour_mesh_points_img = contour_mesh_points * max_dim
            contour_mesh_points_img[:,1] = H - contour_mesh_points_img[:,1]
            for pt in contour_mesh_points_img:
                # Only filter out points that are exactly on the contour (distance == 0)
                is_exactly_on_contour = False
                if len(contour) > 0:
                    for contour_pt in contour:
                        dist = np.sqrt((pt[0] - contour_pt[0])**2 + (pt[1] - contour_pt[1])**2)
                        if dist == 0:  # Exact match - only contour points
                            is_exactly_on_contour = True
                            break
                
                if not is_exactly_on_contour:
                    cv2.circle(overlay, tuple(pt.astype(int)), 6, (0,0,0), -1)  # Black dots

    # === Draw contour ===
    if len(contour) > 0 and not hide_contour_outline:
        # Ensure contour is properly formatted and draw as a single continuous line
        contour_reshaped = contour.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(overlay, [contour_reshaped], False, (255,0,0), 1)  # Red contour line, thin
    
    # === Draw separate contour mesh if provided (replaces original mesh lines) ===
    if contour_mesh_points is not None and contour_mesh_triangles is not None:
        contour_mesh_points_img = contour_mesh_points * max_dim
        contour_mesh_points_img[:,1] = H - contour_mesh_points_img[:,1]
        
        # Draw contour mesh triangles (this replaces the original mesh lines)
        for tri in contour_mesh_triangles:
            pts = contour_mesh_points_img[tri].astype(int)
            # Only draw if triangle is within cutoff
            if cutoff_line is not None:
                tri_center = np.mean(pts, axis=0)
                line_top, line_bottom = cutoff_line
                if is_point_left_of_line(tri_center, line_top, line_bottom):
                    cv2.polylines(overlay, [pts], True, (0,0,0), 3)  # Thicker triangle lines
            else:
                if np.all(pts[:,0] <= cutoff_x+10):
                    cv2.polylines(overlay, [pts], True, (0,0,0), 3)  # Thicker triangle lines
    
    # === Draw Voronoi diagram if requested ===
    if show_voronoi:
        try:
            # ONLY use contour mesh points (Mesh B) for Voronoi
            if contour_mesh_points is not None:
                voronoi_edges = generate_voronoi_diagram(contour_mesh_points, H, W, contour)
                for edge in voronoi_edges:
                    pt1 = edge[0]
                    pt2 = edge[1]
                    # Convert to int coordinates
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0]), int(pt2[1]))
                    cv2.line(overlay, pt1_int, pt2_int, (0, 0, 0), 2)  # Black thicker lines
        except Exception as e:
            print(f"⚠️ Could not generate Voronoi diagram: {e}")

    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_overlay_color_alpha{int(alpha*100)}.png")
    cv2.imwrite(output_file, overlay)
    print(f"✅ Saved: {output_file}")
    
    # === Optional: Create separate contour mesh visualization ===
    if separate_contour_mesh and len(contour) > 0:
        # Create a separate visualization with just the contour mesh
        contour_overlay = img.copy()
        
        # Draw contour as a filled polygon
        if len(contour) > 2:
            cv2.fillPoly(contour_overlay, [contour.reshape((-1,1,2))], (0, 100, 200), lineType=cv2.LINE_AA)
        
        # Draw contour outline
        cv2.polylines(contour_overlay, [contour.reshape((-1,1,2))], False, (0,0,0), 2)
        
        contour_output_file = os.path.join(OUTPUT_DIR, f"{base_name}_contour_mesh_alpha{int(alpha*100)}.png")
        cv2.imwrite(contour_output_file, contour_overlay)
        print(f"✅ Saved contour mesh: {contour_output_file}")

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
    base_name="output",
    cutoff_x=None,
    cutoff_line=None
):
    H,W = img.shape[:2]
    if cutoff_x is None and cutoff_line is None:
        cutoff_x = W//2
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
                if cutoff_line is not None:
                    tri_center = np.mean(pts, axis=0)
                    line_top, line_bottom = cutoff_line
                    if is_point_left_of_line(tri_center, line_top, line_bottom):
                        cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)
                else:
                    if np.any(pts[:,0]<=cutoff_x+10):
                        cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    elif mode == "solution":
        sol_norm = (solution - np.min(solution)) / (np.max(solution)-np.min(solution)+1e-8)
        for tri in coarse_triangles:
            pts = coarse_points_img[tri].astype(int)
            color_val = np.mean(sol_norm[tri])
            color = cv2.applyColorMap(np.uint8([[color_val*255]]), cv2.COLORMAP_JET)[0,0,:].tolist()
            if cutoff_line is not None:
                tri_center = np.mean(pts, axis=0)
                line_top, line_bottom = cutoff_line
                if is_point_left_of_line(tri_center, line_top, line_bottom):
                    cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)
            else:
                if np.any(pts[:,0]<=cutoff_x+10):
                    cv2.fillConvexPoly(triangle_layer, pts, color, lineType=cv2.LINE_AA)

    # Blend
    mask = (triangle_layer>0).any(axis=2)
    overlay[mask] = (alpha * triangle_layer[mask] + (1-alpha) * overlay[mask]).astype(np.uint8)

    print(f"Painted triangles: {np.sum(mask)} pixels")

    # Draw coarse mesh
    for tri in coarse_triangles:
        pts = coarse_points_img[tri].astype(int).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], True, (0,0,0), 1)

    # Removed small black dots that were creating artifacts

    if contour is not None and len(contour) > 0:
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
    parser.add_argument('--cutoff-position', type=float, default=0.5, 
                       help='Cutoff position as fraction of image width (0-1, default: 0.5 for center)')
    parser.add_argument('--separate-contour-mesh', action='store_true',
                       help='Generate separate contour mesh visualization')
    parser.add_argument('--use-different-contour-mesh', action='store_true',
                       help='Use a different mesh for contour visualization (coarser)')
    parser.add_argument('--show-mesh-nodes', action='store_true',
                       help='Show mesh nodes as bigger dots')
    parser.add_argument('--show-voronoi', action='store_true',
                       help='Show Voronoi diagram with thin lines')
    parser.add_argument('--hide-contour-outline', action='store_true',
                       help='Hide the contour outline (only show mesh nodes)')
    parser.add_argument('--use-face-midline', action='store_true',
                       help='Use exact face midline (forehead to nose center) instead of image center cutoff')
    parser.add_argument('--segmentation-mode', choices=['full','face'], default='face',
                       help='Segmentation preset: full (includes neck/clothing) or face (excludes them)')
    parser.add_argument('--labels-to-keep', type=str, default=None,
                       help='Override labels as comma-separated integers, e.g. "1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18"')
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

    # Determine segmentation labels
    if args.labels_to_keep is not None:
        try:
            labels_to_keep = [int(x.strip()) for x in args.labels_to_keep.split(',') if x.strip()]
        except Exception:
            raise ValueError("Invalid --labels-to-keep format. Use comma-separated integers, e.g. 1,2,3")
        print(f"🎯 Using custom labels_to_keep: {labels_to_keep}")
    else:
        if args.segmentation_mode == 'face':
            labels_to_keep = LABELS_PRESET_FACE
        else:
            labels_to_keep = LABELS_PRESET_FULL
        print(f"🎯 Using segmentation preset '{args.segmentation_mode}': {labels_to_keep}")

    net = load_bisenet()
    mask = segment_face(net, img, labels_to_keep)
    
    # Calculate cutoff position - always use angled face midline
    cutoff_line = None
    cutoff_x = None
    try:
        # Use the mask to calculate midline (more reliable than MediaPipe)
        cutoff_line = find_midline(img, mask=mask)
        print(f"ℹ️ Using angled face midline from segmentation mask")
    except Exception as e:
        print(f"⚠️ Face midline detection failed: {e}")
        print("🔄 Falling back to image center cutoff")
        cutoff_x = int(args.cutoff_position * W)
        print(f"ℹ️ Cutoff position: x = {cutoff_x} ({args.cutoff_position*100:.1f}% of image width)")
    
    contour = extract_contour(mask, cutoff_line=cutoff_line, cutoff_x=cutoff_x)

    # === Step 2: Pure-Python meshing from contour ===
    if len(contour) > 0:
        points, triangles = generate_mesh_from_contour(contour, H, W)
        # draw_overlay(cv2.imread(IMAGE_PATH), points, triangles, contour)

        # === EXTRACT IMAGE COLORS AT MESH POINTS ===
        solution = extract_image_colors_at_points(points, img, H, W)

        # === GENERATE DIFFERENT CONTOUR MESH IF REQUESTED ===
        contour_mesh_points = None
        contour_mesh_triangles = None
        if args.use_different_contour_mesh:
            print("🔄 Generating different mesh for contour visualization...")
            contour_mesh_points, contour_mesh_triangles = generate_contour_mesh(contour, H, W, mesh_density_factor=0.3)
            print(f"✅ Generated contour mesh: {len(contour_mesh_points)} nodes, {len(contour_mesh_triangles)} triangles")

        # Write only the two requested visualizations
        draw_overlay_with_pde(cv2.imread(IMAGE_PATH), points, triangles, contour, solution, base_name, alpha=max(0.0, min(1.0, args.alpha)), cutoff_x=cutoff_x, cutoff_line=cutoff_line, separate_contour_mesh=args.separate_contour_mesh, contour_mesh_points=contour_mesh_points, contour_mesh_triangles=contour_mesh_triangles, show_mesh_nodes=args.show_mesh_nodes, show_voronoi=args.show_voronoi, hide_contour_outline=args.hide_contour_outline)

        # Second visualization (solution on coarse mesh)
        coarse_points = points
        coarse_triangles = triangles
        img = cv2.imread(IMAGE_PATH)
        overlay_on_image(
            img, coarse_points, coarse_triangles,
            solution=solution,
            contour=contour, alpha=max(0.0, min(1.0, args.alpha)), mode="solution", base_name=base_name, cutoff_x=cutoff_x, cutoff_line=cutoff_line
        )
    else:
        print("ℹ️ No contour generated (0% cutoff) - saving original image")
        # Save original image when no meshing is applied
        original_img = cv2.imread(IMAGE_PATH)
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_original.png")
        cv2.imwrite(output_file, original_img)
        print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    main()