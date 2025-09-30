import numpy as np
import cv2

def read_mesh(vol_path):
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

def subdivide_triangle(A, B, C, n):
    """
    Subdivide triangle ABC into n×n fine triangles.
    Args:
        A, B, C: (2,) np arrays (vertices)
        n: number of subdivisions per edge
    Returns:
        fine_points: (M,2)
        fine_tris: (K,3) — indices into fine_points
    """
    fine_points = []
    for i in range(n+1):
        for j in range(n+1-i):
            u = i/n
            v = j/n
            w = 1 - u - v
            pt = u*A + v*B + w*C
            fine_points.append(pt)
    fine_points = np.array(fine_points)

    fine_tris = []
    idx = lambda i,j: (i*(n+1) - i*(i-1)//2 + j)
    for i in range(n):
        for j in range(n-i):
            p1 = idx(i,j)
            p2 = idx(i+1,j)
            p3 = idx(i,j+1)
            fine_tris.append([p1,p2,p3])
            if j < n-i-1:
                p4 = idx(i+1,j+1)
                fine_tris.append([p2,p4,p3])
    fine_tris = np.array(fine_tris)
    return fine_points, fine_tris


def write_subdivided_dat(image_path, points, triangles, n_sub=4, outpath="subdivided_image.dat"):
    """
    Writes a Tecplot .dat file with subdivided triangles colored by image RGB.
    Args:
        image_path: original photo
        points: (N,2) coarse mesh points (normalized [0,1])
        triangles: (M,3) coarse mesh triangles
        n_sub: number of subdivisions per triangle
    """
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    max_dim = max(W,H)

    with open(outpath, "w") as f:
        f.write('TITLE = "Subdivided Image RGB"\n')
        f.write('VARIABLES = "X" "Y" "R" "G" "B"\n')

        zone_idx = 1
        for tri in triangles:
            A = points[tri[0]]
            B = points[tri[1]]
            C = points[tri[2]]
            fine_pts, fine_tris = subdivide_triangle(A,B,C,n_sub)

            N = len(fine_pts)
            E = len(fine_tris)

            f.write(f'ZONE T="Triangle_{zone_idx}", N={N}, E={E}, F=FEPOINT, ET=TRIANGLE\n')

            # sample RGB at each fine point
            for pt in fine_pts:
                pt_img = pt * max_dim
                x_pix = np.clip(int(pt_img[0]), 0, W-1)
                y_pix = np.clip(H - int(pt_img[1]), 0, H-1)
                b,g,r = img[y_pix,x_pix]
                # save in normalized mesh coords
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {r} {g} {b}\n")

            # write fine triangles (1-based)
            for fine_tri in fine_tris:
                f.write(f"{fine_tri[0]+1} {fine_tri[1]+1} {fine_tri[2]+1}\n")

            zone_idx +=1

    print(f"✅ Subdivided `.dat` written: {outpath}")


if __name__ == "__main__":
    # === EXAMPLE USAGE ===
    VOL_PATH = "out.mesh"
    IMAGE_PATH = "jk.jpg"
    OUT_DAT = "subdivided_image.dat"
    N_SUB = 10  # subdivisions per triangle

    points, triangles = read_mesh(VOL_PATH)
    write_subdivided_dat(IMAGE_PATH, points, triangles, n_sub=N_SUB, outpath=OUT_DAT)