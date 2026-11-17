import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/mnt/sdb/lys/yolo11/datasets/DAIR-V2X-I")
META = ROOT / "single-infrastructure-side"

CALIB = META / "calib/virtuallidar_to_camera/000000.json"
INTR = META / "calib/camera_intrinsic/000000.json"
LABEL = META / "label/camera/000000.json"
IMAGE = ROOT / "single-infrastructure-side-image/000000.jpg"

OUT = Path("runs/dair_calib_check")
OUT.mkdir(parents=True, exist_ok=True)

cal = json.loads(CALIB.read_text())
intr = json.loads(INTR.read_text())
labels = json.loads(LABEL.read_text())

R_raw = np.asarray(cal["rotation"], dtype=np.float64)
t = np.asarray(cal["translation"], dtype=np.float64).reshape(3)

# SVD projection onto SO(3)
U, S, Vt = np.linalg.svd(R_raw)
Dmat = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
R_so3 = U @ Dmat @ Vt

K = np.asarray(intr["cam_K"]).reshape(3, 3)
D = np.asarray(intr["cam_D"], dtype=np.float64)

img = Image.open(IMAGE).convert("RGB")
W, H = img.size

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def make_corners(obj):
    dim = obj["3d_dimensions"]
    loc = obj["3d_location"]

    h = float(dim["h"])
    w = float(dim["w"])
    l = float(dim["l"])

    x = float(loc["x"])
    y = float(loc["y"])
    z = float(loc["z"])
    yaw = float(obj["rotation"])

    local = np.array([
        [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
        [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],
        [0,0,0,0,h,h,h,h]
    ])

    c, s = np.cos(yaw), np.sin(yaw)

    Rz = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]
    ])

    bottom_center = np.array([x,y,z-h/2])

    return (Rz @ local).T + bottom_center

def project(points, R, use_distortion=True):
    # Virtual LiDAR -> Camera
    cam = points @ R.T + t

    if np.any(cam[:,2] <= 1e-6):
        return None

    xn = cam[:,0] / cam[:,2]
    yn = cam[:,1] / cam[:,2]

    if use_distortion:
        k1,k2,p1,p2,k3 = D

        r2 = xn*xn + yn*yn
        radial = 1+k1*r2+k2*r2**2+k3*r2**3

        xd = xn*radial + 2*p1*xn*yn + p2*(r2+2*xn*xn)
        yd = yn*radial + p1*(r2+2*yn*yn) + 2*p2*xn*yn
    else:
        xd, yd = xn, yn

    uv = np.column_stack([
        K[0,0]*xd + K[0,2],
        K[1,1]*yd + K[1,2]
    ])

    return uv

def box_iou(a,b):
    x1=max(a[0],b[0])
    y1=max(a[1],b[1])
    x2=min(a[2],b[2])
    y2=min(a[3],b[3])

    inter=max(0,x2-x1)*max(0,y2-y1)

    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])

    return inter/max(aa+bb-inter,1e-9)

def evaluate(R, use_distortion, name):
    scores=[]
    canvas=img.copy()
    draw=ImageDraw.Draw(canvas)

    for obj in labels:
        try:
            pts=make_corners(obj)
            uv=project(pts,R,use_distortion)

            if uv is None or not np.isfinite(uv).all():
                continue

            bb=obj["2d_box"]
            gt=[
                float(bb["xmin"]),
                float(bb["ymin"]),
                float(bb["xmax"]),
                float(bb["ymax"])
            ]

            pred=[
                float(uv[:,0].min()),
                float(uv[:,1].min()),
                float(uv[:,0].max()),
                float(uv[:,1].max())
            ]

            clipped=[
                max(0,pred[0]),max(0,pred[1]),
                min(W-1,pred[2]),min(H-1,pred[3])
            ]

            score=box_iou(clipped,gt)
            scores.append(score)

            # Ground-truth 2D box: red
            draw.rectangle(gt,outline="red",width=2)

            # Projected 3D cuboid: green
            for a,b in EDGES:
                draw.line(
                    [tuple(uv[a]),tuple(uv[b])],
                    fill="lime",width=2
                )

        except Exception:
            continue

    if scores:
        print(
            f"{name}: count={len(scores)}, "
            f"mean IoU={np.mean(scores):.4f}, "
            f"median IoU={np.median(scores):.4f}, "
            f"P10={np.percentile(scores,10):.4f}"
        )
    else:
        print(name, "NO VALID OBJECTS")

    canvas.save(OUT/f"{name}.png")
    return scores

print("="*60)
print("RAW ROTATION")
print(R_raw)
print("det:",np.linalg.det(R_raw))

print("\nSVD ROTATION")
print(R_so3)
print("det:",np.linalg.det(R_so3))
print("orthogonality error:",
      np.linalg.norm(R_so3@R_so3.T-np.eye(3)))

print("\n========== PROJECTION COMPARISON ==========")

for distortion in (False,True):
    print("\nDistortion:",distortion)

    evaluate(R_raw,distortion,
             f"raw_dist_{distortion}")

    evaluate(R_so3,distortion,
             f"svd_dist_{distortion}")

print("\nImages saved to:",OUT.resolve())
print("FINISHED")
