#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover a metric KITTI cuboid from 8 ordered 2D corners + P2 + a metric ground plane.

KITTI rectified camera coordinates: X right, Y DOWN, Z forward.
Input corner convention: indices 0-3 are CYCLIC bottom corners, 4-7 corresponding top.
This matches the user's YOLO cuboid labels: cls cx cy w h x1 y1 ... x8 y8.

--source labels: uses existing normalized 21-col YOLO labels (GEOMETRIC GT upper bound,
                 not independently predicted keypoints).
--source gt: projects KITTI label_2 cuboids to 2D (geometry self-check only).

No 3D object ground truth is accessed by the reconstruction solver; KITTI GT is read
for matching/evaluation (and for deriving 2D inputs only in --source gt).
A fixed y=ground-y plane is an APPROXIMATION, not an official KITTI ground plane.
Requires: numpy scipy pillow. Outputs CSV, PNG, and recovered JSON without changing inputs.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from PIL import Image, ImageDraw

SIGNS = np.array([[1,1], [1,-1], [-1,-1], [-1,1]], dtype=float)
EDGE_PAIRS = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]

@dataclass
class Cuboid:
    name: str
    h: float
    w: float
    l: float
    center: np.ndarray    # KITTI bottom center (x, y, z)
    yaw: float
    box2d: np.ndarray
    index: int


def parse_calib(path: Path) -> np.ndarray:
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip().startswith('P2:'):
            arr = np.fromstring(line.split(':', 1)[1], sep=' ', dtype=float)
            if len(arr) == 12:
                P = arr.reshape(3,4)
                if abs(np.linalg.det(P[:,:3])) > 1e-9:
                    return P
    raise ValueError(f'Invalid or missing P2: {path}')


def load_kitti_label(path: Path, allowed: set[str]) -> list[Cuboid]:
    result = []
    for idx, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        f = line.split()
        if len(f) < 15 or f[0] not in allowed:
            continue
        try:
            h,w,l = map(float, f[8:11]); pos = np.array([float(x) for x in f[11:14]])
            yaw = float(f[14]); box = np.array([float(x) for x in f[4:8]])
            if min(h,w,l) <= 0 or pos[2] <= 0 or not np.all(np.isfinite(pos)):
                continue
            result.append(Cuboid(f[0],h,w,l,pos,yaw,box,idx))
        except (ValueError, IndexError):
            continue
    return result


def corners(h: float, w: float, l: float, center: np.ndarray,
            yaw: float, signs: np.ndarray = SIGNS) -> np.ndarray:
    """KITTI box: bottom y=center[1]; top y=center[1]-h; yaw about Y."""
    c,s = math.cos(yaw), math.sin(yaw)
    uv = np.asarray(signs, float)
    base = np.column_stack((uv[:,0]*l/2, np.zeros(4), uv[:,1]*w/2))
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    xyz = (base @ R.T) + center.reshape(1,3)
    above = xyz.copy(); above[:,1] -= h
    return np.vstack((xyz, above))


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz,float)
    q = np.hstack((pts,np.ones((len(pts),1)))) @ P.T
    if np.any(q[:,2] <= 0.001):
        raise ValueError('A corner lies behind the camera')
    return q[:,:2] / q[:,2,None]


def image_rays(P: np.ndarray, uv: np.ndarray):
    # P2 is 3x4, not just a 3x3 intrinsic. P2 includes a fourth translation column.
    M = P[:,:3]
    C = -np.linalg.solve(M, P[:,3])  # optical center in rectified coordinates
    D = np.linalg.solve(M, np.c_[uv,np.ones(len(uv))].T).T
    return C, D


def intersect_ground(P: np.ndarray, bottom_uv: np.ndarray,
                     ground_y: float, min_depth=1.0) -> np.ndarray:
    C,D = image_rays(P, bottom_uv)
    if np.any(np.abs(D[:,1]) < 1e-8):
        raise ValueError('A bottom ray is parallel to the ground')
    t = (ground_y-C[1]) / D[:,1]
    if np.any(t <= 0):
        raise ValueError('Ground intersections lie behind the camera')
    pts = C[None,:] + t[:,None]*D
    if np.any(pts[:,2] <= min_depth) or not np.isfinite(pts).all():
        raise ValueError('Invalid ground-ray intersection/depth')
    return pts


def reorder_signs():
    for reverse in (False,True):
        base = SIGNS[::-1].copy() if reverse else SIGNS.copy()
        for shift in range(4):
            yield np.roll(base,shift,axis=0)


def estimate_height_from_top(P: np.ndarray, bottom_xyz: np.ndarray,
                             top_uv: np.ndarray):
    # Top ray X=C+tD must intersect vertical line through bottom point.
    C,D = image_rays(P,top_uv)
    estimates=[]
    for b,d in zip(bottom_xyz,D):
        if np.linalg.norm(d[[0,2]]) < 1e-8:
            continue
        t = np.dot(d[[0,2]],(b-C)[[0,2]]) / np.dot(d[[0,2]],d[[0,2]])
        h = b[1] - (C[1]+t*d[1])
        if t > 0 and np.isfinite(h) and h > 0:
            estimates.append(h)
    if not estimates:
        raise ValueError('Top points do not yield a positive height')
    return float(np.clip(np.median(estimates),0.25,5.0))


def recover_box(uv: np.ndarray, P: np.ndarray, ground_y: float,
                max_reproj_px: float=60.0):
    """Return predicted KITTI cuboid, 8 metric corners, reprojection RMS.

    All information consumed by the solver: uv, P, ground_y. No GT 3D parameters.
    """
    uv = np.asarray(uv,float).reshape(8,2)
    if not np.isfinite(uv).all():
        raise ValueError('Nonfinite keypoints')
    if ground_y <= 0:
        raise ValueError('ground-y must be positive (KITTI Y points downward)')
    bottom = intersect_ground(P,uv[:4],ground_y)
    xz = bottom[:,[0,2]]
    center_guess = xz.mean(axis=0)
    h_init = estimate_height_from_top(P,bottom,uv[4:])
    candidates=[]
    for signs in reorder_signs():
        u = (signs[:,0,None] * xz).mean(axis=0)
        v = (signs[:,1,None] * xz).mean(axis=0)
        l0 = max(0.5,2*np.linalg.norm(u)); w0 = max(0.4,2*np.linalg.norm(v))
        # Rotated KITTI length axis in x-z plane: (cos(yaw), -sin(yaw)).
        yaw0 = math.atan2(-u[1],u[0])
        p0 = np.array([center_guess[0],center_guess[1],yaw0,l0,w0,h_init])
        low = np.array([-300.,1.0,-20*math.pi,0.25,0.25,0.25])
        high= np.array([ 300.,250.,20*math.pi,25.,12.,8.])
        p0 = np.clip(p0,low+1e-5,high-1e-5)

        def residual(p):
            cx,cz,yaw,l,w,h = p
            pos = np.array([cx,ground_y,cz])
            try:
                pred=project(P,corners(h,w,l,pos,yaw,signs))
                r=(pred-uv).ravel()
            except ValueError:
                r=np.full(16,1e3)
            # Encourage rectangular base estimates to remain near the planar intersections;
            # do not introduce any GT-derived metric priors.
            return r

        try:
            fit=least_squares(residual,p0,bounds=(low,high),loss='soft_l1',
                              f_scale=5.0,max_nfev=150,xtol=1e-9)
            cx,cz,yaw,l,w,h=fit.x
            xyz=corners(h,w,l,np.array([cx,ground_y,cz]),yaw,signs)
            rms=float(np.sqrt(np.mean((project(P,xyz)-uv)**2)))
            # Encourage length >= width for Car boxes, while fitting geometry first.
            shape_penalty=0.15*max(0.,w-l)**2
            candidates.append((rms+shape_penalty,rms,h,w,l,np.array([cx,ground_y,cz]),yaw,xyz,signs))
        except (ValueError,FloatingPointError):
            continue
    if not candidates:
        raise ValueError('All rectangle fits failed')
    candidates.sort(key=lambda c:c[0])
    _,rms,h,w,l,ctr,yaw,xyz,signs=candidates[0]
    if rms > max_reproj_px:
        raise ValueError(f'Large reprojection error: {rms:.2f}px')
    # Canonicalize a rectangular footprint to length >= width without changing
    # its physical shape (orientation is ambiguous up to a 90-degree axis swap).
    if l < w:
        l,w = w,l
        yaw = yaw - math.pi/2
    # yaw is ambiguous modulo pi for symmetric 3D cuboids.
    box=Cuboid('recovered',h,w,l,ctr,float((yaw+math.pi)%(2*math.pi)-math.pi),
               np.zeros(4),-1)
    return box,xyz,rms


def xy_rect(box: Cuboid) -> np.ndarray:
    # footprint polygon CCW in X-Z. Need orientation invariant for clipping.
    c,s=math.cos(box.yaw),math.sin(box.yaw)
    u=np.array([c,-s]); v=np.array([s,c]); ctr=box.center[[0,2]]
    pts=np.array([ctr+su*box.l/2*u+sv*box.w/2*v
                  for su,sv in ((1,1),(-1,1),(-1,-1),(1,-1))])
    if signed_area(pts)<0:pts=pts[::-1]
    return pts


def signed_area(poly):
    if len(poly)<3:return 0.0
    return 0.5*float(np.sum(poly[:,0]*np.roll(poly[:,1],-1) - poly[:,1]*np.roll(poly[:,0],-1)))


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    out=np.asarray(subject,float)
    for j in range(len(clipper)):
        a,b=clipper[j],clipper[(j+1)%len(clipper)]
        if len(out)==0:break
        result=[]
        def inside(p):
            r=b-a; v=p-a
            return (r[0]*v[1]-r[1]*v[0])>=-1e-9
        def intersection(p,q):
            r=q-p; s=b-a
            den=float(np.linalg.det(np.array([r,-s]).T))
            if abs(den)<1e-12:return q
            t=float(np.linalg.det(np.array([a-p,-s]).T))/den
            return p+t*r
        prev=out[-1]
        for pt in out:
            if inside(pt):
                if not inside(prev):result.append(intersection(prev,pt))
                result.append(pt)
            elif inside(prev):result.append(intersection(prev,pt))
            prev=pt
        out=np.asarray(result,dtype=float).reshape(-1,2)
    return out


def iou3d(b1:Cuboid,b2:Cuboid) -> float:
    p1,p2=xy_rect(b1),xy_rect(b2)
    intersection=polygon_clip(p1,p2)
    inter_area=abs(signed_area(intersection))
    y_top=max(b1.center[1]-b1.h,b2.center[1]-b2.h)
    y_bottom=min(b1.center[1],b2.center[1])
    vi=inter_area*max(0.,y_bottom-y_top)
    v1=b1.h*b1.w*b1.l;v2=b2.h*b2.w*b2.l
    return float(np.clip(vi/max(v1+v2-vi,1e-12),0,1))


def corner_rmse(pred_xyz,gt_xyz):
    # Permutation-invariant eval: 4 bottom and 4 top are each matched independently.
    diffs=[]
    for sl in (slice(0,4),slice(4,8)):
        a,b=pred_xyz[sl],gt_xyz[sl]
        cost=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
        i,j=linear_sum_assignment(cost)
        diffs.extend(np.square(cost[i,j]).tolist())
    return float(np.sqrt(np.mean(diffs)))


def angle_err_mod_pi(a,b):
    # Cuboid geometry alone does not reveal front/back direction.
    delta=abs((a-b+math.pi/2)%math.pi-math.pi/2)
    return float(delta)


def box_iou2d(a,b):
    xx1=max(a[0],b[0]);yy1=max(a[1],b[1])
    xx2=min(a[2],b[2]);yy2=min(a[3],b[3]);inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    ar=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    return inter/max(ar(a)+ar(b)-inter,1e-9)


def get_yolo_records(path:Path,W:int,H:int,class_id:int=0):
    records=[]
    if not path.is_file():return records
    for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        vals=line.split()
        if len(vals) not in (21,29,22,30):
            continue
        try:a=np.array([float(v) for v in vals],float)
        except ValueError:continue
        if round(a[0])!=class_id:continue
        box=a[1:5]
        if len(vals) in (21,22):kp=a[5:21].reshape(8,2)
        else:kp=a[5:29].reshape(8,3)[:,:2]
        if not np.isfinite(kp).all():continue
        uv=kp*np.array([W,H]); x,y,w,h=box*np.array([W,H,W,H])
        if w<=0 or h<=0:continue
        bbox=np.array([x-w/2,y-h/2,x+w/2,y+h/2])
        records.append((line_no,uv,bbox))
    return records


def search_pose_label(root:Path,stem:str):
    candidates=[]
    for prefix in ('kitti_',''):
        for sub in ('labels/train','labels/val','labels/test','labels'):
            candidates.append(root/sub/f'{prefix}{stem}.txt')
    for p in candidates:
        if p.is_file():return p
    return None


def locate_kitti(root:Path) -> Path:
    for base in (root,root/'training',root/'object'/'training',root/'data_object_label_2'/'training'):
        if (base/'label_2').is_dir() and (base/'calib').is_dir():
            return base
    raise FileNotFoundError(f'No KITTI training/label_2 and calib under {root}')


def visualize(image_path, uv, xyz_pred, P, gtbox, out):
    if not image_path.is_file():return
    im=Image.open(image_path).convert('RGB');dr=ImageDraw.Draw(im)
    gt_xyz=corners(gtbox.h,gtbox.w,gtbox.l,gtbox.center,gtbox.yaw)
    gt_uv=project(P,gt_xyz);pred_uv=project(P,xyz_pred)
    for pts,color in ((gt_uv,(45,220,65)),(pred_uv,(30,130,255))):
        for i,j in EDGE_PAIRS:
            dr.line([tuple(pts[i]),tuple(pts[j])],fill=color,width=3)
    for x,y in uv:dr.ellipse([x-3,y-3,x+3,y+3],fill=(255,170,10))
    dr.text((8,8),'Blue: recovered 3D  Green: KITTI GT  Orange: input 2D corners',fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0))
    out.parent.mkdir(parents=True,exist_ok=True);im.save(out)



# ========================================================================
# NEW FRONT END: the input is always model-predicted or externally saved
# predicted 2D keypoints. Ground-truth keypoints are NOT input here.
# ========================================================================

def normalize_frame(stem):
    stem = Path(str(stem)).stem
    if stem.startswith("kitti_"):
        stem = stem[6:]
    return stem


def classes_for_gt(cls_id):
    return {
        0: {"Car", "Van", "Truck"},
        1: {"Pedestrian", "Person_sitting"},
        2: {"Cyclist"},
    }.get(cls_id, set())


def scan_images(kitti):
    folder = kitti / "image_2"
    if not folder.is_dir():
        raise FileNotFoundError(f"No KITTI image_2 directory: {folder}")
    return sorted([*folder.glob("*.png"), *folder.glob("*.jpg"), *folder.glob("*.jpeg")])


def get_model_records(result, classes, min_keypoint_conf):
    """Extract actual model prediction pixels, never input labels.

    One record per detected instance.
    """
    if result.boxes is None or result.keypoints is None:
        return []
    b = result.boxes
    k = result.keypoints
    if len(b) == 0:
        return []

    uv_all = k.xy.detach().cpu().numpy()
    xyxy = b.xyxy.detach().cpu().numpy()
    scores = b.conf.detach().cpu().numpy()
    cls_ids = b.cls.detach().cpu().numpy().astype(int)
    kconf = k.conf
    if kconf is not None:
        kconf = kconf.detach().cpu().numpy()

    if len(uv_all) != len(xyxy):
        raise ValueError("Mismatch in YOLO boxes and keypoints instance counts")
    records = []
    for j in range(len(xyxy)):
        if int(cls_ids[j]) not in classes:
            continue
        uv = np.asarray(uv_all[j], dtype=float)
        if uv.shape != (8, 2):
            raise ValueError(
                f"Expected 8 keypoints x2; model produced shape {uv.shape}. "
                "Check this is your trained 8-keypoint pose model."
            )
        if not np.isfinite(uv).all():
            continue
        if kconf is not None and np.any(kconf[j] < min_keypoint_conf):
            continue
        if min_keypoint_conf > 0 and kconf is None:
            # kpt_shape=[8,2] has no per-keypoint confidence; do not
            # fabricate confidence or reject the predictions.
            pass
        records.append({
            "pred_index": j,
            "class_id": int(cls_ids[j]),
            "bbox_xyxy": np.asarray(xyxy[j], dtype=float),
            "score": float(scores[j]),
            "uv": uv,
            "kpt_conf": None if kconf is None else kconf[j].tolist(),
        })
    return records


def get_saved_prediction_records(pred_json, frame, classes):
    """Optional predictions.json for already-ran YOLO outputs; no GT fields."""
    rows = pred_json.get(frame, [])
    records = []
    for j, d in enumerate(rows):
        cls_id = int(d["class_id"])
        if cls_id not in classes:
            continue
        uv = np.asarray(d["keypoints_xy"], float)
        bb = np.asarray(d["bbox_xyxy"], float)
        if uv.shape != (8, 2) or bb.shape != (4,) or not np.isfinite(uv).all():
            continue
        records.append({
            "pred_index": j,
            "class_id": cls_id,
            "bbox_xyxy": bb,
            "score": float(d.get("confidence", 1.0)),
            "uv": uv,
            "kpt_conf": d.get("keypoint_confidence"),
        })
    return records


def try_load_targets(kitti, frame):
    path = kitti / "label_2" / f"{frame}.txt"
    if not path.is_file():
        return []
    return load_kitti_label(path, {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"})


def associate_predictions_with_gt(pred_rows, targets, min_iou):
    """Strictly after 3D prediction, associate for evaluation only."""
    matched = {}
    if not targets or not pred_rows:
        return matched
    cost = np.ones((len(pred_rows), len(targets)))
    for i, pred in enumerate(pred_rows):
        valid_classes = classes_for_gt(pred["class_id"])
        for j, tgt in enumerate(targets):
            if tgt.name in valid_classes:
                cost[i, j] = 1.0 - box_iou2d(pred["bbox_xyxy"], tgt.box2d)
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        if 1.0 - cost[i, j] >= min_iou:
            matched[int(i)] = (int(j), float(1.0 - cost[i, j]))
    return matched


def record_for_json(rec):
    return {
        "det_index": rec["pred_index"],
        "class_id": rec["class_id"],
        "confidence": rec["score"],
        "bbox_xyxy": rec["bbox_xyxy"].tolist(),
        "keypoints_xy_predicted": rec["uv"].tolist(),
    }


def draw_prediction(image_path, recs, P, save_path, targets=None,
                    associations=None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = {"ok": (35, 145, 255), "failed": (250, 100, 60)}
    for idx, r in enumerate(recs):
        coords = r["input"]["keypoints_xy_predicted"]
        for kk, (u, v) in enumerate(coords):
            xy = (float(u), float(v))
            col = (255, 195, 0)
            rad = 3
            draw.ellipse((xy[0]-rad, xy[1]-rad, xy[0]+rad, xy[1]+rad),
                         fill=col)
            draw.text((xy[0]+3, xy[1]-2), str(kk), fill=col)
        x1, y1, x2, y2 = r["input"]["bbox_xyxy"]
        draw.rectangle((x1,y1,x2,y2), outline=(190,190,190), width=1)
        if r["status"] != "ok":
            draw.text((x1,max(0,y1-18)), "3D FAIL", fill=colors["failed"])
            continue
        xyz = np.asarray(r["pred_corners_camera_xyz_m"])
        try:
            xy = project(P, xyz)
            for a, b in EDGE_PAIRS:
                draw.line([tuple(xy[a]), tuple(xy[b])], fill=colors["ok"], width=3)
            draw.text((x1, max(0,y1-18)), "Predicted 3D", fill=colors["ok"])
        except ValueError:
            pass
        if targets is not None and associations and idx in associations:
            t_idx, _ = associations[idx]
            t = targets[t_idx]
            try:
                gt_uvw = project(P,corners(t.h,t.w,t.l,t.center,t.yaw))
                for a,b in EDGE_PAIRS:
                    draw.line([tuple(gt_uvw[a]),tuple(gt_uvw[b])],
                              fill=(20,215,75),width=2)
            except ValueError:
                pass
    draw.text((6,4), "BLUE: reconstructed from MODEL 2D / GREEN: KITTI GT (evaluation only)",
              fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0))
    save_path.parent.mkdir(parents=True,exist_ok=True)
    img.save(save_path)


def run_model_pipeline(args):
    kitti = locate_kitti(args.kitti_root)
    images = scan_images(kitti)

    # Evaluate on the same held-out KITTI frames used by the existing YOLO
    # dataset split, rather than accidentally evaluating training images.
    if args.split_image_dir is not None:
        if not args.split_image_dir.is_dir():
            raise FileNotFoundError(f"Split image directory not found: {args.split_image_dir}")
        allowed_stems = {
            normalize_frame(q.stem)
            for q in args.split_image_dir.iterdir()
            if q.suffix.lower() in (".png", ".jpg", ".jpeg")
            and (q.stem.startswith("kitti_") or q.stem.isdigit())
        }
        images = [q for q in images if normalize_frame(q.stem) in allowed_stems]
        print(f"Filtering to user dataset split: {args.split_image_dir} -> {len(images)} KITTI images", flush=True)

    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError("No KITTI images selected; check --kitti-root and --split-image-dir")

    if args.input_mode == "model":
        from ultralytics import YOLO
        if not args.weights.is_file():
            raise FileNotFoundError(f"Cannot find trained best.pt: {args.weights}")
        model = YOLO(str(args.weights))
        saved = None
        print("Inference model:", args.weights, flush=True)
    else:
        saved = json.loads(args.predictions_json.read_text(encoding="utf-8"))
        model = None
        print("Using SAVED MODEL PREDICTIONS (not 2D dataset labels):",
              args.predictions_json, flush=True)

    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    frames_json = []
    metrics = []
    skipped = {}
    nrec = 0
    success = 0
    image_saved = 0
    saved_pred_json = {}
    permitted = set(args.class_ids)

    def bump(s):
        skipped[s] = skipped.get(s,0)+1

    for fi,image_path in enumerate(images, start=1):
        frame = normalize_frame(image_path.stem)
        calib_path = kitti / "calib" / f"{frame}.txt"
        if not calib_path.is_file():
            bump("no_P2_calib")
            continue
        try:
            P = parse_calib(calib_path)
            if args.input_mode == "model":
                result = model.predict(
                    source=str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.nms_iou,
                    device=args.device,
                    verbose=False,
                    max_det=args.max_det,
                )[0]
                pred = get_model_records(result, permitted, args.min_kpt_conf)
            else:
                pred = get_saved_prediction_records(saved, frame, permitted)
            # Store actual model predictions for reproducible evaluation.
            saved_pred_json[frame] = [
                {
                    "class_id":p["class_id"],
                    "bbox_xyxy":p["bbox_xyxy"].tolist(),
                    "confidence":p["score"],
                    "keypoints_xy":p["uv"].tolist(),
                    "keypoint_confidence":p["kpt_conf"],
                }
                for p in pred
            ]
            # Predict every object *before* loading any ground truth label.
            records=[]
            for p in pred:
                nrec += 1
                item={"input":record_for_json(p),
                      "frame":frame,"status":"failed"}
                try:
                    rec, xyz, reproj = recover_box(
                        p["uv"],P,args.ground_y,
                        max_reproj_px=args.max_reproj_px
                    )
                    item.update({
                        "status":"ok",
                        "pred_corners_camera_xyz_m":xyz.tolist(),
                        "reprojection_rmse_px":reproj,
                        "pred_3d_box":{
                            "h":rec.h, "w":rec.w, "l":rec.l,
                            "location_bottom_center_xyz_m":rec.center.tolist(),
                            "yaw_about_camera_y_mod_pi_rad":rec.yaw,
                        }
                    })
                    success += 1
                except Exception as exc:
                    item["reason"] = str(exc)
                    bump(type(exc).__name__ + ": " + str(exc)[:70])
                records.append(item)

            gt = try_load_targets(kitti,frame) if args.evaluate else []
            associations = associate_predictions_with_gt(pred,gt,args.match_iou) if args.evaluate else {}

            # GT is used only here: *after* prediction and reconstruction.
            if args.evaluate:
                for i, (t_idx, ov) in associations.items():
                    r = records[i]
                    if r["status"] != "ok":
                        continue
                    tgt = gt[t_idx]
                    pred3d = r["pred_3d_box"]
                    b = Cuboid("pred",
                               pred3d["h"], pred3d["w"], pred3d["l"],
                               np.asarray(pred3d["location_bottom_center_xyz_m"]),
                               pred3d["yaw_about_camera_y_mod_pi_rad"],
                               pred[i]["bbox_xyxy"],pred[i]["pred_index"])
                    gt3 = corners(tgt.h,tgt.w,tgt.l,tgt.center,tgt.yaw)
                    xyz = np.asarray(r["pred_corners_camera_xyz_m"])
                    err = b.center - tgt.center
                    metrics.append({
                        "frame":frame,
                        "det_index":pred[i]["pred_index"],
                        "class_id":pred[i]["class_id"],
                        "confidence":pred[i]["score"],
                        "gt_type":tgt.name,
                        "gt_line_index":tgt.index,
                        "match_iou_2d":ov,
                        "reproj_rmse_px":r["reprojection_rmse_px"],
                        "iou3d":iou3d(b,tgt),
                        "center_error_m":float(np.linalg.norm(err)),
                        "depth_z_error_m":float(err[2]),
                        "corner_rmse_m":corner_rmse(xyz,gt3),
                        "height_error_m":float(b.h-tgt.h),
                        "width_error_m":float(b.w-tgt.w),
                        "length_error_m":float(b.l-tgt.l),
                        "yaw_error_mod_pi_rad":angle_err_mod_pi(b.yaw,tgt.yaw)
                    })
                    r["evaluation_only"]={
                        "matched_gt_line_index":tgt.index,
                        "match_2d_iou":ov,
                    }

            if args.save_images > image_saved and records:
                path = out / "visualizations" / f"{frame}.png"
                draw_prediction(image_path,records,P,path,
                                gt if args.evaluate else None,associations)
                image_saved += 1

            frames_json.append({
                "frame":frame,
                "image_path":str(image_path),
                "P2":P.tolist(),
                "ground_y_assumed_m":args.ground_y,
                "detections":records,
            })

        except Exception as exc:
            bump("frame_error_"+type(exc).__name__)
            print(f"[WARN] {frame}: {exc}",flush=True)

        if fi % args.log_every == 0 or fi == len(images):
            print(
                f"[{fi}/{len(images)}] predicted_2d={nrec} "
                f"recovered_3d={success} evaluated={len(metrics)}",
                flush=True
            )

    # All exported predictions are genuine model outputs (or explicitly
    # provided saved model outputs). Never exports training GT as predictions.
    (out/"predicted_2d_keypoints.json").write_text(
        json.dumps(saved_pred_json,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    (out/"recovered_3d_from_model.json").write_text(
        json.dumps(frames_json,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    if metrics:
        with (out/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)

    summary={
        "input_mode":args.input_mode,
        "weights":str(args.weights) if args.input_mode=="model" else None,
        "source_images":len(images),
        "frames_with_predictions":sum(bool(p) for p in saved_pred_json.values()),
        "num_predicted_instances":nrec,
        "num_3d_success":success,
        "num_eval_matches":len(metrics),
        "ground_y_assumed_m":args.ground_y,
        "note":"True 2D input: YOLO11-Pose predicted 8 corners. "
               "Original KITTI label_2 not accessed by reconstruction. "
               "Evaluation accesses GT only after reconstruction. "
               "Metric ground y is an approximation and errors are expected.",
        "skipped":skipped,
        "metrics":{},
    }
    for key in ("iou3d","center_error_m","depth_z_error_m",
                "corner_rmse_m","reproj_rmse_px","height_error_m"):
        vals = np.asarray([row[key] for row in metrics],float)
        if len(vals):
            summary["metrics"][key] = {
                "median":float(np.median(vals)),
                "mean":float(vals.mean()),
                "p90":float(np.percentile(vals,90)),
            }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    print("\n" + json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("Output directory:",out,flush=True)


def self_test():
    # Synthetic rectified KITTI-like camera P2 (with a real fourth column).
    K=np.array([[700.,0.,620.],[0.,700.,180.],[0.,0.,1.]])
    P=np.c_[K,np.array([-350.,0.,0.])]
    # Exact ground plane, exact 2D corners -> should perfectly reconstruct geometry.
    gt=Cuboid('Car',1.55,1.8,4.4,np.array([2.0,1.65,19.0]),0.37,np.zeros(4),0)
    uv=project(P,corners(gt.h,gt.w,gt.l,gt.center,gt.yaw))
    rec,xyz,rms=recover_box(uv,P,1.65)
    err=corner_rmse(xyz,corners(gt.h,gt.w,gt.l,gt.center,gt.yaw))
    overlap=iou3d(rec,gt)
    print(f'SELFTEST rmse_px={rms:.8g} corner_rmse_m={err:.8g} 3d_iou={overlap:.8g}')
    assert rms<1e-4 and err<1e-3 and overlap>0.999,'Synthetic self test failed'
    # Verify changing ground height does change recovered metric scale.
    wrong,_,_=recover_box(uv,P,1.8)
    assert abs(wrong.h-gt.h)>.02, 'Ground height sensitivity test failed'
    print('SELFTEST PASSED (also confirmed estimated 3D scale depends on assumed ground height).')




def main():
    ap = argparse.ArgumentParser(description="YOLO11 pose predicted keypoints -> KITTI 3D boxes")
    ap.add_argument("--self-test",action="store_true",help="Run analytic 3D geometry self-test")
    ap.add_argument("--input-mode",choices=["model","predictions-json"],default="model")
    ap.add_argument("--weights",type=Path,default=Path("runs/pose/train-3/weights/best.pt"))
    ap.add_argument("--predictions-json",type=Path,default=None,
                    help="Previously exported YOLO predictions; never ground-truth labels")
    ap.add_argument("--kitti-root",type=Path,
                    default=Path("/mnt/sdb/lys/MonoCoP/data/KITTIDataset"))
    ap.add_argument("--split-image-dir",type=Path,default=None,
                    help="Optional user's YOLO dataset images/test folder; only those KITTI frames are used.")
    ap.add_argument("--ground-y",type=float,default=1.65,
                    help="Assumed KITTI camera Y-down local road plane (metres).")
    ap.add_argument("--class-ids",type=int,nargs="+",default=[0],
                    help="Model class IDs to reconstruct, default 0 (Car)")
    ap.add_argument("--conf",type=float,default=0.25)
    ap.add_argument("--nms-iou",type=float,default=0.7)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    ap.add_argument("--min-kpt-conf",type=float,default=0.0,
                    help="Only meaningful for kpt_shape=[8,3].")
    ap.add_argument("--max-det",type=int,default=30)
    ap.add_argument("--max-reproj-px",type=float,default=30.0)
    ap.add_argument("--match-iou",type=float,default=0.25)
    ap.add_argument("--evaluate",action="store_true",
                    help="Use KITTI label_2 ONLY AFTER reconstruction for evaluation")
    ap.add_argument("--limit",type=int,default=100,
                    help="Number of images, 0 to process all")
    ap.add_argument("--save-images",type=int,default=20)
    ap.add_argument("--log-every",type=int,default=20)
    ap.add_argument("--out",type=Path,
                    default=Path("runs/kitti_yolo11_predicted_3d"))
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.input_mode=="predictions-json" and args.predictions_json is None:
        ap.error("--predictions-json required in predictions-json mode")
    if args.ground_y <= 0:
        ap.error("--ground-y must be positive in KITTI camera Y-down coordinates")
    run_model_pipeline(args)

if __name__ == "__main__":
    main()
