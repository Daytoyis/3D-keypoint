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


def run(args):
    kroot=locate_kitti(args.kitti_root)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    gt_files=sorted((kroot/'label_2').glob('*.txt'))
    if args.limit>0:gt_files=gt_files[:args.limit]
    if args.source=='labels' and args.pose_root is None:
        raise ValueError('--pose-root is required when --source labels')
    csv_rows=[];pred_records=[];skipped={};image_count=0
    classes={args.kitti_class}
    for n,lab in enumerate(gt_files,1):
        stem=lab.stem
        try:
            P=parse_calib(kroot/'calib'/f'{stem}.txt')
            targets=load_kitti_label(lab,classes)
            if not targets:continue
            image=kroot/'image_2'/f'{stem}.png'
            if not image.is_file():image=kroot/'image_2'/f'{stem}.jpg'
            # KITTI source images normally have width=1242/1224 and height=375.
            if not image.is_file():
                skipped['no_source_image'] = skipped.get('no_source_image',0)+1
                continue
            with Image.open(image) as im:
                W,H=im.size
            inputs=[]
            if args.source=='gt':
                for tgt in targets:
                    uv=project(P,corners(tgt.h,tgt.w,tgt.l,tgt.center,tgt.yaw))
                    bb=np.array([uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()])
                    inputs.append((tgt.index,uv,bb))
                assignment=[(i,i) for i in range(len(inputs))]
            else:
                txt=search_pose_label(args.pose_root,stem)
                if txt is None:
                    skipped['no_pose_label'] = skipped.get('no_pose_label',0)+1;continue
                inputs=get_yolo_records(txt,W,H,args.class_id)
                if not inputs:continue
                # Matching uses KITTI GT only to associate each 2D-label instance with
                # an evaluation object. The 3D recovery solver never sees the GT.
                reference_boxes=[]
                for tgt in targets:
                    try:
                        pg=project(P,corners(tgt.h,tgt.w,tgt.l,tgt.center,tgt.yaw))
                        projected_gt_box=np.array([pg[:,0].min(),pg[:,1].min(),
                                                   pg[:,0].max(),pg[:,1].max()])
                    except ValueError:
                        projected_gt_box=tgt.box2d
                    reference_boxes.append((tgt.box2d,projected_gt_box))
                cost=np.array([[1-max(box_iou2d(r[2],bb2d),box_iou2d(r[2],bbprojected))
                                for bb2d,bbprojected in reference_boxes]
                                for r in inputs])
                rix,tix=linear_sum_assignment(cost)
                assignment=[(int(i),int(j)) for i,j in zip(rix,tix)
                            if 1-cost[i,j]>=args.match_iou]
            for ai,ti in assignment:
                tgt=targets[ti];idx,uv,_=inputs[ai]
                try:
                    pred,xyz,rmse=recover_box(uv,P,args.ground_y,args.max_reproj_px)
                    gt_xyz=corners(tgt.h,tgt.w,tgt.l,tgt.center,tgt.yaw)
                    errors=pred.center-tgt.center
                    row={
                        'frame':stem,'gt_line_index':tgt.index,'2d_line_index':idx,
                        'class':tgt.name,'input_source':args.source,'ground_y_assumed_m':args.ground_y,
                        'reprojection_rmse_px':rmse,'center_error_m':float(np.linalg.norm(errors)),
                        'center_error_x_m':float(errors[0]),'center_error_y_m':float(errors[1]),
                        'center_error_z_m':float(errors[2]),
                        'h_error_m':pred.h-tgt.h,'w_error_m':pred.w-tgt.w,'l_error_m':pred.l-tgt.l,
                        'corner_rmse_m':corner_rmse(xyz,gt_xyz),
                        'yaw_error_mod_pi_rad':angle_err_mod_pi(pred.yaw,tgt.yaw),
                        'iou3d':iou3d(pred,tgt),
                        'pred_h_m':pred.h,'pred_w_m':pred.w,'pred_l_m':pred.l,
                        'pred_x_m':pred.center[0],'pred_y_m':pred.center[1],
                        'pred_z_m':pred.center[2],'pred_yaw_rad':pred.yaw,
                        'gt_h_m':tgt.h,'gt_w_m':tgt.w,'gt_l_m':tgt.l,
                        'gt_x_m':tgt.center[0],'gt_y_m':tgt.center[1],
                        'gt_z_m':tgt.center[2], 'gt_yaw_rad':tgt.yaw,
                    }
                    csv_rows.append(row)
                    pred_records.append({'frame':stem,'gt_line_index':tgt.index,
                                         'input_2d_uv':uv.tolist(),
                                         'pred_corners_camera_xyz_m':xyz.tolist(),
                                         'pred_3d_box':{'h':pred.h,'w':pred.w,'l':pred.l,
                                                        'location_bottom_center':pred.center.tolist(),
                                                        'rotation_y_mod_pi':pred.yaw},
                                         'reproj_rmse_px':rmse})
                    if image_count<args.save_images:
                        visualize(image,uv,xyz,P,tgt,out/'images'/f'{stem}_{tgt.index:02d}.png')
                        image_count+=1
                except Exception as exc:
                    key=str(exc).split(':')[0][:70]
                    skipped[key]=skipped.get(key,0)+1
        except Exception as exc:
            key='frame_error:'+type(exc).__name__
            skipped[key]=skipped.get(key,0)+1
        if n%100==0:print(f'frames {n}/{len(gt_files)}; recovered={len(csv_rows)}; skipped={sum(skipped.values())}',flush=True)
    if csv_rows:
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=csv_rows[0].keys());w.writeheader();w.writerows(csv_rows)
    (out/'recovered_3d.json').write_text(json.dumps(pred_records,indent=2,ensure_ascii=False),encoding='utf-8')
    keys=['reprojection_rmse_px','center_error_m','corner_rmse_m','yaw_error_mod_pi_rad','iou3d',
          'h_error_m','w_error_m','l_error_m','center_error_z_m']
    summary={'source':args.source,'ground_y_assumed_m':args.ground_y,
             'frames_examined':len(gt_files),'success':len(csv_rows),
             'skipped':skipped,'evaluation_warning':(
                 'Ground plane is a constant approximation (not measured per KITTI frame); '
                 'labels source is 2D GT derived from KITTI 3D annotations, not an independent model prediction. '
                 'The method cannot determine 180-degree front/back yaw from a symmetric cuboid.'
             ),'metrics':{}}
    for key in keys:
        values=np.array([r[key] for r in csv_rows],float)
        if len(values):summary['metrics'][key]={'mean':float(values.mean()),
                  'median':float(np.median(values)),'p90':float(np.percentile(values,90))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nFINISHED\n'+json.dumps(summary,indent=2),flush=True)
    print('Output:',out)


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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kitti-root',type=Path,default=Path('/mnt/sdb/lys/MonoCoP/data/KITTIDataset'))
    p.add_argument('--pose-root',type=Path,default=None,help='Dataset containing labels/train|val|test/kitti_*.txt')
    p.add_argument('--source',choices=['labels','gt'],default='labels')
    p.add_argument('--ground-y',type=float,default=1.65,
                   help='Approx. camera height above local ground in metres, camera Y down. Not per-frame ground truth.')
    p.add_argument('--kitti-class',choices=['Car','Van','Truck','Pedestrian','Cyclist'],default='Car')
    p.add_argument('--class-id',type=int,default=0,help='YOLO class ID in pose .txt for the chosen KITTI class')
    p.add_argument('--match-iou',type=float,default=0.25)
    p.add_argument('--max-reproj-px',type=float,default=60.0)
    p.add_argument('--limit',type=int,default=100,help='Number of KITTI training frames; 0 = all')
    p.add_argument('--save-images',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('runs/kitti_2d8_to_3d_ground'))
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test();return
    run(args)

if __name__=='__main__':
    main()
