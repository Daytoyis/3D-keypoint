#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KITTI monocular 8x2 -> metric 3D cuboids, Scheme C1.

Uses the already-saved *YOLO11-Pose predictions*, per-frame KITTI P2, and
independent class-size priors. Optional per-frame road-plane consistency is
estimated from image-derived 3D bottom centers; it is NOT observed GT road.
No LiDAR, no GT 2D/3D input, no training data rewrites, no model inference.
GT label_2 is read only after recovery, if --evaluate is enabled.

Fundamental caveat: metric scale is UNOBSERVABLE from 8 2D points and P2
without an external prior. A prior vehicle height anchors the scale here.
KITTI labels are yaw-only upright cuboids, so fitted boxes also use this model.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

# 0..3 cyclic bottom, 4..7 corresponding top; the orientation / starting corner
# is selected per object among 8 cyclic/reflected variants (no unverified label convention).
SIGNS = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
DEFAULT_PRIORS = {
    # These are INFERENCE assumptions, NOT values read from KITTI annotations.
    0: (1.55, .18, 1.80, .30, 4.20, .90),  # car h,sh,w,sw,l,sl [meters]
    1: (1.70, .26, .65, .35, .65, .35),   # person: exploratory only
    2: (1.65, .30, .75, .35, 1.80, .65),  # cyclist: exploratory only
}
GT_CLASSES = {0: {'Car', 'Van', 'Truck'}, 1: {'Pedestrian'}, 2: {'Cyclist'}}


@dataclass
class Box:
    h: float
    w: float
    l: float
    center: np.ndarray  # bottom center, KITTI camera (X right,Y down,Z forward)
    yaw: float


def parse_p2(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"No P2 calibration: {path}")
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip().startswith('P2:'):
            a = np.fromstring(line.split(':', 1)[1], sep=' ', dtype=float)
            if a.size != 12:
                break
            P = a.reshape(3, 4)
            if abs(np.linalg.det(P[:, :3])) > 1.e-9:
                return P
    raise ValueError(f"Invalid KITTI P2 calibration: {path}")


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    q = np.c_[xyz, np.ones(len(xyz))] @ P.T
    if np.any(q[:, 2] < .1) or not np.isfinite(q).all():
        raise ValueError("Point behind/near camera")
    return q[:, :2] / q[:, 2, None]


def corners(h: float, w: float, l: float, ctr: np.ndarray,
            yaw: float, signs=SIGNS) -> np.ndarray:
    s = np.asarray(signs, float)
    xyz = np.c_[s[:, 0]*l/2., np.zeros(4), s[:, 1]*w/2.]
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0., sn], [0., 1., 0.], [-sn, 0., c]])
    base = xyz @ R.T + np.asarray(ctr, float)
    return np.r_[base, base-np.array([0., h, 0.])]


def sign_variants():
    for rev in (False, True):
        base = SIGNS[::-1] if rev else SIGNS
        for k in range(4):
            yield np.roll(base, k, axis=0).copy()


def backproject_fixed_denominator(P, uv, denom):
    # For P2 rows (including nonzero translation) solve P2*[X,Y,Z,1]^T
    # = denom*[u,v,1]^T. KITTI P2 row3=(0,0,1,tz).
    rhs = np.array([uv[0]*denom, uv[1]*denom, denom])-P[:, 3]
    return np.linalg.solve(P[:, :3], rhs)


def initial_bottom(P: np.ndarray, uv: np.ndarray, assumed_h: float):
    """Depth from top-bottom pixel gap and a class HEIGHT prior, not road Y.

    For KITTI upright boxes and rectified P2, vertical edge endpoints share Z,
    so dv=fy*height/(Z+P2[2,3]). Model noise is handled with clipping and median.
    """
    if abs(P[2, 0]) > 1e-7 or abs(P[2, 1]) > 1e-7 or abs(P[2, 2]-1) > 1e-5:
        raise ValueError("Non-standard KITTI P2 third row; height initialization unsupported")
    fy = float(P[1, 1])
    if fy <= 0:
        raise ValueError('Invalid P2 focal length')
    dv = uv[:4, 1] - uv[4:, 1]
    valid = np.isfinite(dv) & (dv > 1.5)
    if valid.sum() < 3:
        raise ValueError(f'Too few vertical keypoint pairs with positive dv: {valid.sum()}')
    denom = np.full(4, np.nan)
    denom[valid] = fy*assumed_h/dv[valid]
    med = float(np.nanmedian(denom))
    # Low-quality/collapsed edges can give excessive depth: do not let them
    # determine initial yaw and dimensions.
    denom[~valid] = med
    denom = np.clip(denom, max(2.0, med*.55), min(220., med*1.6))
    bottom = np.array([backproject_fixed_denominator(P, uv[j], denom[j]) for j in range(4)])
    return bottom, med


def wrap_pi(yaw):
    return float((yaw+np.pi)%(2*np.pi)-np.pi)


def fit_one(uv, P, prior, px_sigma, max_reproj, plane=None, plane_sigma=.28):
    uv = np.asarray(uv, float)
    if uv.shape != (8, 2) or not np.isfinite(uv).all():
        raise ValueError('Expected 8 finite predicted (u,v) pairs')
    hm, hs, wm, ws, lm, ls = prior
    bottom, approx_depth = initial_bottom(P, uv, hm)
    center_init = bottom.mean(axis=0)
    center_init[2] = float(np.clip(center_init[2], 1.5, 220))
    # parameter order: bottom X,Y,Z; yaw; l,w,h
    low = np.array([-250, -15., 1.5, -4*np.pi, .3, .2, .3], float)
    high = np.array([250, 20., 220., 4*np.pi, 20., 10., 6.0], float)
    all_cands = []
    for signs in sign_variants():
        plane_points = bottom[:, [0,2]]
        bcenter = plane_points.mean(axis=0)
        u_vec = (signs[:, 0, None]*(plane_points-bcenter)).mean(axis=0)
        v_vec = (signs[:, 1, None]*(plane_points-bcenter)).mean(axis=0)
        yaw0 = math.atan2(-u_vec[1], u_vec[0])
        l0 = 2*np.linalg.norm(u_vec)
        w0 = 2*np.linalg.norm(v_vec)
        # When box footprint is nearly collapsed, lean on class-size priors.
        if l0 < .5: l0 = lm
        if w0 < .3: w0 = wm
        p0 = np.array([center_init[0], center_init[1], center_init[2],
                       yaw0, l0, w0, hm])
        p0 = np.clip(p0, low+1e-5, high-1e-5)

        def residual(p):
            x,y,z,yaw,l,w,h = p
            if z <= 1:
                pix = np.full(16, 1.e3)
            else:
                try:
                    pix = ((project(P, corners(h,w,l,[x,y,z],yaw,signs))-uv)/px_sigma).ravel()
                except ValueError:
                    pix = np.full(16, 1.e3)
            # Gaussian independent metric size priors. Not GT label_2 data.
            prior_r = np.array([(h-hm)/hs, (w-wm)/ws, (l-lm)/ls])
            if plane is None:
                return np.r_[pix, prior_r]
            # Y = yref + ax*(X-xref) + az*(Z-zref).
            ground_y = plane['yref'] + plane['ax']*(x-plane['xref']) + plane['az']*(z-plane['zref'])
            return np.r_[pix, prior_r, (y-ground_y)/plane_sigma]

        try:
            opt = least_squares(residual, p0, bounds=(low,high),
                                loss='soft_l1', f_scale=2.5, max_nfev=120)
            x,y,z,yaw,l,w,h = opt.x
            xyz = corners(h,w,l,np.array([x,y,z]),yaw,signs)
            rms = float(np.sqrt(np.mean((project(P,xyz)-uv)**2)))
            # Score all sign variants using the same prior objective; don't
            # select solely by reprojection, because 90deg swaps are ambiguous.
            objective = float(np.sum(residual(opt.x)**2))
            # If l<w, swap physically equivalent principal axes to standardize.
            if l < w:
                l,w = w,l
                yaw -= np.pi/2
            box = Box(float(h),float(w),float(l),np.array([x,y,z],float),wrap_pi(yaw))
            all_cands.append((objective,rms,box,xyz))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if not all_cands:
        raise ValueError('Optimization failed for every cyclic keypoint assignment')
    all_cands.sort(key=lambda p: p[0])
    obj, rms, box, xyz = all_cands[0]
    if rms > max_reproj:
        raise ValueError(f'Reprojection RMSE {rms:.2f}px > {max_reproj}px')
    return box, xyz, rms, obj, approx_depth


def estimate_frame_road(initial_rows, max_slope=.08):
    """Estimate a weak common road from predicted contact centers, NO GT.

    Multiple independent vehicles with enough spatial span are needed to
    identify slope. If not enough evidence, use flat reference at median
    inferred Y; do not pretend a true camera-to-road extrinsic is known.
    """
    if not initial_rows:
        return None
    xyz = np.array([r['box'].center for r in initial_rows], float)
    # Anchor ground in image-inferred metric coordinates, not fixed 1.65m.
    xref, yref, zref = np.median(xyz,axis=0)
    X = xyz[:, 0]-xref
    Z = xyz[:, 2]-zref
    ax,az=0.,0.
    reliable = len(xyz)>=4 and np.ptp(xyz[:, 2])>=8. and np.ptp(xyz[:, 0])>=3.
    if reliable:
        # Limit overfitting to depth error from noisy keypoints / uncertain car height.
        def r_plane(v):
            a,b,c = v
            return np.r_[((c+a*X+b*Z)-xyz[:,1])/.28,
                          a/.035, b/.035]
        opt=least_squares(r_plane, [0.,0.,yref],
                          bounds=([-max_slope,-max_slope,-15.],
                                  [max_slope,max_slope,20.]),
                          loss='soft_l1',f_scale=1.0,max_nfev=120)
        ax,az,yref=map(float,opt.x)
    n = np.array([-ax, 1., -az], float)
    n/=np.linalg.norm(n)
    plane_d = float(n @ np.array([xref,yref,zref]))
    return {'ax':float(ax), 'az':float(az), 'xref':float(xref),
            'yref':float(yref), 'zref':float(zref), 'normal_xyz':n.tolist(),
            'normal_dot_xyz_m':plane_d,
            'slope_fit':bool(reliable), 'num_initial_boxes':len(xyz)}


def locate_kitti(root):
    for b in (root,root/'training',root/'object'/'training'):
        if (b/'calib').is_dir() and (b/'image_2').is_dir():
            return b
    raise FileNotFoundError(f'No KITTI image_2/calib under {root}')


def find_image(kitti, stem):
    for sfx in ('.png','.jpg','.jpeg'):
        p=kitti/'image_2'/f'{stem}{sfx}'
        if p.is_file(): return p
    return None


def normalize_frame(name):
    s=Path(name).stem
    return s[6:] if s.startswith('kitti_') else s


def get_3source_splits(root):
    """Audit whether evaluated images belong to training/val/test in 3Source."""
    paths={}
    for split in ('train','val','test'):
        folder=root/'images'/split
        if folder.is_dir():
            for p in folder.iterdir():
                if p.suffix.lower() not in ('.png','.jpg','.jpeg'):
                    continue
                if p.stem.startswith('kitti_'):
                    paths.setdefault(normalize_frame(p.stem),set()).add(split)
    return paths


def read_gt(path, cls_ids):
    # Called strictly AFTER both reconstruction passes; evaluation only.
    if not path.is_file():return []
    out=[]
    allowed=set().union(*(GT_CLASSES.get(i,set()) for i in cls_ids))
    for idx,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        f=line.split()
        if len(f)<15 or f[0] not in allowed:continue
        try:
            h,w,l=map(float,f[8:11])
            pos=np.array(list(map(float,f[11:14])),float)
            yaw=float(f[14]); bbox=np.array(list(map(float,f[4:8])),float)
            if min(h,w,l)>0 and pos[2]>0:
                out.append((idx,f[0],Box(h,w,l,pos,yaw),bbox))
        except ValueError:pass
    return out


def iou2d(a,b):
    a=np.asarray(a);b=np.asarray(b)
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    ar=lambda q:max(0.,q[2]-q[0])*max(0.,q[3]-q[1])
    return inter/max(ar(a)+ar(b)-inter,1e-10)


def signed_area(poly):
    if len(poly)<3:return 0.
    return float(.5*np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1)))


def footprint(b):
    c,s=math.cos(b.yaw), math.sin(b.yaw)
    ux=np.array([c,-s]);uz=np.array([s,c]);ctr=b.center[[0,2]]
    p=np.array([ctr+a*b.l/2*ux+d*b.w/2*uz for a,d in ((1,1),(-1,1),(-1,-1),(1,-1))])
    return p if signed_area(p)>=0 else p[::-1]


def clip_polygon(subject, clip):
    out=np.asarray(subject,float)
    for i in range(len(clip)):
        if not len(out):break
        a,b=clip[i],clip[(i+1)%len(clip)]
        def inside(p):
            q=b-a;v=p-a
            return q[0]*v[1]-q[1]*v[0]>=-1.e-9
        def intersect(p,q):
            r=q-p;s=b-a
            den=r[0]*s[1]-r[1]*s[0]
            if abs(den)<1e-12:return q
            t=((a[0]-p[0])*s[1]-(a[1]-p[1])*s[0])/den
            return p+t*r
        result=[]; prev=out[-1]
        for curr in out:
            if inside(curr):
                if not inside(prev):result.append(intersect(prev,curr))
                result.append(curr)
            elif inside(prev):result.append(intersect(prev,curr))
            prev=curr
        out=np.array(result,float).reshape(-1,2)
    return out


def iou3d(a,b):
    overlap=abs(signed_area(clip_polygon(footprint(a),footprint(b))))
    bot=min(a.center[1],b.center[1]);top=max(a.center[1]-a.h,b.center[1]-b.h)
    inter=overlap*max(0.,bot-top)
    va=a.h*a.w*a.l;vb=b.h*b.w*b.l
    return float(np.clip(inter/max(1e-9,va+vb-inter),0,1))


def corner_error(pred_xyz, gt_xyz, ordered=False):
    if ordered:
        return float(np.sqrt(np.mean(np.sum((pred_xyz-gt_xyz)**2,axis=1))))
    d=[]
    for i0 in (0,4):
        cost=np.linalg.norm(pred_xyz[i0:i0+4,None,:]-gt_xyz[None,i0:i0+4,:],axis=2)
        ii,jj=linear_sum_assignment(cost)
        d.extend(cost[ii,jj]**2)
    return float(np.sqrt(np.mean(d)))


def assign_gt(detections, gt, min_iou):
    if not detections or not gt:return {}
    cost=np.full((len(detections),len(gt)),1e3)
    for i,d in enumerate(detections):
        for j,t in enumerate(gt):
            if t[1] in GT_CLASSES.get(d['class_id'],set()):
                cost[i,j]=1.-iou2d(d['bbox_xyxy'],t[3])
    ii,jj=linear_sum_assignment(cost)
    return {int(i):(int(j),float(1.-cost[i,j])) for i,j in zip(ii,jj)
            if cost[i,j]<=1.-min_iou}


def compare_paired_old_new(old_path, new_metrics, out):
    """Compare only identical frame+det+GT rows, not incomparable cohorts."""
    if not old_path.is_file() or not new_metrics:
        return None
    with old_path.open(encoding='utf-8-sig',newline='') as f:
        oldrows=list(csv.DictReader(f))
    old={ (r.get('frame'),r.get('det_index'),r.get('gt_line_index')):r
          for r in oldrows }
    pairs=[]
    for row in new_metrics:
        key=(row['frame'],str(row['det_index']),str(row['gt_line_index']))
        if key not in old:continue
        o=old[key]
        pair={'frame':row['frame'],'det_index':row['det_index'],
              'gt_line_index':row['gt_line_index']}
        for m in ('iou3d','center_error_m','depth_z_error_m','corner_rmse_m','reproj_rmse_px'):
            pair['baseline_'+m]=float(o[m])
            pair['new_'+m]=float(row[m])
        pairs.append(pair)
    if not pairs:
        return {'matched_common_pairs':0,'note':'No identical baseline and new GT matches.'}
    with (out/'comparison_common_matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(pairs[0].keys()));wr.writeheader();wr.writerows(pairs)
    stats={'matched_common_pairs':len(pairs)}
    for m in ('iou3d','center_error_m','depth_z_error_m','corner_rmse_m','reproj_rmse_px'):
        a=np.array([row['baseline_'+m] for row in pairs]);b=np.array([row['new_'+m] for row in pairs])
        stats[m]={'old_median':float(np.median(a)),'new_median':float(np.median(b)),
                  'old_mean':float(np.mean(a)),'new_mean':float(np.mean(b))}
    stats['CAUTION']='Same detection+GT rows only; cohort may contain train images unless split filtered.'
    (out/'comparison_common_matches.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def draw_frame(image_path,P,rows,out_path,gt=None,assoc=None):
    im=Image.open(image_path).convert('RGB'); dr=ImageDraw.Draw(im)
    for i, r in enumerate(rows):
        pts=np.asarray(r['uv'],float)
        for k,(u,v) in enumerate(pts):
            dr.ellipse([u-3,v-3,u+3,v+3],fill=(255,177,0))
            dr.text((u+2,v-7),str(k),fill=(255,230,5))
        if r.get('final') is None:continue
        try:
            q=project(P,r['final']['xyz'])
            for a,b in EDGES:dr.line([tuple(q[a]),tuple(q[b])],fill=(10,140,250),width=2)
            if gt is not None and assoc and i in assoc:
                target=gt[assoc[i][0]][2]
                gtq=project(P,corners(target.h,target.w,target.l,target.center,target.yaw))
                for a,b in EDGES:dr.line([tuple(gtq[a]),tuple(gtq[b])],fill=(30,240,70),width=2)
        except ValueError:pass
    dr.text((5,5),'ORANGE predicted 2D  BLUE new 3D  GREEN eval GT',
            fill='white',stroke_width=2,stroke_fill='black')
    out_path.parent.mkdir(parents=True,exist_ok=True)
    im.save(out_path)


def serialize(box):
    return {'h':box.h,'w':box.w,'l':box.l,
            'location_bottom_center_xyz_m':box.center.tolist(),
            'yaw_about_camera_y_rad':box.yaw}


def make_prior(cls,args):
    prior=list(DEFAULT_PRIORS[cls])
    if cls==0:
        prior=[args.car_height,args.car_height_sigma,args.car_width,
               args.car_width_sigma,args.car_length,args.car_length_sigma]
    if min(prior)>0:return tuple(prior)
    raise ValueError('All physical size prior values and sigmas must be >0')


def self_test():
    K=np.array([[700.,0.,620.],[0.,700.,180.],[0.,0.,1.]])
    P=np.c_[K,np.array([-350.,-.2,.004])]
    cases=[
        Box(1.55,1.80,4.2,np.array([2.0,1.50,19.]),.37),
        Box(1.42,1.72,3.95,np.array([-1.5,1.8,36.]),-.61),
        Box(1.7,1.90,4.65,np.array([4.,1.35,12.]),1.25),
    ]
    for k,b in enumerate(cases):
        true_xyz=corners(b.h,b.w,b.l,b.center,b.yaw)
        uv=project(P,true_xyz)
        prior=DEFAULT_PRIORS[0]
        rec,xyz,rms,_,_=fit_one(uv,P,prior,2.0,12.)
        overlap=iou3d(rec,b)
        err=np.linalg.norm(rec.center-b.center)
        print(f"[{'SELF TEST' if k==0 else 'PRIOR SENSITIVITY'} {k}] reproj={rms:.4f}px, IoU3D={overlap:.4f}, center={err:.3f}m, h={rec.h:.3f}m")
        if k==0:
            assert rms<.2 and overlap>.97 and err<.10, 'Exact-prior geometry test failed'
        assert rms < .2 and np.isfinite(overlap) and np.isfinite(err), 'Numerical metric recovery failed'
    # Perspective scale ambiguity: any global scale around camera center yields
    # the EXACT same pixel projections, hence an independent prior is required.
    C=-np.linalg.solve(P[:,:3],P[:,3]);x0=corners(1.55,1.8,4.2,[2,1.5,19],.37)
    x1=C+1.3*(x0-C)
    assert np.max(np.abs(project(P,x0)-project(P,x1)))<1.e-9
    print('SCALE AMBIGUITY VERIFIED: 1.3x scaled 3D -> identical 2D pixels.')
    # Frame road fit from several car anchors, without GT ground.
    samples=[]
    for xx,zz in ((-5.,12.),(-1.,18.),(3.,26.),(9.,34.),(-7.,42.)):
        yy=1.5+.015*xx+.02*(zz-18.)
        car=Box(1.55,1.8,4.2,np.array([xx,yy,zz]),.3)
        im_uv=project(P,corners(car.h,car.w,car.l,car.center,car.yaw))
        recovered,_,_,_,_=fit_one(im_uv,P,DEFAULT_PRIORS[0],2.,12.)
        samples.append({'box':recovered})
    road=estimate_frame_road(samples)
    assert road is not None and road['slope_fit']
    assert abs(road['ax']-.015)<.02 and abs(road['az']-.02)<.02
    print(f"FRAME ROAD SELFTEST ax={road['ax']:.4f}, az={road['az']:.4f}, pass")
    print('SELF-TEST PASSED; note height prior biases dimensions for other-sized vehicles.')



def load_new_prediction_payload(path: Path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Prediction JSON root must be a dict')
    split_payload = payload.get('split')
    if not isinstance(split_payload, dict) or not isinstance(split_payload.get('images'), list):
        raise ValueError('Expected new inference schema: {metadata:..., split:{images:[...]}}')
    return split_payload


def aggregate_metric(rows, key):
    vals = np.asarray([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
    if not len(vals):
        return None
    return {'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
            'p90': float(np.percentile(vals, 90))}


def main():
    ap = argparse.ArgumentParser(
        description='Recover KITTI Car metric 3D boxes from new YOLO11 8x2 predictions.'
    )
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--kitti-root', type=Path,
                    default=Path('/mnt/sdb/lys/MonoCoP/data/KITTIDataset'))
    ap.add_argument('--dataset-root', type=Path,
                    default=Path('/mnt/sdb/lys/yolo11/datasets/KITTI_3D_Pose_Standalone_v4'))
    ap.add_argument('--predictions-json', type=Path,
                    default=Path('runs/kitti_full_2d_predictions/predicted_2d_keypoints_val.json'))
    ap.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    ap.add_argument('--confidence', type=float, default=0.25,
                    help='Car confidence threshold; 0.25 was selected on Val by 2D F1.')
    ap.add_argument('--limit', type=int, default=0, help='0=all frames in the selected split JSON')
    ap.add_argument('--out', type=Path, default=Path('runs/kitti_car_3d_recovery_val'))
    ap.add_argument('--evaluate', action='store_true',
                    help='Read KITTI label_2 only after recovery to calculate 3D metrics')
    ap.add_argument('--match-iou', type=float, default=0.50,
                    help='2D bbox IoU threshold for post-recovery GT association')
    ap.add_argument('--save-images', type=int, default=30)
    ap.add_argument('--reproj-sigma-px', type=float, default=5.0,
                    help='Pixel residual scale; chosen to reflect observed Car keypoint noise')
    ap.add_argument('--max-reproj-px', type=float, default=25.0,
                    help='Reject implausible fits above this 8-point reprojection RMSE')
    ap.add_argument('--plane-sigma-m', type=float, default=0.28)
    ap.add_argument('--disable-frame-plane', action='store_true',
                    help='Ablation: use only per-object size-anchored fits')
    ap.add_argument('--car-height', type=float, default=1.55)
    ap.add_argument('--car-height-sigma', type=float, default=0.18)
    ap.add_argument('--car-width', type=float, default=1.80)
    ap.add_argument('--car-width-sigma', type=float, default=0.30)
    ap.add_argument('--car-length', type=float, default=4.20)
    ap.add_argument('--car-length-sigma', type=float, default=0.90)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not 0.0 <= args.confidence <= 1.0:
        ap.error('--confidence must be in [0,1]')
    if not 0.0 < args.match_iou <= 1.0:
        ap.error('--match-iou must be in (0,1]')
    for k in ('reproj_sigma_px', 'plane_sigma_m', 'max_reproj_px'):
        if getattr(args, k) <= 0:
            ap.error(f'--{k.replace("_", "-")} must be >0')
    if not args.predictions_json.is_file():
        ap.error(f'Predictions not found: {args.predictions_json}')
    if not args.dataset_root.is_dir():
        ap.error(f'Dataset root not found: {args.dataset_root}')

    kitti = locate_kitti(args.kitti_root)
    split_payload = load_new_prediction_payload(args.predictions_json)
    json_split = split_payload.get('split')
    if json_split and json_split != args.split:
        ap.error(f'Prediction JSON split is {json_split}, but --split={args.split}')
    image_records = split_payload['images']
    if args.limit > 0:
        image_records = image_records[:args.limit]
    if not image_records:
        ap.error('No image records to process')

    split_lookup = get_3source_splits(args.dataset_root)
    args.out.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    frame_results = []
    metric_rows = []
    drawn = 0
    seen_frames = set()
    total_gt_objects = 0
    matched_gt_objects = 0
    total_recovered_detections = 0

    print('========== KITTI CAR 3D RECOVERY (C1) ==========', flush=True)
    print('Prediction JSON:', args.predictions_json, flush=True)
    print('Selected split:', args.split, flush=True)
    print('Frames:', len(image_records), flush=True)
    print('Class: car (class_id=0)', flush=True)
    print('Confidence threshold:', args.confidence, flush=True)
    print('3D priors (h,w,l):',
          (args.car_height, args.car_width, args.car_length), 'm', flush=True)
    print('3D GT use: evaluation only' if args.evaluate else '3D GT use: disabled', flush=True)

    for fi, record in enumerate(image_records, 1):
        image_file = record.get('image_file')
        if not image_file:
            counters['invalid_image_record'] += 1
            continue
        frame = normalize_frame(image_file)
        if frame in seen_frames:
            raise RuntimeError(f'Duplicate frame in prediction JSON: {frame}')
        seen_frames.add(frame)

        split_names = sorted(split_lookup.get(frame, set()))
        if args.split not in split_names:
            raise RuntimeError(
                f'Frame {frame} is not listed in dataset split {args.split}; '
                f'found membership={split_names}'
            )

        image_path = find_image(kitti, frame)
        calib_path = kitti / 'calib' / f'{frame}.txt'
        if image_path is None or not calib_path.is_file():
            counters['missing_image_or_calib'] += 1
            continue
        try:
            P = parse_p2(calib_path)
        except Exception as exc:
            counters['bad_calib'] += 1
            counters[f'bad_calib_detail:{type(exc).__name__}'] += 1
            continue

        rows = []
        detections = record.get('detections', [])
        for det_idx, det in enumerate(detections):
            try:
                cls = int(det.get('class_id', -1))
                if cls != 0:
                    continue
                conf = float(det.get('confidence', 0.0))
                if conf < args.confidence:
                    counters['below_confidence'] += 1
                    continue
                uv = np.asarray(det.get('keypoints_px', []), dtype=float)
                bbox = np.asarray(det.get('bbox_xyxy_px', []), dtype=float)
                if (uv.shape != (8, 2) or bbox.shape != (4,) or
                        not np.isfinite(uv).all() or not np.isfinite(bbox).all()):
                    counters['invalid_2d_prediction'] += 1
                    continue
                row = {
                    'frame': frame,
                    'det_index': det_idx,
                    'class_id': 0,
                    'confidence': conf,
                    'bbox_xyxy': bbox.tolist(),
                    'uv': uv.tolist(),
                    'initial': None,
                    'final': None,
                    'plane_applied': False,
                    'reason': None,
                }
                try:
                    box, xyz, rms, objective, approx_z = fit_one(
                        uv, P, make_prior(0, args), args.reproj_sigma_px,
                        args.max_reproj_px
                    )
                    row['initial'] = {
                        'box': box, 'xyz': xyz, 'rmse': rms,
                        'obj': objective, 'approx_z': approx_z
                    }
                except Exception as exc:
                    row['reason'] = f'{type(exc).__name__}: {exc}'
                    counters['initial_fit_failed'] += 1
                rows.append(row)
            except Exception as exc:
                counters['invalid_detection_record'] += 1
                continue

        initial = [
            {'box': r['initial']['box']}
            for r in rows if r['initial'] is not None
        ]
        plane = None if args.disable_frame_plane else estimate_frame_road(initial)

        for row in rows:
            if row['initial'] is None:
                continue
            fit = row['initial']
            # Apply frame-plane refinement only if >=4 spatially distributed
            # vehicles support a slope fit; otherwise keep the per-object fit.
            if plane is not None and plane.get('slope_fit', False) and plane['num_initial_boxes'] >= 4:
                try:
                    b, x, rm, obj, az = fit_one(
                        np.asarray(row['uv']), P, make_prior(0, args),
                        args.reproj_sigma_px, args.max_reproj_px,
                        plane, args.plane_sigma_m
                    )
                    if rm <= max(fit['rmse'] + 3.0, fit['rmse'] * 1.6):
                        fit = {'box': b, 'xyz': x, 'rmse': rm,
                               'obj': obj, 'approx_z': az}
                        row['plane_applied'] = True
                        counters['frame_plane_refined_objects'] += 1
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    counters['frame_plane_refinement_failed'] += 1
            row['final'] = fit
            counters['recovered'] += 1

        total_recovered_detections += sum(r['final'] is not None for r in rows)
        counters['frames_processed'] += 1
        counters['car_predictions_after_conf'] += len(rows)

        gt = []
        matches = {}
        if args.evaluate:
            # Crucial: read label_2 only after every prediction in this frame
            # has been reconstructed. It is used solely for association/scoring.
            gt = read_gt(kitti / 'label_2' / f'{frame}.txt', [0])
            total_gt_objects += len(gt)
            matchable_rows = [r for r in rows if r['final'] is not None]
            original_indices = [i for i, r in enumerate(rows) if r['final'] is not None]
            local_matches = assign_gt(matchable_rows, gt, args.match_iou)
            matches = {
                original_indices[local_i]: pair
                for local_i, pair in local_matches.items()
            }
            matched_gt_objects += len(matches)
            counters['matched_2d_gt_objects'] += len(matches)
            counters['unmatched_gt_objects'] += max(0, len(gt) - len(matches))
            counters['recovered_unmatched_detections'] += max(0, len(matchable_rows) - len(matches))

            for row_idx, (gt_idx, iou_2d) in matches.items():
                row = rows[row_idx]
                pred = row['final']['box']
                truth = gt[gt_idx][2]
                pred_xyz = np.asarray(row['final']['xyz'])
                truth_xyz = corners(truth.h, truth.w, truth.l, truth.center, truth.yaw)
                center_delta = pred.center - truth.center
                abs_depth = abs(float(center_delta[2]))
                depth = float(truth.center[2])
                depth_bin = '<20m' if depth < 20 else ('20-40m' if depth < 40 else ('40-60m' if depth < 60 else '>=60m'))
                stage1 = row['initial']['box']
                metric_rows.append({
                    'split': args.split,
                    'frame': frame,
                    'det_index': row['det_index'],
                    'confidence': row['confidence'],
                    'gt_line_index': gt[gt_idx][0],
                    'gt_source_class': gt[gt_idx][1],
                    'match_2d_iou': iou_2d,
                    'gt_depth_m': depth,
                    'depth_bin': depth_bin,
                    'iou3d': iou3d(pred, truth),
                    'center_error_m': float(np.linalg.norm(center_delta)),
                    'depth_z_error_m': float(center_delta[2]),
                    'depth_abs_error_m': abs_depth,
                    'height_error_m': float(pred.h - truth.h),
                    'width_error_m': float(pred.w - truth.w),
                    'length_error_m': float(pred.l - truth.l),
                    'yaw_error_mod_pi_rad': float(abs((pred.yaw-truth.yaw+np.pi/2) % np.pi - np.pi/2)),
                    'corner_rmse_m': corner_error(pred_xyz, truth_xyz),
                    'reproj_rmse_px': row['final']['rmse'],
                    'stage1_iou3d': iou3d(stage1, truth),
                    'stage1_center_error_m': float(np.linalg.norm(stage1.center-truth.center)),
                    'stage1_depth_abs_error_m': abs(float(stage1.center[2]-truth.center[2])),
                    'stage1_reproj_rmse_px': row['initial']['rmse'],
                    'frame_plane_refinement_applied': row['plane_applied'],
                })

        if drawn < args.save_images and any(r['final'] is not None for r in rows):
            draw_frame(
                image_path, P, rows,
                args.out / 'visualizations' / f'{frame}.png',
                gt if args.evaluate else None, matches
            )
            drawn += 1

        out_dets = []
        for row in rows:
            item = {
                'frame': frame,
                'det_index': row['det_index'],
                'class_id': 0,
                'class_name': 'car',
                'confidence': row['confidence'],
                'bbox_xyxy_px': row['bbox_xyxy'],
                'keypoints_px_predicted': row['uv'],
                'status': 'ok' if row['final'] is not None else 'failed',
                'frame_plane_refinement_applied': row['plane_applied'],
            }
            if row['initial'] is not None:
                item['stage1_3d_box'] = serialize(row['initial']['box'])
                item['stage1_reprojection_rmse_px'] = row['initial']['rmse']
            if row['final'] is not None:
                item['pred_3d_box'] = serialize(row['final']['box'])
                item['pred_corners_camera_xyz_m'] = row['final']['xyz'].tolist()
                item['reprojection_rmse_px'] = row['final']['rmse']
            else:
                item['failure_reason'] = row['reason']
            out_dets.append(item)

        frame_results.append({
            'frame': frame,
            'image_file': image_file,
            'image_path': str(image_path),
            'split': args.split,
            'frame_road_estimate': plane,
            'num_car_detections_after_confidence': len(rows),
            'num_recovered': sum(r['final'] is not None for r in rows),
            'detections': out_dets,
        })

        if fi % 50 == 0 or fi == len(image_records):
            print(
                f'[{fi}/{len(image_records)}] frames={counters["frames_processed"]} '
                f'car_pred={counters["car_predictions_after_conf"]} '
                f'recovered={counters["recovered"]} '
                f'3D_eval_matches={len(metric_rows)}',
                flush=True
            )

    # Save recovered boxes independently from evaluation metrics.
    (args.out / 'recovered_3d_car.json').write_text(
        json.dumps(frame_results, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    if args.evaluate:
        metric_path = args.out / 'metrics.csv'
        if metric_rows:
            with metric_path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metric_rows)
        else:
            metric_path.write_text('', encoding='utf-8')

        depth_rows = []
        for depth_bin in ('<20m', '20-40m', '40-60m', '>=60m'):
            subset = [r for r in metric_rows if r['depth_bin'] == depth_bin]
            if not subset:
                continue
            for metric in ('iou3d', 'center_error_m', 'depth_abs_error_m',
                           'corner_rmse_m', 'reproj_rmse_px'):
                stats = aggregate_metric(subset, metric)
                depth_rows.append({
                    'split': args.split, 'depth_bin': depth_bin,
                    'num_matches': len(subset), 'metric': metric,
                    'mean': stats['mean'], 'median': stats['median'], 'p90': stats['p90'],
                })
        if depth_rows:
            with (args.out / 'metrics_by_depth.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
                writer.writeheader()
                writer.writerows(depth_rows)

    result = {
        'method': 'C1: predicted 8x2 keypoints + full KITTI P2 + independent car-size priors + optional image-derived weak frame plane',
        'split': args.split,
        'kitti_root': str(args.kitti_root),
        'dataset_root_readonly': str(args.dataset_root),
        'prediction_json': str(args.predictions_json),
        'confidence_threshold': args.confidence,
        'match_iou_for_evaluation_only': args.match_iou,
        'class': 'car',
        'source_GT_classes_for_evaluation': ['Car', 'Van', 'Truck'],
        'frames_in_prediction_json': len(image_records),
        'frames_processed': counters['frames_processed'],
        'counts': dict(counters),
        'car_size_priors_m': {
            'height': {'mean': args.car_height, 'sigma': args.car_height_sigma},
            'width': {'mean': args.car_width, 'sigma': args.car_width_sigma},
            'length': {'mean': args.car_length, 'sigma': args.car_length_sigma},
        },
        'frame_plane': {
            'enabled': not args.disable_frame_plane,
            'application_rule': 'Only when >=4 successful car fits span >=3m laterally and >=8m in depth; otherwise no refinement',
            'plane_is_observed': False,
            'description': 'Weak plane estimated from initial image-derived bottom-center estimates, not an independently measured road plane.'
        },
        'reconstruction_input_exclusion': [
            'No KITTI label_2 dimensions, locations or yaw are used by the solver.',
            'No LiDAR point cloud is used.',
            'KITTI label_2 is read only after reconstruction for 2D association and 3D scoring.'
        ],
        'evaluation': {},
        'cautions': [
            'Absolute metric scale is not observable from a single monocular image and P2 alone; the class-size prior anchors scale.',
            'Car is a merged v4 category containing original KITTI Car, Van and Truck; one generic car prior may be biased for vans/trucks.',
            '3D cuboid geometry cannot distinguish yaw from yaw+pi; yaw error is reported modulo pi.',
            'The frame plane is a weak image-derived consistency constraint, not a measured physical road plane.'
        ]
    }
    if args.evaluate:
        result['evaluation'] = {
            'ground_truth_objects': total_gt_objects,
            'matched_ground_truth_objects': matched_gt_objects,
            'unmatched_ground_truth_objects': max(0, total_gt_objects - matched_gt_objects),
            'matched_recall_over_GT': float(matched_gt_objects / total_gt_objects) if total_gt_objects else None,
            'metrics_on_matched_objects': {}
        }
        for key in ('iou3d', 'center_error_m', 'depth_abs_error_m',
                    'height_error_m', 'width_error_m', 'length_error_m',
                    'yaw_error_mod_pi_rad', 'corner_rmse_m', 'reproj_rmse_px',
                    'stage1_iou3d', 'stage1_center_error_m',
                    'stage1_depth_abs_error_m', 'stage1_reproj_rmse_px'):
            result['evaluation']['metrics_on_matched_objects'][key] = aggregate_metric(metric_rows, key)
        result['evaluation']['depth_bin_counts'] = dict(Counter(r['depth_bin'] for r in metric_rows))

    (args.out / 'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    report = [
        'KITTI CAR MONOCULAR 3D RECOVERY',
        f'Split: {args.split}',
        f'Frames processed: {counters["frames_processed"]}/{len(image_records)}',
        f'Confidence threshold: {args.confidence}',
        f'Car detections after threshold: {counters["car_predictions_after_conf"]}',
        f'Recovered 3D boxes: {counters["recovered"]}',
        f'Initial fit failures: {counters["initial_fit_failed"]}',
        f'Frame-plane refined boxes: {counters["frame_plane_refined_objects"]}',
        '',
        'The solver consumes only predicted 2D keypoints, P2 and independent size priors.',
        'KITTI label_2 is used only after recovery for matching/evaluation.',
        'Car evaluation includes original KITTI Car, Van and Truck labels to match v4 mapping.',
        ''
    ]
    if args.evaluate:
        report += [
            f'GT objects: {total_gt_objects}',
            f'Matched GT objects: {matched_gt_objects}',
            f'Matched recall: {matched_gt_objects / total_gt_objects:.4f}' if total_gt_objects else 'Matched recall: N/A',
        ]
        for key in ('iou3d', 'center_error_m', 'depth_abs_error_m',
                    'corner_rmse_m', 'reproj_rmse_px', 'stage1_iou3d',
                    'stage1_center_error_m'):
            report.append(f'{key}: {aggregate_metric(metric_rows, key)}')
    report += [
        '',
        'Cautions:',
        '- Absolute scale depends on the independent car-size prior.',
        '- This is a C1 baseline, not a claim of uniquely observable monocular 3D.',
        '- Use Val for method development; freeze all choices before running Test.',
    ]
    (args.out / 'report.txt').write_text('\n'.join(report), encoding='utf-8')

    print('\n========== CAR 3D RECOVERY COMPLETE ==========', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Output:', args.out, flush=True)
    print('Recovered boxes:', args.out / 'recovered_3d_car.json', flush=True)
    if args.evaluate:
        print('Metrics:', args.out / 'metrics.csv', flush=True)
        print('Depth metrics:', args.out / 'metrics_by_depth.csv', flush=True)
    print('Summary:', args.out / 'summary.json', flush=True)
    print('Report:', args.out / 'report.txt', flush=True)


if __name__ == '__main__':
    main()
