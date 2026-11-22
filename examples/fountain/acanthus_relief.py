"""Acanthus relief swept from hand-traced centerlines on the fountain reference.

Each control point stores reference-image x/y, brush half-width in image pixels,
and projection from the carved stone face in voxel units.
"""
from functools import lru_cache

import numpy as np


# Central leaf, four overlapping lobes, and the rolled-over basal leaf tips.
CROWN = (
    ((772,969,3,6),(770,948,12,15),(775,903,12,19),(776,852,11,20),
     (776,803,14,20),(770,776,16,16),(779,762,7,13),(785,772,2,6)),
    ((782,956,7,11),(789,918,10,16),(798,868,16,21),(815,816,23,21),
     (836,788,20,20),(852,787,12,16),(861,792,3,9)),
    ((786,953,7,10),(805,918,12,18),(823,874,22,20),(848,836,23,22),
     (875,808,16,20),(887,805,10,13),(898,815,3,7)),
    ((799,942,5,8),(821,915,14,16),(842,886,20,20),(858,870,21,18),
     (877,853,14,19),(890,854,9,14),(883,866,3,9)),
    ((806,945,4,8),(833,942,11,13),(849,921,15,18),(852,902,11,21),
     (847,889,9,15),(836,896,4,11)),
    ((779,961,4,11),(790,954,11,13),(801,944,12,15),(813,950,9,15),
     (815,960,5,10),(804,966,2,7)),
)

# Smaller rolled lobe tips traced separately so they have their own edges.
LOBES = (
    ((809,849,4,11),(814,827,8,16),(827,816,10,18),(838,818,8,17),(844,827,3,10)),
    ((825,877,4,10),(836,858,9,14),(849,847,11,18),(861,848,8,15),(866,856,3,10)),
    ((846,869,4,9),(858,852,10,14),(872,839,10,17),(886,838,8,16),(891,846,3,10)),
    ((837,914,4,10),(831,900,9,15),(824,892,8,16),(817,897,4,10)),
    ((863,927,5,9),(875,911,8,14),(879,895,9,15),(876,885,7,14),(868,883,3,10)),
)

# Sweep rightward from the fan, then fold the living branch back into a tight curl.
VINES = (
    ((809,954,6,11),(843,949,11,15),(874,937,13,15),(908,914,12,16),
     (944,895,12,16),(981,878,14,19),(1017,874,15,19),(1044,882,12,18),
     (1060,898,10,17),(1055,916,9,15),(1039,917,8,12),(1030,906,6,12),
     (1038,897,4,10),(1047,904,2,6)),
    ((908,913,6,11),(930,889,13,14),(953,870,16,17),(981,851,15,17),
     (1015,837,12,14),(1033,837,3,8)),
    ((877,938,5,9),(898,921,10,14),(915,917,12,15),(943,921,12,16),
     (957,931,8,12),(954,939,3,7)),
    ((858,947,6,10),(876,947,10,13),(891,955,10,13),(894,968,4,8)),
    ((1000,872,7,11),(1028,861,11,14),(1056,862,12,16),(1084,874,11,16),
     (1101,891,8,14),(1109,908,4,8)),
    ((1063,871,5,9),(1091,876,8,12),(1118,889,10,13),(1144,902,11,14),
     (1177,906,8,12),(1205,899,5,9),(1237,883,3,6)),
    ((1112,887,5,9),(1100,870,9,12),(1082,853,10,13),(1064,849,6,10),
     (1051,855,2,6)),
    ((924,905,4,9),(932,885,8,12),(931,870,8,13),(919,856,6,10),
     (908,851,2,6)),
)

# Supporting leaf clusters farther around the stone apron.
SIDE = (
    ((1215,900,6,8),(1238,887,13,13),(1254,870,15,16),
     (1268,845,13,16),(1263,827,11,15),(1250,823,5,10)),
    ((1239,899,4,8),(1261,889,11,12),(1284,863,13,16),(1298,841,13,16),
     (1305,827,6,12),(1296,820,3,8)),
    ((1270,884,4,8),(1291,880,10,13),(1311,866,11,15),(1333,849,13,17),
     (1353,842,10,16),(1368,847,8,14),(1374,862,7,13),(1365,869,4,9)),
    ((1314,860,4,7),(1321,839,11,13),(1337,824,11,14),(1355,814,5,10)),
    ((1350,849,3,7),(1369,837,10,15),(1390,833,12,15),(1415,839,8,12)),
)

PEDESTAL = (
    ((772,650,4,6),(772,629,8,10),(773,600,9,14),(776,569,10,15),
     (781,547,7,13),(790,535,2,7)),
    ((780,650,4,7),(787,626,7,12),(799,597,12,15),(818,569,14,16),
     (833,555,10,13),(844,563,3,9)),
    ((781,653,4,7),(795,640,9,12),(812,619,12,14),(832,606,13,16),
     (854,605,9,14),(862,615,3,10)),
    ((788,654,3,7),(808,647,8,12),(830,636,11,15),(850,638,10,15),
     (863,649,3,9)),
    ((825,570,4,7),(851,546,9,11),(862,524,10,15),(871,499,8,14),
     (886,486,9,15),(902,489,8,12),(908,504,6,10),(900,511,3,8)),
    ((826,567,5,7),(848,571,11,12),(868,572,13,14),(887,583,12,15),
     (899,600,10,15),(895,615,9,13),(880,618,8,12),(871,609,4,10)),
)

H_STEP = .5


def smooth_segments(points):
    """Interpolate the centerline without the polygonal corners of a polyline."""
    p = np.vstack([points[0], points, points[-1]])
    for i in range(1, len(p)-2):
        a, b, c, d = p[i-1:i+3]
        samples = max(3, int(np.linalg.norm(c[:2]-b[:2])/.64))
        for t in np.arange(samples)/samples:
            q = .5*(2*b+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t)
            yield q
    yield points[-1]


def sweep(atlas, points, *, leaf):
    samples = tuple(smooth_segments(points))
    traveled = 0.
    for index, (cx, cy, width, depth) in enumerate(samples):
        width = max(.58, width)
        x0 = max(0, int((cx-width)/H_STEP))
        x1 = min(atlas.shape[0], int((cx+width)/H_STEP)+2)
        y0 = max(0, int((cy-width)/H_STEP))
        y1 = min(atlas.shape[1], int((cy+width)/H_STEP)+2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0,x1)[:,None]*H_STEP-cx
        dy = np.arange(y0,y1)[None,:]*H_STEP-cy
        u = np.hypot(dx,dy)/width
        shape = np.sqrt(np.maximum(0.,1.-u*u))
        if leaf:
            prev = samples[max(0,index-1)][:2]
            ahead = samples[min(len(samples)-1,index+1)][:2]
            tangent = ahead-prev
            normal = np.array([-tangent[1],tangent[0]])/max(.001,np.linalg.norm(tangent))
            side = abs(normal[0]*dx+normal[1]*dy)/width
            traveled += np.linalg.norm(np.array([cx,cy])-prev)
            midrib = np.exp(-(side/.20)**2)
            folds = np.maximum(0,np.cos(1.15*traveled-4.2*side))**11
            grooves = np.exp(-((side-.38)/.12)**2)+.7*np.exp(-((side-.68)/.12)**2)
            shape *= .92+.16*midrib-.19*grooves-.14*folds*np.clip(side*1.45,0,1)
        tile = atlas[x0:x1,y0:y1]
        np.maximum(tile, shape*depth, out=tile)


@lru_cache(maxsize=2)
def relief_atlas(pedestal=False):
    if pedestal:
        atlas = np.zeros((205,221),dtype=np.float32)
        groups = [(PEDESTAL, None, .55, .41, 4., .52)]
    else:
        atlas = np.zeros((2721,193),dtype=np.float32)
        # Main frontal composition, followed by the smaller side cartouches.
        groups = [(CROWN, None, .78, .318, 14.5, .50),
                  (VINES, None, .80, .316, 14.6, .49),
                  (LOBES, None, .79, .318, 14.6, .46),
                  (SIDE, None, .85, .318, 14.6, .52),
                  (SIDE, 605., .54, .29, 16.3, .46),
                  (SIDE, 717., .72, .30, 15.8, .49),
                  (SIDE, 856., .76, .32, 14.8, .49),
                  (SIDE, 1011., .72, .30, 16.1, .46),
                  (SIDE, 1152., .75, .32, 15.3, .47),
                  (SIDE, 1298., .34, .33, 14.6, .40)]
    for paths, offset, sx, sy, base_y, width in groups:
        for i,path in enumerate(paths):
            points = np.array(path,dtype=np.float64)
            origin_x, origin_y = (772,656) if pedestal else (772,969)
            depth_scale = .87 if offset is not None else 1.
            if offset is not None:
                origin_x = 1215
            points[:,0] = (points[:,0]-origin_x)*sx+(offset or 0.)
            points[:,1] = base_y+(origin_y-points[:,1])*sy
            points[:,2] *= width
            points[:,3] *= depth_scale
            sweep(atlas, points, leaf=(i<5 if paths is CROWN else i not in (0,4,5)))
    return atlas


def relief_at(tangent, height, *, pedestal=False):
    """Map the traced design onto a cylindrical wall, with a subtly varied other side."""
    atlas = relief_atlas(pedestal)
    u = np.where(tangent < 0, -tangent*(1.+.021*np.sin(height*.13)), tangent)
    h = height+np.where(tangent < 0,.72*np.sin(u*.035),0.)
    u, h = np.broadcast_arrays(u/H_STEP,h/H_STEP)
    ix = np.clip(np.floor(u).astype(np.int32),0,atlas.shape[0]-2)
    iy = np.clip(np.floor(h).astype(np.int32),0,atlas.shape[1]-2)
    fx, fy = np.clip(u-ix,0,1), np.clip(h-iy,0,1)
    depth = ((1-fx)*((1-fy)*atlas[ix,iy]+fy*atlas[ix,iy+1])
             +fx*((1-fy)*atlas[ix+1,iy]+fy*atlas[ix+1,iy+1]))
    return np.where((u < atlas.shape[0]-1)&(h >= 0)&(h < atlas.shape[1]-1),depth,0.)
