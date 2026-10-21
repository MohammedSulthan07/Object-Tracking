from dataclasses import dataclass
from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[..., 2]-bb_test[..., 0]) * (bb_test[..., 3]-bb_test[..., 1])
    area_gt = (bb_gt[..., 2]-bb_gt[..., 0]) * (bb_gt[..., 3]-bb_gt[..., 1])
    return inter / np.maximum(area_test + area_gt - inter, 1e-6)


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=float)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history, self.hits, self.hit_streak = [], 0, 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x,y,s,r]).reshape((4,1))


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(max(x[2] * x[3], 1e-6))
    h = x[2] / max(w, 1e-6)
    if score is None:
        return np.array([x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]).reshape((1,4))
    return np.array([x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2, score]).reshape((1,5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections), dtype=int), np.empty((0,), dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    rows, cols = linear_sum_assignment(-iou_matrix)
    matched = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))
    for r, c in zip(rows, cols):
        if iou_matrix[r, c] < iou_threshold:
            continue
        matched.append([r,c])
        if r in unmatched_dets: unmatched_dets.remove(r)
        if c in unmatched_trks: unmatched_trks.remove(c)
    return np.array(matched, dtype=int), np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


class Sort:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        detections = np.asarray(detections, dtype=float).reshape((-1, 5)) if len(detections) else np.empty((0,5))
        trks = np.zeros((len(self.trackers), 4))
        to_del=[]; ret=[]
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections[:, :4] if len(detections) else np.empty((0,4)), trks, self.iou_threshold)
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1,5))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.concatenate(ret) if ret else np.empty((0,5))
