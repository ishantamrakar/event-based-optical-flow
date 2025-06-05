import cv2
import numpy as np

class OpticalFlowBase: 
    def compute_flow(self, prev_frame, next_frame):
        raise NotImplementedError("Must implement in subclass")
    
class LucasKanadeOpticalFlow(OpticalFlowBase):
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7, win_size=(15,15)):
        self.feature_params = dict(maxCorners=max_corners, 
                                   qualityLevel=quality_level, 
                                   minDistance = min_distance,
                                   blockSize=7)
        self.lk_params = dict(winSize = win_size, 
                              maxLevel = 2, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_pts = None
        
    def compute_flow(self, prev_gray, next_gray):
        if self.prev_pts is None: 
            self.prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.prev_pts, None, **self.lk_params)   
        
        good_new = next_pts[status == 1]
        good_old = self.prev_pts[status == 1]
        
        self.prev_pts = good_new.reshape(-1,1,2)
        return good_old, good_new
        