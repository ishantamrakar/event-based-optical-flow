import numpy as np
import cv2
from optical_flow import LucasKanadeOpticalFlow

def draw_flow(frame, good_old, good_new):
    mask = np.zeros_like(frame)
    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
    return cv2.add(frame, mask)

def main(video_path=0):
    cap = cv2.VideoCapture(video_path)
    of_model = LucasKanadeOpticalFlow()
    
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_old, good_new = of_model.compute_flow(old_gray, gray)
        
        vis = draw_flow(frame.copy(), good_old, good_new)
        cv2.imshow('Optical Flow', vis)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        old_gray = gray.copy()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()