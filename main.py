import cv2 as cv
import mediapipe as mp
import numpy as np
import math
mp_face_mesh = mp.solutions.face_mesh

# Euclidean distance 
def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def position_indicator(img,dist1, total_dist, eye='R-eye: ',text_pos = (30,40),):
    ratio1 = dist1/total_dist
    pos = ""
    color=(0,0,0)
    if ratio1 <=0.43:
        pos = "left"
        color=(0,255,255)
    elif ratio1 >=0.56:
        pos ='right'
        color=(0,255,0)
    else: 
        pos = 'center'
        color=(0,0,255)
  
    cv.putText(img, f"{eye} {pos} {round(ratio1,3)}", text_pos, cv.FONT_HERSHEY_PLAIN, 1.2, color, 2, cv.LINE_AA)

cap = cv.VideoCapture(0)
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

with mp_face_mesh.FaceMesh(
    max_num_faces =1,
    refine_landmarks=True,
    min_detection_confidence =0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)    
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w = frame.shape[:2]
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
            for p in results.multi_face_landmarks[0].landmark ])
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255, 0), 2)
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            right_center =np.array([r_cx, r_cy], dtype =np.int32)
            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            left_center =np.array([l_cx, l_cy], dtype =np.int32)
            cv.circle(frame, right_center, int(r_rad), (0,255,0), 1, cv.LINE_AA)
            cv.circle(frame, left_center, int(l_rad), (0,255,0), 1, cv.LINE_AA)
            cv.circle(frame, right_center, 2, (0,200,0), -1, cv.LINE_AA)
            cv.circle(frame, left_center, 2, (0,200,0), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[LEFT_EYE[0]], 3, (0,255,0), 1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[LEFT_EYE[8]], 3, (0,0,255), 1, cv.LINE_AA)
            right_dist_total = euclidean_distance(mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]])
            right_dist = euclidean_distance(mesh_points[RIGHT_EYE[0]], right_center)
            left_dist_total = euclidean_distance(mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]])
            left_dist = euclidean_distance(mesh_points[LEFT_EYE[0]], left_center)

            position_indicator(frame,right_dist, right_dist_total)
            position_indicator(frame,left_dist, left_dist_total, eye="L-eye", text_pos=(30, 60))
            # print(mesh_points)
        cv.imshow('img', frame)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            break
cv.destroyAllWindows()
cap.release()
