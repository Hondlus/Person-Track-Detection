import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np


model = YOLO('./weights/yolo11x.pt')    ### Pre-trained weights

# Store the track history
track_history = defaultdict(lambda: [])

# Open the video file
video_path = "./test_video/test.mp4"
cap = cv2.VideoCapture(video_path)

# Write the video file
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）
output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4" # 设置输出视频的保存路径
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)  # 创建一个VideoWriter对象用于写视频

count = 0
offset = 6
# cy1 = int(h / 2)
# cy2 = cy1 + 20
cy1 = 500
cy2 = 520
vh_down = []
counter = []
vh_up = []
counter2 = []

def is_crossing():
    #####going DOWN#####
    if cy1 < (y + offset) and cy1 > (y - offset):
        vh_down.append(track_id)
    if track_id in vh_down:
        if cy2 < (y + offset) and cy2 > (y - offset):
            if counter.count(track_id) == 0:
                counter.append(track_id)

    #####going UP#####
    if cy2 < (y + offset) and cy2 > (y - offset):
        vh_up.append(track_id)
    if track_id in vh_up:
        if cy1 < (y + offset) and cy1 > (y - offset):
            if counter2.count(track_id) == 0:
                counter2.append(track_id)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cuda()
        track_ids = results[0].boxes.id.int().cuda().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot(conf=False)

        for box, track_id in zip(boxes, track_ids):
            x, y, box_w, box_h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=5)

            # 越线检测
            is_crossing()

        cv2.line(annotated_frame, (0, cy1), (w, cy1), (0, 0, 255), 3)
        cv2.putText(annotated_frame, ('L1'), (182, cy1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(annotated_frame, (0, cy2), (w, cy2), (0, 0, 255), 3)
        cv2.putText(annotated_frame, ('L2'), (182, cy2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        d = (len(counter))
        u = (len(counter2))
        cv2.putText(annotated_frame, ('goingdown: ') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, ('goingup: ') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # Write the video frame
        output_video.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
output_video.release()
cap.release()
cv2.destroyAllWindows()