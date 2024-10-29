from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


# Load the YOLO11 model
# model = YOLO('./weights/yolo11n.pt')        ### Pre-trained weights
# model = YOLO('./weights/yolo11m.pt')        ### Pre-trained weights
# model = YOLO('./weights/yolo11x.pt')        ### Pre-trained weights
# model = YOLO('./weights/yolov8n.pt')            ### Pre-trained weights
# model = YOLO('./weights/yolov8m.pt')            ### Pre-trained weights
model = YOLO('./weights/yolov8x.pt')            ### Pre-trained weights
# model = YOLO('./weights/yolov5nu.pt')            ### Pre-trained weights

# Open the video file
video_path = "C:/Users/DXW/Desktop/yolo_track_reid/test_video/test.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Write the video file
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）
output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/track_video/test.mp4" # 设置输出视频的保存路径
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, size, isColor=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], device=0)

        # Get the boxes and track IDs
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cuda()
        track_ids = results[0].boxes.id.int().cuda().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            # if len(track) > 120:  # retain 90 tracks for 90 frames
            #     track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=5)

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