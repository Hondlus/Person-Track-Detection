import cv2

from ultralytics import YOLO

# Load the YOLO11 model
# model = YOLO('./weights/yolo11n.pt')        ### Pre-trained weights
# model = YOLO('./weights/yolo11m.pt')        ### Pre-trained weights
model = YOLO('./weights/yolo11x.pt')        ### Pre-trained weights
# model = YOLO('./weights/yolov8n.pt')            ### Pre-trained weights
# model = YOLO('./weights/yolov8m.pt')            ### Pre-trained weights
# model = YOLO('./weights/yolov8x.pt')            ### Pre-trained weights
# model = YOLO('./weights/yolov5nu.pt')            ### Pre-trained weights
# model = YOLO('runs/detect/train2/weights/best.pt')  ### weights from trained model

# Open the video file
video_path = "C:/Users/DXW/Desktop/yolo_track_reid/test_video/MOT-0.mp4"
cap = cv2.VideoCapture(video_path)

# Write the video file
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）
output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4" # 设置输出视频的保存路径
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, size, isColor=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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