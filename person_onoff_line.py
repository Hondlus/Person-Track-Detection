import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import matplotlib.path as mplPath


model = YOLO('./weights/yolo11n.pt')    ### Pre-trained weights

# Store the track history
track_history = defaultdict(lambda: [])

# Open the video file
# video_path = "./test_video/test.mp4"
video_path = 0
# video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/101" # 外走廊高清  1920 1080
# video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/102" # 外走廊标清 640 360
# video_path = "rtsp://admin:HikNJQXFP@192.168.50.10:554/Streaming/Channels/102" # 屋内大屏摄像头
# video_path = "rtsp://admin:Dxw202409@192.168.50.20:554/stream2"  # 15fps 640 480
cap = cv2.VideoCapture(video_path)

# Write the video file
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 640 x 360
w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）
print("fps, w, h: ", fps, w, h)
output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4" # 设置输出视频的保存路径
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)  # 创建一个VideoWriter对象用于写视频

# 多边形坐标  高清 x:795、950 标清 x:260、320
POLYGON_1 = np.array([
    [0, 0], # 左上
    [440, 0], # 右上
    [440, 360], # 右下
    [0, 360], # 左下
])

# 是否穿越区域
def is_online(xc, yc):
    # mplPath.Path(POLYGON_1).contains_point((xc, yc)判断中心点是否在区域内
    if mplPath.Path(POLYGON_1).contains_point((xc, yc)):
        return True
    else:
        return False

# tm = cv2.TickMeter()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # tm.start()
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], device=0)
        # print(results[0].boxes.is_track)
        if results[0].boxes.is_track is True:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cuda()
            track_ids = results[0].boxes.id.int().cuda().tolist()

            # Visualize the results on the frame
            frame = results[0].plot(conf=False)

            for box, track_id in zip(boxes, track_ids):
                x, y, box_w, box_h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=5)

                # 判断行人是否在区域
                if is_online(x, y):
                    cv2.polylines(img=frame, pts=[POLYGON_1], isClosed=True, color=(255, 0, 255), thickness=4)
                    cv2.putText(frame, 'ONLINE', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.polylines(img=frame, pts=[POLYGON_1], isClosed=True, color=(255, 255, 0), thickness=4)
                    cv2.putText(frame, 'OFFLINE', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        # Write the video frame
        output_video.write(frame)

        # Display the annotated frame
        # frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Tracking", frame)

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