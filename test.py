import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import matplotlib.path as mplPath
from datetime import datetime


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

red_region = set()
blue_region = set()
in_count = 0
out_count = 0

# 多边形坐标  高清 x:795、950 标清 x:260、320
RED_POLYGON = np.array([
    [0, 0], # 左上
    [320, 0], # 右上
    [320, 360], # 右下
    [0, 360], # 左下
])
BLUE_POLYGON = np.array([
    [330, 0], # 左上
    [640, 0], # 右上
    [640, 360], # 右下
    [330, 360], # 左下
])

def is_crossing(x_center, y_center):
    global in_count, out_count
    # 当区域人数达到一定数量，进行定量删除，节省内存，稳定性能
    if len(red_region) > 10000:
        red_region.pop(0)
    if len(blue_region) > 10000:
        blue_region.pop(0)

    # 判断行人在哪个区域内，是否跨越区域
    if mplPath.Path(RED_POLYGON).contains_point((x, y)):
        red_region.add(track_id)
        if track_id in red_region:
            if track_id in blue_region:
                in_count += 1
                blue_region.remove(track_id)

    if mplPath.Path(BLUE_POLYGON).contains_point((x, y)):
        blue_region.add(track_id)
        if track_id in blue_region:
            if track_id in red_region:
                out_count += 1
                red_region.remove(track_id)

    return in_count, out_count

# tm = cv2.TickMeter()
# Loop through the video frames
while cap.isOpened():

    # 获取当前系统时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Read a frame from the video
    success, frame = cap.read()

    # tm.start()
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], device='cpu')
        results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0, 67], device=0)
        # print(results[0].boxes.is_track)
        if results[0].boxes.is_track is True:
            # Get the boxes and track IDs
            # boxes = results[0].boxes.xywh.cpu()
            boxes = results[0].boxes.xywh.cuda()
            # track_ids = results[0].boxes.id.int().cpu().tolist()
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

                # 判断行人在哪个区域，是否跨越区域
                in_count, out_count = is_crossing(x, y)

        # 实时显示系统时间
        cv2.putText(frame, current_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Draw the polygon on the frame
        cv2.polylines(img=frame, pts=[RED_POLYGON], isClosed=True, color=(255, 0, 255), thickness=4)
        cv2.polylines(img=frame, pts=[BLUE_POLYGON], isClosed=True, color=(255, 255, 0), thickness=4)

        cv2.putText(frame, 'IN: ' + str(in_count), (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
        cv2.putText(frame, 'OUT: ' + str(out_count), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

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