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
# video_path = 0
# video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/101" # 外走廊高清
# video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/102" # 外走廊标清 640 360
# video_path = "rtsp://admin:HikNJQXFP@192.168.50.10:554/Streaming/Channels/102" # 屋内大屏摄像头
video_path = "rtsp://admin:Dxw202409@192.168.50.20:554/stream2"  # 15fps 640 480
cap = cv2.VideoCapture(video_path)

# Write the video file
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 640 x 360
w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）
print("fps, w, h: ", fps, w, h)
output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4" # 设置输出视频的保存路径
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)  # 创建一个VideoWriter对象用于写视频

vh_down = []
counter = []
vh_up = []
counter2 = []
# 多边形坐标
POLYGON_UP = np.array([
    [220, 0], # 左上
    [430, 0], # 右上
    [430, 480], # 右下
    [220, 480], # 左下
])
POLYGON_DOWN = np.array([
    [0, 0], # 左上
    [210, 0], # 右上
    [210, 480], # 右下
    [0, 480], # 左下
])
POLYGON_DOWN2 = np.array([
    [440, 0],  # 左上
    [640, 0],  # 右上
    [640, 480],  # 右下
    [440, 480],  # 左下
])

# 是否穿越区域
def is_crossing(xc, yc):
    #####going DOWN#####
    # mplPath.Path(POLYGON_UP).contains_point((xc, yc)判断中心点是否在区域内
    if mplPath.Path(POLYGON_UP).contains_point((xc, yc)):
        vh_down.append(track_id)
    if track_id in vh_down:
        if mplPath.Path(POLYGON_DOWN).contains_point((xc, yc)):
            if counter.count(track_id) == 0:
                counter.append(track_id)

    #####going UP#####
    if mplPath.Path(POLYGON_DOWN).contains_point((xc, yc)) or mplPath.Path(POLYGON_DOWN2).contains_point((xc, yc)):
        vh_up.append(track_id)
    if track_id in vh_up:
        if mplPath.Path(POLYGON_UP).contains_point((xc, yc)):
            if counter2.count(track_id) == 0:
                counter2.append(track_id)

# tm = cv2.TickMeter()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    cv2.polylines(img=frame, pts=[POLYGON_UP], isClosed=True, color=(255, 0, 255), thickness=4)
    cv2.polylines(img=frame, pts=[POLYGON_DOWN], isClosed=True, color=(255, 255, 0), thickness=4)
    cv2.polylines(img=frame, pts=[POLYGON_DOWN2], isClosed=True, color=(255, 255, 0), thickness=4)

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

                # 判断行人是否跨越区域
                is_crossing(x, y)

        # d = (len(counter))
        # u = (len(counter2))

        cv2.putText(frame, ('goingdown: ') + str(len(counter)), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, ('goingup: ') + str(len(counter2)), (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # Write the video frame
        output_video.write(frame)

        # Display the annotated frame
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