import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
# import matplotlib.path as mplPath


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

vh_down = []
counter = []
vh_up = []
counter2 = []
# 多边形坐标
POLYGON_UP = [
    [0, 0], # 左上
    [1920, 0], # 右上
    [1920, 800], # 右下
    [0, 310], # 左下
]
POLYGON_DOWN = [
    [0, 311], # 左上
    [1920, 801], # 右上
    [1920, 1080], # 右下
    [0, 1080], # 左下
]

numpy_poly_up = np.array(POLYGON_UP)
numpy_poly_down = np.array(POLYGON_DOWN)

def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
    # AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[0]
    RTPoint = AreaPoint[1]
    RBPoint = AreaPoint[2]
    LBPoint = AreaPoint[3]
    a = (LTPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(LTPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    b = (RTPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(RTPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    c = (RBPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(RBPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    d = (LBPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(LBPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    #print(a,b,c,d)
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False

# 是否穿越区域
def is_crossing(xc, yc):
    #####going DOWN#####
    # mplPath.Path(POLYGON_UP).contains_point((xc, yc)判断中心点是否在区域内
    if isInterArea([int(xc), int(yc)], POLYGON_UP):
        vh_down.append(track_id)
    if track_id in vh_down:
        if isInterArea([int(xc), int(yc)], POLYGON_DOWN):
            if counter.count(track_id) == 0:
                counter.append(track_id)

    #####going UP#####
    if isInterArea([int(xc), int(yc)], POLYGON_DOWN):
        vh_up.append(track_id)
    if track_id in vh_up:
        if isInterArea([int(xc), int(yc)], POLYGON_UP):
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

            # 判断行人是否跨越区域
            is_crossing(x, y)

        cv2.polylines(img=annotated_frame, pts=[numpy_poly_up], isClosed=True, color=(255, 0, 255), thickness=4)
        cv2.polylines(img=annotated_frame, pts=[numpy_poly_down], isClosed=True, color=(255, 255, 0), thickness=4)

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