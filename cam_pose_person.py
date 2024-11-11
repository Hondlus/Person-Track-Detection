import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


def determine_pose(keypoint_list):
    """
    通过关键点判断人体的姿态
    :param keypoints: 关键点坐标和置信度，格式为 [(x1, y1, conf1), (x2, y2, conf2), ...]
    :return: 姿态类别： 'stand'（站立）, 'walk'（行走）, 'jump'（跳跃）, 'unknown'（未知）
    """
    # 如果关键点数量少于17个，直接返回'unknown'
    if len(keypoints) < 17:
        return '无法判断'

    # 定义一些关键点的索引
    left_ankle = keypoints[15]  # 左脚踝
    right_ankle = keypoints[16]  # 右脚踝
    left_knee = keypoints[13]  # 左膝盖
    right_knee = keypoints[14]  # 右膝盖
    left_hip = keypoints[11]  # 左髋关节
    right_hip = keypoints[12]  # 右髋关节

    # 获取关键点的坐标和置信度
    left_ankle_x, left_ankle_y, left_ankle_conf = left_ankle
    right_ankle_x, right_ankle_y, right_ankle_conf = right_ankle
    left_knee_x, left_knee_y, left_knee_conf = left_knee
    right_knee_x, right_knee_y, right_knee_conf = right_knee
    left_hip_x, left_hip_y, left_hip_conf = left_hip
    right_hip_x, right_hip_y, right_hip_conf = right_hip

    # 计算膝盖关节之间的角度
    def calculate_knee_angle(knee, hip, ankle):
        # 计算两个向量的夹角
        vector1 = (knee[0] - hip[0], knee[1] - hip[1])
        vector2 = (ankle[0] - knee[0], ankle[1] - knee[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if magnitude1 * magnitude2 == 0:
            return 0

        cos_theta = dot_product / (magnitude1 * magnitude2)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    # 简单的姿态判断逻辑
    if left_ankle_conf > 0.5 and right_ankle_conf > 0.5:
        # 在 OpenCV 坐标系中，y 坐标值越大，位置越低

        # 计算左脚角度
        knee_angle_left = calculate_knee_angle(left_knee, left_hip, left_ankle)
        # 计算右脚角度
        knee_angle_right = calculate_knee_angle(right_knee, right_hip, right_ankle)
        print("左脚角度,", knee_angle_left)
        print("右脚角度,", knee_angle_right)

        # 计算脚踝之间的距离
        ankle_distance = abs(left_ankle_x - right_ankle_x)
        # 计算膝盖之间的距离
        knee_distance = abs(left_knee_x - right_knee_x)
        print("脚踝之间的距离,", ankle_distance)
        print("膝盖之间的距离,", knee_distance)

        # 判断逻辑
        flag_0 = left_ankle_y > left_knee_y > left_hip_y and right_ankle_y > right_knee_y > right_hip_y
        flag_1 = knee_angle_left < 30 and knee_angle_right < 30
        flag_2 = ankle_distance > knee_distance * 1.5
        flag_3 = ankle_distance < knee_distance * 1.5

        # 如果左脚踝和右脚踝的 y 坐标都大于膝盖和髋关节的 y 坐标，并且膝盖之间的角度小于一定阈值，判定为站立
        if (flag_0 and flag_1 and flag_3):
            return '站立'

        # 如果脚踝之间的距离大于膝盖之间的距离，判定为行走
        if flag_2:
            return '行走'

        # 如果左脚踝和右脚踝的 y 坐标都大于髋关节的 y 坐标，并且膝盖之间的角度大于一定阈值，判定为跳跃
        if left_ankle_y > left_hip_y and right_ankle_y > right_hip_y:
            if knee_angle_left > 60 or knee_angle_right > 60:  # 设置角度阈值为160度
                return '跳跃'

    # 如果以上条件都不满足，返回'unknown'
    return '未知'

def draw_chinese_text(image, text, position, font_path, font_size, color):
    """
    在图像上绘制中文文本
    :param image: OpenCV 图像
    :param text: 要绘制的中文文本
    :param position: 文本绘制位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 绘制中文文本后的图像
    """
    # 将 OpenCV 图像转换为 PIL 图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # 绘制中文文本
    draw.text(position, text, font=font, fill=color)

    # 将 PIL 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image

if __name__ == '__main__':

    model = YOLO('./weights/yolo11n-pose.pt')  ### Pre-trained weights

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Open the video file
    # video_path = "./test_video/test.mp4"
    # video_path = 0
    # video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/101" # 外走廊高清  1920 1080
    video_path = "rtsp://admin:HikFIATCT@192.168.50.11:554/Streaming/Channels/102" # 外走廊标清 640 360
    # video_path = "rtsp://admin:HikNJQXFP@192.168.50.10:554/Streaming/Channels/102" # 屋内大屏摄像头
    # video_path = "rtsp://admin:Dxw202409@192.168.50.20:554/stream2"  # 15fps 640 480
    cap = cv2.VideoCapture(video_path)

    # Write the video file
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 640 x 360
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频的分辨率（宽度和高度）
    print("fps, w, h: ", fps, w, h)
    output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4"  # 设置输出视频的保存路径
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)  # 创建一个VideoWriter对象用于写视频

    # Loop through the video frames
    while cap.isOpened():

        # 获取当前系统时间
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], device=0)

            if results[0].boxes.is_track is True:
                # Get the boxes and track IDs and keypoints
                boxes = results[0].boxes.xywh.cuda()
                track_ids = results[0].boxes.id.int().cuda().tolist()
                keypoints = results[0].keypoints.xy.cuda().tolist()

                # Visualize the results on the frame
                frame = results[0].plot(conf=False)

                for box, track_id, keypoint in zip(boxes, track_ids, keypoints):
                    # box: tensor
                    x, y, box_w, box_h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=5)

                    # 根据人体关键点判断姿态
                    pose = determine_pose(keypoint)

                    frame = draw_chinese_text(frame, pose, (50, 50), "./STSONG.TTF", 24, (0, 255, 0))

            # 实时显示系统时间
            # cv2.putText(frame, current_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Write the video frame
            output_video.write(frame)

            # Display the annotated frame
            cv2.imshow("Posing", frame)

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