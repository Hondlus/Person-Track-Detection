import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


def determine_pose(keypoints):
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

    frame = cv2.imread("./images/tiao.png")

    # Write the video file
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 640 x 360
    # w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频的分辨率（宽度和高度）
    # print("fps, w, h: ", fps, w, h)
    # output_video_path = "C:/Users/DXW/Desktop/yolo_track_reid/output_video/test.mp4"  # 设置输出视频的保存路径
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h), isColor=True)  # 创建一个VideoWriter对象用于写视频

    results = model.predict(frame, classes=[0], device=0)

    # Get the boxes and track IDs and keypoints
    boxes = results[0].boxes.xywh.cuda()
    keypoints = results[0].keypoints.data.cuda()

    # Visualize the results on the frame
    frame = results[0].plot(conf=False, kpt_radius=1)

    for box, keypoint in zip(boxes, keypoints):
        # box: tensor
        x, y, box_w, box_h = box

        # 根据人体关键点判断姿态
        pose = determine_pose(keypoint)

        frame = draw_chinese_text(frame, pose, (int(x-box_w/2), int(y-box_h/2)), "./STSONG.TTF", 14, (255, 0, 0))

    # Display the annotated frame
    cv2.imshow("Posing", frame)

    # Break the loop if 'q' is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()