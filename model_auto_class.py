import os
from ultralytics import YOLO
import cv2

# 0: person
class_id = 0

# 注意：路径最后要带斜杠
img_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/images/train/"
# 存储图片的路径
image_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/sleep/"
image_path2 = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/play/"

# Load a model
model = YOLO("./weights/yolo11x.pt")  # load an official model

if __name__ == '__main__':
    count = 0
    # 批量读取图片
    for img_name in os.listdir(img_path):
        img = cv2.imread(img_path + img_name)
        # height, width, _ = img.shape
        # print("img_h, img_w: ", height, width)

        # Predict with the model
        results = model(img, classes=[0], device=0)  # predict on an image
        boxes = results[0].boxes.xywh.cuda()
        for box in boxes:
            x, y, box_w, box_h = box  # x,y center point
            xmin = int(x) - int(box_w / 2)
            ymin = int(y) - int(box_h / 2)
            xmax = int(x) + int(box_w / 2)
            ymax = int(y) + int(box_h / 2)
            crop_img = img[ymin:ymax, xmin:xmax]
            # 根据标注信息画框
            cv2.imshow('crop_img', crop_img)

            # 按下d键继续下一张图片
            if cv2.waitKey(0) & 0xff == ord("s"):
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                cv2.imwrite(image_path + str(count) + ".jpg", crop_img)
                count += 1

            # 按下s键保存标注框坐标到txt文件
            if cv2.waitKey(0) & 0xff == ord("p"):
                if not os.path.exists(image_path2):
                    os.makedirs(image_path2)
                cv2.imwrite(image_path2 + str(count) + ".jpg", crop_img)
                count += 1

        ## 按下q键退出程序
        if cv2.waitKey(0) & 0xff == ord("q"):
            break

        cv2.destroyAllWindows()