import os
from ultralytics import YOLO
import cv2

# 0: person
class_id = 0

# 注意：路径最后要带斜杠
img_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/images/"
txt_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/labels/"

# Load a model
model = YOLO("./weights/yolo11x.pt")  # load an official model

if __name__ == '__main__':

    # 批量读取图片
    for img in os.listdir(img_path):
        img_name = img
        txt_name = img.split(".")[0] + ".txt"
        img = cv2.imread(img_path + img_name)
        height, width, _ = img.shape
        print("img_h, img_w: ", height, width)

        # Predict with the model
        results = model(img, classes=[0], device=0)  # predict on an image

        img = results[0].plot(labels=False)

        # 可视化观察标注框是否准确
        resize_img = cv2.resize(img, (240, 320))
        cv2.imshow(img_name, resize_img)

        # 按下d键继续下一张图片
        if cv2.waitKey(0) & 0xff == ord("d"):
            pass

        # 按下s键保存标注框坐标到txt文件
        if cv2.waitKey(0) & 0xff == ord("s"):
            boxes = results[0].boxes.xywh.cuda()
            for box in boxes:
                x, y, box_w, box_h = box  # x,y center point
                guiyi_x = float(x) / float(width)
                guiyi_y = float(y) / float(height)
                guiyi_w = float(box_w) / float(width)
                guiyi_h = float(box_h) / float(height)
                # print("x, y, w, h: ", x, y, box_w, box_h)
                # print("guiyi_x, guiyi_y, guiyi_w, guiyi_h: ", guiyi_x, guiyi_y, guiyi_w, guiyi_h)

                with open(txt_path + txt_name, "a") as f:
                    f.write("{}".format(class_id) + ' ' + str(guiyi_x) + ' ' + str(guiyi_y) + ' ' + str(guiyi_w) + ' ' + str(guiyi_h) + "\n")
                    # f.write("{}".format(class_id) + ' ' + '%0.4f'%str(guiyi_x) + ' ' + '%0.4f'%str(guiyi_y) + ' ' + '%0.4f'%str(guiyi_w) + ' ' + '%0.4f'%str(guiyi_h) + "\n")

        ## 按下q键退出程序
        if cv2.waitKey(0) & 0xff == ord("q"):
            break

        cv2.destroyAllWindows()