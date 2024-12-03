import cv2
import os


txt_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/labels/train/"
img_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/images/train/"

for img_name in os.listdir(img_path):

    img = cv2.imread(img_path + img_name)
    img_height, img_width, _ = img.shape
    # print(img_height, img_width)

    txt_name = img_name.split(".")[0] + ".txt"
    with open(txt_path + txt_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, guiyi_x_center, guiyi_y_center, guiyi_width, guiyi_height = map(float, line.split())
            # print(label, guiyi_x_center, guiyi_y_center, guiyi_width, guiyi_height)
            xmin = int((guiyi_x_center - guiyi_width / 2) * img_width)
            ymin = int((guiyi_y_center - guiyi_height / 2) * img_height)
            label_width = int(guiyi_width * img_width)
            label_height = int(guiyi_height * img_height)
            # 根据标注信息画框
            cv2.rectangle(img, (xmin, ymin), (xmin + label_width, ymin + label_height), (0, 255, 0), thickness=4)

    img = cv2.resize(img, (320, 480))
    cv2.imshow("label_show", img)

    ## 按下d键下一张显示
    if cv2.waitKey(0) & 0xff == ord("d"):
        pass

    ## 按下q键退出程序
    if cv2.waitKey(0) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()

