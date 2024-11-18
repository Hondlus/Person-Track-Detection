from ultralytics import YOLO
import cv2


# person标签为0
class_id = 0

# Load a model
model = YOLO("./weights/yolo11x.pt")  # load an official model

img = cv2.imread("./test_video/duoren.jpeg")
height, width, _ = img.shape
print("img_h, img_w: ", height, width)

# Predict with the model
results = model(img, classes=[0], device=0)  # predict on an image

img = results[0].plot(labels=False)

# 可视化进行标注框观察
resize_img = cv2.resize(img, (240, 320))
cv2.imshow("Detecting", resize_img)

if cv2.waitKey(0) & 0xff == ord("d"):
    pass

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

        with open("./result.txt", "a") as f:
            f.write("{}".format(class_id) + ' ' + str(guiyi_x) + ' ' + str(guiyi_y) + ' ' + str(guiyi_w) + ' ' + str(guiyi_h))
            # f.write("{}".format(class_id) + ' ' + '%0.4f'%str(guiyi_x) + ' ' + '%0.4f'%str(guiyi_y) + ' ' + '%0.4f'%str(guiyi_w) + ' ' + '%0.4f'%str(guiyi_h))
            f.write("\n")
    f.close()

cv2.destroyAllWindows()