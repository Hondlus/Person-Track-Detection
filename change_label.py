import os


txt_content = []

txt_path = "C:/Users/DXW/Desktop/yolo_train/datasets/sleep/labels/val/"

for txt_name in os.listdir(txt_path):
    # 读取内容
    with open(txt_path + txt_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = "1" + ' ' + line[2:]
            txt_content.append(line)

    # 写入内容
    with open(txt_path + txt_name, "w") as f:
        f.writelines(txt_content)

    # print(txt_content)
    txt_content.clear()
    # print(txt_content)