from ultralytics import YOLO
from collections import defaultdict
import cv2
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from datetime import datetime, timedelta
import re
from ultralytics.utils import LOGGER, colorstr
from openpyxl import Workbook

# check_requirements('shapely>=2.0.0')


class ObjectCounter:

    def __init__(self):
        self.is_drawing = False
        self.selected_point = None
        self.reg_pts = None
        self.counting_region = None
        self.region_color = (255, 255, 255)
        self.im0 = None
        self.tf = None
        self.view_img = True
        self.names = None
        self.annotator = None
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = True
        self.env_check = check_imshow(warn=True)
        self.incount_label = 'InCount: 0'
        self.outcount_label = 'OutCount: 0'
        self.k = 1

    def set_args(self, classes_names, reg_pts, region_color=None, line_thickness=2, track_thickness=2, view_img=True,
                 draw_tracks=True):
        self.tf = line_thickness
        self.view_img = view_img
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.reg_pts = reg_pts
        self.counting_region = Polygon(self.reg_pts)
        self.names = classes_names
        self.region_color = region_color if region_color else self.region_color

    def mouse_event_for_region(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if isinstance(point, (tuple, list)) and len(point) >= 2:
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                        self.selected_point = i
                        self.is_drawing = True
                        break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        # boxes = tracks[0].boxes.xyxy.cpu()
        # clss = tracks[0].boxes.cls.cpu().tolist()
        # track_ids = tracks[0].boxes.id.int().cpu().tolist()
        boxes = tracks[0].boxes.xyxy.cuda()
        clss = tracks[0].boxes.cls.cuda().tolist()
        track_ids = tracks[0].boxes.id.int().cuda().tolist()
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color)
        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(int(cls), True))
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(track_line, color=(0, 255, 0),
                                                        track_thickness=self.track_thickness)
            if self.counting_region.contains(Point(track_line[-1])):
                if track_id not in self.counting_list:
                    self.counting_list.append(track_id)
                    if box[0] < self.counting_region.centroid.x:
                        self.out_counts += 1
                    else:
                        self.in_counts += 1
        if self.env_check and self.view_img:
            if swap_in_and_out:
                self.incount_label = 'InCount: ' + f'{self.out_counts}'
                self.outcount_label = 'OutCount: ' + f'{self.in_counts}'
            else:
                self.incount_label = 'InCount: ' + f'{self.in_counts}'
                self.outcount_label = 'OutCount: ' + f'{self.out_counts}'
            if now_time == start_time + timedelta(seconds=self.k * record_interval_time):
                sheet.append([str(now_time), int(self.incount_label[9:]), int(self.outcount_label[10:])])
                LOGGER.info(
                    f"{colorstr('magenta', 'Time: ')}{colorstr('yellow', now_time)}. {colorstr('cyan', self.incount_label + ' persons')}, {colorstr('blue', self.outcount_label + ' persons')}")
                self.k += 1
            self.annotator.count_labels(in_count=self.incount_label, out_count=self.outcount_label)
            cv2.namedWindow('Object Counter Powered By Anperlanch')
            cv2.setMouseCallback('Object Counter Powered By Anperlanch', self.mouse_event_for_region,
                                 {'region_points': self.reg_pts})
            cv2.imshow('Object Counter Powered By Anperlanch', self.im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def start_counting(self, im0, tracks):
        self.im0 = im0
        if tracks[0].boxes.id is None:
            cv2.putText(self.im0, 'No person detected. ' + self.incount_label + ', ' + self.outcount_label, (400, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), self.tf)
            cv2.imshow('Object Counter Powered By Anperlanch', self.im0)
            if now_time == start_time + timedelta(seconds=self.k * record_interval_time):
                sheet.append([str(now_time), int(self.incount_label[9:]), int(self.outcount_label[10:])])
                LOGGER.info(
                    f"{colorstr('magenta', 'Time: ')}{colorstr('yellow', now_time)}. {colorstr('cyan', self.incount_label + ' persons')}, {colorstr('blue', self.outcount_label + ' persons')}. {colorstr('red', 'No person detected')}")
                self.k += 1
            return self.im0
        else:
            self.extract_and_process_tracks(tracks)
            return self.im0


def main():
    global start_time, now_time, sheet
    workbook = Workbook()
    sheet = workbook.active
    year, month, day, hours, minutes, seconds = map(int, re.split('[-:\s]+', video_start_time))
    start_time = datetime(year, month, day, hours, minutes, seconds)
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    x, y, height, width = 0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2), 30, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_writer = cv2.VideoWriter("runs/in_and_out_counting.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (int(cap.get(3)), int(cap.get(4))))
    counter = ObjectCounter()
    counter.set_args(view_img=view_img, line_thickness=line_thickness,
                     reg_pts=[(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
                     classes_names=model.names, region_color=region_color, track_thickness=track_thickness,
                     draw_tracks=draw_tracks)
    vid_frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        vid_frame_count += 1
        now_time = start_time + timedelta(seconds=int(vid_frame_count / fps))
        cv2.putText(frame, 'Frame time: ' + str(now_time), (120, 27), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                    line_thickness)
        tracks = model.track(source=frame, persist=True, classes=classes)
        frame = counter.start_counting(frame, tracks)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    workbook.save(r"runs\in_and_out_counting.xlsx")
    video_writer.release()


if __name__ == '__main__':
    video_path = "test_video/test.mp4"
    # video_path = 0
    # video_start_time = '2021-08-09 14:07:25'
    video_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    weights = r"./weights/yolov8x.pt"
    device = '0'
    swap_in_and_out = True
    record_interval_time = 1
    classes = 0
    view_img = True
    draw_tracks = True
    line_thickness = 2
    track_thickness = 1
    region_color = (255, 0, 0)
    main()