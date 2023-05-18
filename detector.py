import json
import os
import threading

import cv2

from drone_detection.object_detection import ObjectDetection
import numpy as np

from point import Point
import pickle as pkl


class Detector:
    def __init__(self, loc_points: dict, video_path: str, origin_shape: tuple):
        self.loc_points = loc_points
        self.video_path = video_path
        self.origin_shape = origin_shape
        self.video_name = os.path.split(self.video_path)[-1].split('.')[0]
        self.loc_cnts = {gate_name: (self.__get_cnt_by_point(x1, y1, x2, y2), in_angle) for
                         gate_name, ((x1, y1), (x2, y2), in_angle) in
                         self.loc_points.items()}
        self.loc_records = {gate_name: [] for gate_name, _ in self.loc_cnts.items()}
        self.points = []
        self.steps = []
        self.yolo_model = ObjectDetection(weights_path='./drone_detection/dnn_model/yolov4.weights',
                                          cfg_path='./drone_detection/dnn_model/yolov4.cfg')
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.old_gray = None
        self.colors = np.random.randint(0, 255, (1000, 3))

    @staticmethod
    def load(record_path: str):
        load_data = pkl.load(open(record_path, 'rb'))
        loc_points = load_data['loc_points']
        loc_cnts = load_data['loc_cnts']
        loc_records = load_data['loc_records']
        points = load_data['points']
        video_path = load_data['video_path']
        video_name = load_data['video_name']
        origin_shape = load_data['origin_shape']
        steps = load_data['steps']
        detector = Detector(loc_points=loc_points, video_path=video_path, origin_shape=origin_shape)
        detector.loc_points = loc_points
        detector.loc_cnts = loc_cnts
        detector.loc_records = loc_records
        detector.points = points
        detector.video_path = video_path
        detector.video_name = video_name
        detector.origin_shape = origin_shape
        detector.steps = steps
        return detector

    def __get_clock_angle(self, v1, v2):
        # 2个向量模的乘积
        TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
        # 叉乘
        rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
        # 点乘
        theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
        if rho < 0:
            theta = -theta
        return (theta + 360) % 360

    def detect(self, bgr_frame: np.ndarray, step: int):
        bgr_frame_yolo = cv2.resize(bgr_frame.copy(), (608, 608))
        draw_frame = bgr_frame.copy()
        class_ids, _, boxes = self.yolo_model.detect(bgr_frame_yolo)
        if self.old_gray is None:
            self.old_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        new_boxes = []
        for x, y, w, h in boxes:
            nx = int(x / 608 * bgr_frame.shape[1])
            nw = int(w / 608 * bgr_frame.shape[1])
            ny = int(y / 608 * bgr_frame.shape[0])
            nh = int(h / 608 * bgr_frame.shape[0])
            new_boxes.append((nx, ny, nw, nh))

        for class_id, (x, y, w, h) in zip(class_ids, new_boxes):
            if class_id not in [2, 5, 7]:
                continue
            mx = int(x + w / 2)
            my = int(y + h / 2)
            for point in self.points:
                ret = point.check_point((x, y, w, h))
                if ret is not None:
                    break
            else:
                # 新车
                self.points.append((Point(mx, my, step, class_id)))

        # 漂移
        for point in self.points:
            for x, y, w, h in new_boxes:
                if point.check_point((x, y, w, h)) is not None:
                    break
            else:
                point.set_disable()
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if len(self.points) != 0:
            p0 = np.float32(np.array([[[p.x, p.y]] for p in list(filter(lambda p: p.able, self.points))]))
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, p0, None, **self.lk_params)
            able_idx = 0
            for point in self.points:
                if not point.able:
                    continue
                newp = p1[able_idx]
                if st[able_idx] == 1:
                    newx, newy = newp.ravel()
                    point.add_history(newx, newy, step)
                else:
                    point.set_disable()
                able_idx += 1

        draw_frame = cv2.drawContours(draw_frame, list(list(map(lambda v: v[0], self.loc_cnts.values()))), -1,
                                      (0, 255, 255))
        for cnt_name, (cnt, in_angle) in self.loc_cnts.items():
            for point in self.points:
                if not point.able:
                    continue
                check_result = cv2.pointPolygonTest(cnt, (point.x, point.y), True)
                # 在轮廓内
                if check_result >= 0 and len(point.history) > 2 and point.id not in list(
                        map(lambda e: e[0], self.loc_records[cnt_name])):
                    start_point = (point.history[0][0], point.history[0][1])
                    end_point = (point.history[-1][0], point.history[-1][1])
                    angle = self.__get_clock_angle([0, -1],
                                                   [end_point[0] - start_point[0], end_point[1] - start_point[1]])
                    status = abs(in_angle - angle) < 90
                    self.loc_records[cnt_name].append((point.id, angle, status, step))
                    if point.out_gate_name is None:
                        point.in_gate_name = cnt_name
                    else:
                        point.out_gate_name = cnt_name

        self.old_gray = gray_frame.copy()

        for point_idx, point in enumerate(self.points):
            if not point.able:
                continue
            if len(point.history) >= 2:
                for (x1, y1, step1, _), (x2, y2, step2, _) in zip(point.history[1:], point.history[:-1]):
                    draw_frame = cv2.line(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          self.colors[point_idx].tolist(), 2)
            draw_frame = cv2.circle(draw_frame, (int(point.x), int(point.y)), 5, self.colors[point_idx].tolist(), -1)
        draw_frame = cv2.resize(draw_frame, self.origin_shape)
        print(self.loc_records)
        self.steps.append(step)
        return draw_frame, self.analysis(step)

    def __get_cnt_by_point(self, x1, y1, x2, y2):
        return np.array([
            [[int(x1), int(y1)]],
            [[int(x1), int(y2)]],
            [[int(x2), int(y2)]],
            [[int(x2), int(y1)]]
        ])

    def export(self):
        return {
            'loc_points': self.loc_points,
            'loc_cnts': self.loc_cnts,
            'loc_records': self.loc_records,
            'points': self.points,
            'video_path': self.video_path,
            'video_name': self.video_name,
            'origin_shape': self.origin_shape,
            'steps': self.steps
        }

    def dump(self):
        pkl.dump(
            {
                'loc_points': self.loc_points,
                'loc_cnts': self.loc_cnts,
                'loc_records': self.loc_records,
                'points': self.points,
                'video_path': self.video_path,
                'video_name': self.video_name,
                'origin_shape': self.origin_shape,
                'steps': self.steps
            }
            , open(f'./data/{self.video_name}.pkl', 'wb')
        )
        print(f'result dumped! to "./data/{self.video_name}.pkl"')

    def __get_point_by_id(self, point_id):
        for point in self.points:
            if point.id == point_id:
                return point

    def analysis(self, cur_idx):
        if np.max(np.array(self.steps)) < cur_idx:
            return False
        result_dict = {gate_name: {'直行': [], '左转弯': [], '右转弯': []} for gate_name in self.loc_records.keys()}
        for gate_name, records in self.loc_records.items():
            for point_id, angle, status, step in records:
                if step > cur_idx:
                    break
                point = self.__get_point_by_id(point_id)
                if abs(angle - self.loc_cnts[gate_name][1]) < 180:
                    if abs(angle - self.loc_cnts[gate_name][1]) < 30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['直行'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
                    elif angle - self.loc_cnts[gate_name][1] < -30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['右转弯'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
                    elif angle - self.loc_cnts[gate_name][1] > 30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['左转弯'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
                else:
                    if abs(angle - self.loc_cnts[gate_name][1]) < 30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['直行'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
                    elif angle - self.loc_cnts[gate_name][1] < -30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['右转弯'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
                    elif angle - self.loc_cnts[gate_name][1] > 30:
                        in_gate_name = point.in_gate_name
                        out_gate_name = point.out_gate_name
                        if in_gate_name is not None:
                            result_dict[in_gate_name]['左转弯'].append(point_id)
                        if out_gate_name is not None:
                            result_dict[out_gate_name]['直行'].append(point_id)
        return result_dict

    def get_display_info(self, bgr_frame, cur_idx):
        if np.max(np.array(self.steps)) < cur_idx:
            return False
        draw_frame = bgr_frame.copy()
        draw_frame = cv2.drawContours(draw_frame, list(list(map(lambda v: v[0], self.loc_cnts.values()))), -1,
                                      (0, 255, 255))
        for point_idx, point in enumerate(self.points):
            if point.history[0][2] > cur_idx:
                continue
            if point.history[-1][2] < cur_idx:
                continue
            draw_history = list(filter(lambda h: h[2] < cur_idx, point.history))
            if len(draw_history) >= 2:
                for (x1, y1, step1, _), (x2, y2, step2, _) in zip(draw_history[1:], draw_history[:-1]):
                    draw_frame = cv2.line(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          self.colors[point_idx].tolist(), 2)
            if len(draw_history) >= 1:
                draw_frame = cv2.circle(draw_frame, (int(draw_history[-1][0]), int(draw_history[-1][1])), 5,
                                        self.colors[point_idx].tolist(), -1)
        analysis_data = self.analysis(cur_idx)
        return draw_frame, analysis_data


if __name__ == '__main__':
    detector_load = Detector.load('./data/test.pkl')
    analysis_data = detector_load.analysis(10)
    print(analysis_data)
