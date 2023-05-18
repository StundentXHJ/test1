import json
import os
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QPushButton, QMessageBox, QFileDialog
from PyQt5.QtCore import pyqtSignal
import numpy as np

from corrector import Corrector
from detector import Detector
from mainui import Ui_MainWindow as MainWindow
from config import Ui_MainWindow as ConfigWindow
import imageio


class DrawThread(QThread):
    def __init__(self, video_path, background_start, background_end, count_start, count_end, img_signal,
                 video_origin_shape, loc_points,
                 video_total_length, display_type, process_signal):
        super(DrawThread, self).__init__()
        self.video_path = video_path
        self.video_name = os.path.split(self.video_path)[-1].split('.')[0]
        self.background_start = background_start
        self.background_end = background_end
        self.count_start = count_start
        self.count_end = count_end
        self.video_origin_shape = video_origin_shape
        self.video_total_length = video_total_length
        self.loc_points = loc_points
        self.corrector = Corrector(self.video_path, self.background_start, self.background_end)

        self.img_signal = img_signal
        self.process_signal = process_signal
        self.processing = True
        # 0生成,1展示
        self.display_type = display_type
        self.detector = Detector(self.loc_points, self.video_path,
                                 self.video_origin_shape) if self.display_type == 0 else Detector.load(
            f'./data/{self.video_name}.pkl')

    def run(self) -> None:
        if self.display_type == 0:
            handled_finish = True
            cap = cv2.VideoCapture(self.video_path)
            bar_length = min((self.count_end - self.count_start), (self.video_total_length - self.count_start))
            self.process_signal.emit(bar_length, 2)
            able_idx = 0
            for frame_idx in range(self.video_total_length):
                print(frame_idx)
                if self.processing is False:
                    handled_finish = False
                    break
                if frame_idx < self.count_start:
                    continue
                if frame_idx > self.count_end:
                    break
                ret, frame = cap.read()
                if ret is not True:
                    handled_finish = False
                    self.img_signal.emit(np.zeros(1), {}, 0)
                    break
                corrected_frame = self.corrector.transform(frame)
                detected_frame, analysis_data = self.detector.detect(corrected_frame, frame_idx)
                resized_data = cv2.resize(detected_frame, (640, 480))
                self.img_signal.emit(resized_data, analysis_data, 1)
                self.process_signal.emit(able_idx + 1, 3)
                able_idx += 1
            if handled_finish is True:
                self.detector.dump()
            self.img_signal.emit(np.zeros(1), {}, 2)
        else:
            cap = cv2.VideoCapture(self.video_path)
            step = 0
            able_steps = self.detector.steps
            able_step = 0
            bar_length = max(len(able_steps) - 1, 0)
            self.process_signal.emit(bar_length, 2)
            while self.processing:
                if able_step >= bar_length:
                    break
                ret, frame = cap.read()
                if not step in able_steps:
                    step += 1
                    continue
                if ret is not True:
                    self.img_signal.emit(np.zeros(1), {}, 0)
                    break
                corrected_frame = self.corrector.transform(frame)
                ret = self.detector.get_display_info(corrected_frame, step)
                detected_frame, analysis_data = ret
                resized_data = cv2.resize(detected_frame, (640, 480))
                self.img_signal.emit(resized_data, analysis_data, 1)
                self.process_signal.emit(able_step + 1, 3)
                step += 1
                able_step += 1

            self.img_signal.emit(np.zeros(1), {}, 2)

    def stop(self):
        self.processing = False


class App(MainWindow):
    img_signal = pyqtSignal(np.ndarray, dict, int)

    process_signal = pyqtSignal(int, int)

    def __init__(self):
        super(App, self).__init__()
        self.setupUi()
        self.setupAction()

    # 定义动作
    def setupAction(self):
        self.config_window = ConfigWindow()
        self.config_window.setupUi()
        self.config_window.hide()
        self.processing = False
        self.video_loaded = False
        self.video_path = ''
        self.video_handled = False
        self.process_thread = None
        self.background_start = 0
        self.background_end = 999999
        self.count_start = 0
        self.count_end = 999999
        self.origin_video_shape = None
        self.video_total_length = None
        self.loc_points = {}

        self.img_signal.connect(self.display)
        self.process_signal.connect(self.process_bar_action)
        self.pushButton_2.clicked.connect(self.process)
        self.toolButton.clicked.connect(self.open_file)
        self.pushButton_3.clicked.connect(self.stop_processing)
        self.pushButton.clicked.connect(self.click_config_show)
        self.pushButton_4.clicked.connect(self.process_display_only)
        self.config_window.pushButton.clicked.connect(self.config_ok)
        self.config_window.pushButton_2.clicked.connect(self.config_cancel)

        self.progressBar.setValue(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(1)

        self.label_6.setStyleSheet('#label_6{font-size:30px;color:red;font-family:simhei;}')
        self.label_6.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

    def open_file(self):
        self.video_path = ''
        self.video_loaded = False
        self.video_handled = False
        self.progressBar.setValue(0)
        self.label_3.setText('')
        self.stop_draw_thread()
        video_path = QFileDialog.getOpenFileName(self, '请选择视频文件', './videos', 'Video files (*.avi *.mp4 *.mpeg)')[0]
        if video_path == '':
            return
        video_name = os.path.split(video_path)[-1].split('.')[0]
        point_json_path = f'./videos/{video_name}_cnts.json'
        if not os.path.exists(point_json_path):
            QtWidgets.QMessageBox.warning(self, '警告', f'视频计数区域配置文件不存在，请检查!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        try:
            loc_points = json.load(open(point_json_path, 'r', encoding='utf-8'))
            self.loc_points = loc_points
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'视频计数区域配置文件载入错误，请检查文件格式!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        try:
            vdata = imageio.get_reader(video_path, 'ffmpeg')
            self.origin_video_shape = vdata.get_next_data().shape[:2]
            self.video_total_length = vdata.count_frames()
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'视频载入错误，请检查视频格式!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return

        video_name = os.path.split(video_path)[-1].split('.')[0]
        if os.path.exists(f'./data/{video_name}.pkl') and os.path.exists(f'./data/{video_name}.jpg'):
            self.video_handled = True
        self.video_loaded = True
        self.video_path = video_path
        self.label_3.setText(self.video_path)

    def process(self):
        if self.video_handled is True:
            retBtn = QtWidgets.QMessageBox.question(self, '警告', f'该视频已经处理过,是否重新处理?',
                                                    buttons=QMessageBox.Yes | QMessageBox.No,
                                                    defaultButton=QMessageBox.No)
            if retBtn == QMessageBox.No:
                return
        if self.video_loaded is not True:
            QtWidgets.QMessageBox.warning(self, '警告', f'请先选择处理视频再试!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.processing:
            QtWidgets.QMessageBox.warning(self, '警告', f'处理中,请等待处理完毕后重试!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        self.processing = True
        self.process_thread = DrawThread(self.video_path, self.background_start, self.background_end, self.count_start,
                                         self.count_end, self.img_signal,
                                         self.origin_video_shape, self.loc_points, self.video_total_length, 0,
                                         self.process_signal)
        self.process_thread.start()

    def process_display_only(self):
        if self.video_loaded is not True:
            QtWidgets.QMessageBox.warning(self, '警告', f'请先选择视频再试!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.video_handled is not True:
            QtWidgets.QMessageBox.warning(self, '警告', f'该视频未经处理,请处理后再试!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.processing:
            QtWidgets.QMessageBox.warning(self, '警告', f'处理中,请等待处理结束重试!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        self.processing = True
        self.process_thread = DrawThread(self.video_path, self.background_start, self.background_end, self.count_start,
                                         self.count_end, self.img_signal,
                                         self.origin_video_shape, self.loc_points, self.video_total_length, 1,
                                         self.process_signal)
        self.process_thread.start()

    def display(self, draw_frame, analysis_data, error_code):
        if error_code == 1:
            print(1)
            draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            draw_frame = QtGui.QImage(draw_frame.data, draw_frame.shape[1], draw_frame.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(draw_frame))
            self.textBrowser.setText(self.data_to_str(analysis_data))
        elif error_code == 2:
            print(2)
            self.stop_draw_thread()
            self.label.setPixmap(QPixmap(''))
            self.textBrowser.setText('结束')
            self.video_handled = True
            print(3)
        elif error_code == 0:
            QtWidgets.QMessageBox.warning(self, '警告', f'视频载入错误，请检查视频格式!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            self.stop_draw_thread()
            self.label.setPixmap(QPixmap(''))
            self.textBrowser.setText('')

    def stop_draw_thread(self):
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None
            self.processing = False

    def stop_processing(self):
        if not self.process_thread is None:
            self.stop_draw_thread()

    def click_config_show(self):
        self.stop_draw_thread()
        self.processing = False
        self.video_loaded = False
        self.video_path = ''
        self.video_handled = False
        self.process_thread = None
        self.origin_video_shape = None
        self.video_total_length = None
        self.loc_points = {}
        self.config_window.lineEdit.setText(str(self.background_start))
        self.config_window.lineEdit_2.setText(str(self.background_end))
        self.config_window.lineEdit_4.setText(str(self.count_start))
        self.config_window.lineEdit_3.setText(str(self.count_end))
        self.label.setPixmap(QPixmap(''))
        self.label_3.setText('')
        self.textBrowser.setText('')
        self.hide()
        self.config_window.show()

    def config_ok(self):
        try:
            background_start = int(self.config_window.lineEdit.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'背景生成开始帧数输入有误,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        try:
            background_end = int(self.config_window.lineEdit_2.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'背景生成结束帧数输入有误,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        try:
            count_start = int(self.config_window.lineEdit_4.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'统计开始帧数输入有误,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        try:
            count_end = int(self.config_window.lineEdit_3.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, '警告', f'统计结束帧数输入有误,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        if background_end < background_start:
            QtWidgets.QMessageBox.warning(self, '警告', f'背景生成结束帧数需要大于开始帧,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        if count_end < count_start:
            QtWidgets.QMessageBox.warning(self, '警告', f'统计结束帧数需要大于开始帧,请重新输入!',
                                          buttons=QtWidgets.QMessageBox.Ok)
            return
        self.background_start = background_start
        self.background_end = background_end
        self.count_start = count_start
        self.count_end = count_end
        self.config_window.hide()
        self.show()

    def config_cancel(self):
        self.config_window.lineEdit.setText(str(self.background_start))
        self.config_window.lineEdit_2.setText(str(self.background_end))
        self.config_window.lineEdit_4.setText(str(self.count_start))
        self.config_window.lineEdit_3.setText(str(self.count_end))

    def data_to_str(self, data_dict: dict):
        result = ''
        for gate_name, data in data_dict.items():
            result += f'{gate_name}:\n'
            for method, record in data.items():
                result += f'    {method}: {len(record)}\n'
        return result

    def process_bar_action(self, value, code):
        if code == 1:
            self.progressBar.setMinimum(value)
        elif code == 2:
            self.progressBar.setMaximum(value)
        elif code == 3:
            self.progressBar.setValue(value)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = App()
    mainwindow.setFixedSize(1246, 756)
    mainwindow.show()
    sys.exit(app.exec_())
