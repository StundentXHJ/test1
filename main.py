from corrector import Corrector
from detector import Detector
import cv2

if __name__ == '__main__':
    video_path = './data/test.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    img_h, img_w = img.shape[:2]
    corrector = Corrector(video_path, 1, 500)
    detector = Detector(
        loc_points={'左侧路口': ((724, 264), (662, 377), 325), '上方路口': ((1146, 264), (1465, 324), 225),
                    '右侧路口': ((1872, 725), (1865, 921), 135)},
        video_path=video_path, origin_shape=(img_w, img_h))
    count = 1
    while cap.isOpened():
        ret, rgb_frame = cap.read()
        if ret is not True:
            break
        transform_rgb_frame = corrector.transform(rgb_frame)
        detected_rgb_frame = detector.detect(transform_rgb_frame, count)
        cv2.imshow('Result', detected_rgb_frame)
        cv2.waitKey(10)
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    detector.dump()
