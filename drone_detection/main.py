import cv2
from drone_detection.object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture("./test.mp4")

while True:
    _, frame = cap.read()
    (class_ids, scores, boxes) = od.detect(frame)
    for class_id, box in zip(class_ids, boxes):
        if class_id in [2, 5, 7]:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("chen", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()