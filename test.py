import numpy as np
import cv2


# 鼠标回调函数
def draw_p0(event, x, y, a, b):
    global p0
    # 最后终点
    if event == cv2.EVENT_LBUTTONUP:
        # 当前次坐标点绘制结束坐标点，结束鼠标移动监听
        p0 = np.vstack([p0, np.array([[[x, y]]], dtype=np.float32)])


cap = cv2.VideoCapture('./data/test.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = np.zeros((0, 1, 2), dtype=np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if ret is not True:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.setMouseCallback('frame', draw_p0)
    frame = cv2.putText(frame, str(len(p0)), (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    if len(p0) > 0:
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(p1, p0)):
            if st[i] == 1:
                a, b = new.ravel()
                c, d = old.ravel()
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        frame = cv2.add(frame, mask)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = p1
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
