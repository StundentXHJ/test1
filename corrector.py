import os

import numpy as np
import cv2


class Corrector:
    def __init__(self, video_path, background_start, background_end):
        self.video_path = video_path
        self.video_name = os.path.split(self.video_path)[-1].split('.')[0]
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher()
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.template_img = None
        if not os.path.exists(f'./data/{self.video_name}.jpg'):
            self.__backgroud_generate(background_start, background_end)
        else:
            self.__set_template_img(f'./data/{self.video_name}.jpg')

    def __set_template_img(self, pic_path: str):
        template_img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.template_img = template_img

    # 直方图规定化
    def __hist_correct(self, src_bgr_img: np.ndarray, target_bgr_img: np.ndarray):
        color = ('h', 's', 'v')
        for i, col in enumerate(color):
            # histr = cv2.calcHist([img_hsv1], [i], None, [256], [0, 256])
            hist1, bins = np.histogram(src_bgr_img[:, :, i].ravel(), 256, [0, 256])
            hist2, bins = np.histogram(target_bgr_img[:, :, i].ravel(), 256, [0, 256])
            cdf1 = hist1.cumsum()  # 灰度值0-255的累计值数组
            cdf2 = hist2.cumsum()
            cdf1_hist = hist1.cumsum() / cdf1.max()  # 灰度值的累计值的比率
            cdf2_hist = hist2.cumsum() / cdf2.max()
            diff_cdf = [[0 for j in range(256)] for k in range(256)]  # diff_cdf 里是每2个灰度值比率间的差值
            for j in range(256):
                for k in range(256):
                    diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])
            lut = [0 for j in range(256)]  # 映射表
            for j in range(256):
                min = diff_cdf[j][0]
                index = 0
                for k in range(256):  # 直方图规定化的映射原理
                    if min > diff_cdf[j][k]:
                        min = diff_cdf[j][k]
                        index = k
                lut[j] = ([j, index])
            h = int(src_bgr_img.shape[0])
            w = int(src_bgr_img.shape[1])
            for j in range(h):  # 对原图像进行灰度值的映射
                for k in range(w):
                    src_bgr_img[j, k, i] = lut[src_bgr_img[j, k, i]][1]
        return src_bgr_img

    # 获得ORB特征
    def __get_des(self, bgr_img: np.ndarray):
        kps = self.orb.detect(bgr_img, None)
        kps, des = self.orb.compute(bgr_img, kps)
        return kps, des

    def transform(self, bgr_img: np.ndarray):
        good_matches = []
        template_kps, template_des = self.__get_des(self.template_img)
        cur_img_kps, cur_img_des = self.__get_des(bgr_img)
        matches = self.matcher.knnMatch(template_des, cur_img_des, k=2)
        if len(matches) == 0 or len(matches[0]) != 2:
            return []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append([m])
        # 目标坐标
        ptsA = np.float32([template_kps[match[0].queryIdx].pt for match in good_matches])
        # 原始坐标
        ptsB = np.float32([cur_img_kps[match[0].trainIdx].pt for match in good_matches])
        M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)

        h, w = bgr_img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 校正后的图像
        imgOut = cv2.warpPerspective(bgr_img, M, (self.template_img.shape[1], self.template_img.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return imgOut
        # return bgr_img

    def __backgroud_generate(self, background_start, background_end):
        print('生成背景图中...')
        capture = cv2.VideoCapture(self.video_path)
        bmog = cv2.createBackgroundSubtractorMOG2()
        gmog = cv2.createBackgroundSubtractorMOG2()
        rmog = cv2.createBackgroundSubtractorMOG2()
        ret, backimage = capture.read()
        bbackimage, gbackimage, rbackimage = cv2.split(backimage)
        self.template_img = backimage.copy()
        count = 0
        while True:
            ret, image = capture.read()
            if ret is True:
                if not background_start < count < background_end:
                    count += 1
                    continue
                image = self.transform(image)
                bimage, gimage, rimage = cv2.split(image.copy())
                bfgmask = bmog.apply(bimage)
                bfgmask_mask = bfgmask == 255
                bfgmask[bfgmask_mask] = 1
                bfgmask[~bfgmask_mask] = 0
                bback = bmog.getBackgroundImage()
                bbackimage = bbackimage * bfgmask + bback * (1 - bfgmask)

                gfgmask = gmog.apply(gimage)
                gfgmask_mask = gfgmask == 255
                gfgmask[gfgmask_mask] = 1
                gfgmask[~gfgmask_mask] = 0
                gback = gmog.getBackgroundImage()
                gbackimage = gbackimage * gfgmask + gback * (1 - gfgmask)

                rfgmask = rmog.apply(rimage)
                rfgmask_mask = rfgmask == 255
                rfgmask[rfgmask_mask] = 1
                rfgmask[~rfgmask_mask] = 0
                rback = rmog.getBackgroundImage()
                rbackimage = rbackimage * rfgmask + rback * (1 - rfgmask)

                backimage = cv2.merge([bbackimage, gbackimage, rbackimage])
                count += 1
                print(count)
            else:
                break
        cv2.imwrite(f'./data/{self.video_name}.jpg', backimage)
