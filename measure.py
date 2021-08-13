import numpy as np
import cv2

def measure(img, gtd, epoch, iter, area_thr = 10, width = 10.5):
    new_gtd_contours = []
    TP = 0
    FP = 0
    FN = 0
    # otsu_img = np.clip(img, 0, 1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, otsu_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_contours, _ = cv2.findContours(otsu_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fake_img = np.zeros(img.shape, dtype=np.uint8)
    fake_img = cv2.drawContours(fake_img, img_contours, -1, [255,255,255], -1)
    #cv2.imwrite(f'./check_measure_img/fake_img/{epoch}_{iter}.png', fake_img)

    gtd = cv2.GaussianBlur(gtd, (1, 1), 0)
    _, otsu_gtd = cv2.threshold(gtd, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gtd_contours, _ = cv2.findContours(otsu_gtd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if np.sum(otsu_gtd) > 0:
        for i in range(len(gtd_contours)):
            gtd_img = np.zeros(img.shape, np.uint8)

            area_gtd_line = cv2.contourArea(gtd_contours[i])
            new_gtd_contours.append(gtd_contours[i])

            gtd_blob = cv2.drawContours(gtd_img, [gtd_contours[i]], -1, 1, -1)
            w = 0
            if np.sum(gtd_blob) != 0:
                dilate_gtd = gtd_blob
                if area_gtd_line > area_thr:
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                while (w <= width):
                    dilate_gtd = cv2.dilate(dilate_gtd, kernel, iterations=1)
                    dilate_gtd_contour, _ = cv2.findContours(dilate_gtd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(dilate_gtd_contour[0]) < 5:
                        break
                    _, (MajorAxis, MinorAxis), _ = cv2.fitEllipse(dilate_gtd_contour[0])
                    w = MajorAxis

                    if (w >= width):
                        dilate_gtd = cv2.erode(dilate_gtd, kernel, iterations=1)
                        dilate_gtd_contour, _ = cv2.findContours(dilate_gtd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if len(dilate_gtd_contour[0]) < 5:
                            break
                        _, (MajorAxis, MinorAxis), _ = cv2.fitEllipse(dilate_gtd_contour[0])
                        w = MajorAxis
                        break

            dilate_gtd_draw = np.zeros(img.shape, np.uint8)
            dilate_gtd_draw = cv2.drawContours(dilate_gtd_draw, dilate_gtd_contour, -1, 1, -1)
            for j in range(len(img_contours)):
                pred_blob_area = cv2.contourArea(img_contours[j])
                if pred_blob_area > area_thr:

                    pred_blob = np.zeros(img.shape, np.uint8)
                    pred_blob = cv2.drawContours(pred_blob, [img_contours[j]], -1, 1, -1)

                    out_blob = cv2.subtract(pred_blob, dilate_gtd_draw)

                    if np.array_equal(pred_blob, out_blob) != True:
                        out_blob_contour, _ = cv2.findContours(out_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        pred_rest_blob = pred_blob
                        if len(out_blob_contour) != 0:
                            for h in range(len(out_blob_contour)):
                                out_blob_draw = np.zeros(img.shape, np.uint8)
                                out_blob_draw = cv2.drawContours(out_blob_draw, out_blob_contour, -1, 1, 1)
                                out_area = cv2.contourArea(out_blob_contour[h])

                                if out_area > area_thr:
                                    FP += 1
                                pred_rest_blob = cv2.subtract(pred_rest_blob, out_blob_draw)

                        rest_blob = cv2.subtract(gtd_blob, pred_rest_blob)
                        rest_blob_contour, _ = cv2.findContours(rest_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        rest_blob_draw = np.zeros(img.shape, np.uint8)
                        rest_blob_draw = cv2.drawContours(rest_blob_draw, rest_blob_contour, -1, [255,255,255], -1)
                        #cv2.imwrite(f'./check_measure_img/overlap/{epoch}_{iter}_{i}.png', rest_blob_draw)
                        if np.array_equal(rest_blob_draw, gtd_blob) != True:
                            rest_blob_sum = 0
                            for k in range(len(rest_blob_contour)):
                                rest_area_blob = cv2.contourArea(rest_blob_contour[k])
                                rest_blob_sum += rest_area_blob
                            if area_gtd_line != 0:
                                overlap_line = area_gtd_line - rest_blob_sum
                                overlap_rate = (overlap_line / area_gtd_line) * 100
                                if overlap_rate >= 50:
                                    TP += 1
                                else:
                                    FP += 1

        FN = len(gtd_contours) - TP

    else:
        if np.sum(otsu_img) == 0:
            TP += 1
        else:
            true_p = 0
            for j in range(len(img_contours)):
                pred_blob_area = cv2.contourArea(img_contours[j])
                if pred_blob_area > area_thr:
                    FP += 1
                else:
                    true_p += 1
            if len(img_contours) == true_p:
                TP += 1

    # if (TP + FN != 0):
    #     recall = (TP / (TP + FN)) * 100
    # else:
    #     recall = 0
    #
    # if (TP + FP != 0):
    #     FDR = (FP / (TP + FP))
    #     precision = (1 - FDR)
    #     FDR = FDR * 100
    #     precision = precision * 100
    # else:
    #     FDR = 0
    #     precision = 0
    # if recall != 0 or precision != 0:
    #     F1_score = (2 * ((recall * precision) / (recall + precision)))
    # else:
    #     F1_score = 0

    return TP, FP, FN, _

