#!/usr/bin/python
# -*- coding: utf-8 -*-                                        
import numpy as np
import cv2

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

class Features2DTracker(object):
    def __init__(self, kp, desc):
        """kp - координаты особых точек, desc - описания"""
        self.kp = kp
        self.desc = desc
    
    def MatchedImageFeature(self, img2desc, img2kp, threshold=0.8):
        """
        img2desc, img2kp - ключевые точки цели 
        threshold - порог различия
        
        возвращает кортеж вида (m1, m2), где m1 - координаты совпавших точек на 
        себе, m2 - на цели 
        """
        match_result = []
        # сравниваем все особые точки:
        for i in range(len(self.desc)):
            A = img2desc - self.desc[i]
            dist = np.sqrt((A*A).sum(-1))
            n1, n2 = dist.argsort()[:2]
            # если отношение расстояний между текущей точкой и двумя ближайшими 
            # на цели не превышает порог, то сохраняем индексы точек
            r = dist[n1] / dist[n2]
            if r < threshold:
                match_result.append((i, n1))        
        
        matched_at_self = np.array([self.kp[i].pt for i, j in match_result])
        matched_at_img2 = np.array([img2kp[j].pt for i, j in match_result])
        return matched_at_self, matched_at_img2

def surf_demo(image1, image2, hessianThreshold=500, matchingThreshold=0.8):
    """
    Демонстрация работы алгоритма SURF
        image1 - имя файла изображения обекта
        image2 - имя файла изображения, содержащего объект
        hessianThreshold - переметр детектора SURF, порог для гессиана ключевых
            точек, используемого в детекторе
    """
    # 1.Загрузить два изображения ("box.png" и "box_in_scene.png"):
    img1 = cv2.imread(image1, 0)
    img2 = cv2.imread(image2, 0)

    # 2.Создать SURFDetector с параметрами hessianThreshold = 500
    surf = cv2.SURF(hessianThreshold)
    
    # 3.Рассчитать дескрипторы особых точек для обоих изображений:
    kp1, desc1 = surf.detect(img1, None, useProvidedKeypoints=False)
    kp2, desc2 = surf.detect(img2, None, useProvidedKeypoints=False)    
    desc1.shape = (-1, surf.descriptorSize())
    desc2.shape = (-1, surf.descriptorSize())
    #print u'%d особых точек на первом изображении, \
    #        %d на втором' % (len(kp1), len(kp2))
    result = {
        "Image1 keypoints count": len(kp1),
        "Image2 keypoints count": len(kp2)
    }
    # 4.Создать класс Features2DTracker для первого изображения:
    ft1 = Features2DTracker(kp1, desc1)

    # 5.При помощи метода Features2DTracker.MatchedImageFeature сделать 
    # сравнение описаний особых точек первого изображения с особыми точками 
    # второго. Каждой особой точке первого изображения ставится в соответствие 
    # одна особая точка второго изображения
    mp1, mp2 = ft1.MatchedImageFeature(desc2, kp2, matchingThreshold)

    # 6.Построить гомографию:
    H, status = cv2.findHomography(mp1, mp2, cv2.RANSAC, 5.0)
    #print u'сопоставлено %d из %d' % (np.sum(status), len(status))
    result.update({
        "Distanse matched points count": len(status),
        "Homography matched points count": np.sum(status)
    })
    # 7.Объединить первое и второе изображение в одно и соединить 
    # соответствующие точки.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result_img = np.zeros((h1 + h2, max(w1 + w1, w2)), np.uint8)
    result_img[:h1, :w1] = img1
    result_img[h1:h1+h2, :w2] = img2

    if H is not None:
        src_corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        dst_corners = np.int32(
            cv2.perspectiveTransform(src_corners.reshape(1, 4, 2), H)
            ).reshape(4, 2) + (0, h1)
        cv2.polylines(result_img, [dst_corners], True, WHITE)
        img3 = cv2.warpPerspective(img2, np.linalg.inv(H), (w1, h1))
        result_img[:h1, w1:w1+w1] = img3
        img3_corners = np.int32(src_corners) + (w1, 0)
        cv2.polylines(result_img, [img3_corners], True, WHITE)

        for (x1, y1), (x2, y2) in zip(dst_corners, img3_corners):
            cv2.line(result_img, (x1, y1), (x2, y2), WHITE)

    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

    if status is None:
        status = np.ones(len(mp1), np.bool_)

    for (x1, y1), (x2, y2), inlier in zip(np.int32(mp1), np.int32(mp2), status):
        color = GREEN if inlier else RED
        if inlier:
            cv2.line(result_img, (x1, y1), (x2, y2 + h1), color)
        else:
            r = 2 #cross radius
            t = 1 #thickness
            cv2.line(result_img, (x1 - r, y1 - r), (x1 + r, y1 + r), color, t)
            cv2.line(result_img, (x1 - r, y1 + r), (x1 + r, y1 - r), color, t)
            cv2.line(result_img, (x2 - r, y2 - r + h1), (x2 + r, y2 + r + h1),
                     color, t)
            cv2.line(result_img, (x2 - r, y2 + r + h1), (x2 + r, y2 - r + h1), 
                     color, t)
    # 8.Вывести первое изображение приведенное при помощи гомографии ко второму 
    # изображению 
    result.update({
        "Result image": result_img
    })    
    return result

if __name__ == '__main__':
    #result = surf_demo('./img/box.png', './img/box_in_scene.png')
    result = surf_demo('./img/object2.jpg', './img/object2_2.jpg')
    img = result.pop("Result image")
    print result
    cv2.imshow('SURF', img)
    cv2.waitKey()
    