import cv2
import numpy as np
import glob
from matplotlib import pyplot
import numpy.linalg as la
from SGBMTuner import *


def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # If the image is grayscale get the size differently
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        # Scale the image down by half each time
        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))

    return image



def rectify_stereo_pair_uncalibrated(imgL, imgR, threshold):
    width, height = imgL.shape[:2]

    F, mask, left_points, right_points = findFundementalMatrix(imgL, imgR, threshold)

    # linesL, linesR = calcualteEpilines(left_points, right_points, F)
    # img5, img6 = drawlines(imgL.copy(), imgR.copy(), linesL, left_points, right_points)
    # img3, img4 = drawlines(imgR.copy(), imgL.copy(), linesR, right_points, left_points)
    # pyplot.subplot(121), pyplot.imshow(img5)
    # pyplot.subplot(122), pyplot.imshow(img3)
    # pyplot.show()

    # Rectify the images
    ret, h_left, h_right = cv2.stereoRectifyUncalibrated(left_points, right_points, F,
                                                         (imgL.shape[1], imgL.shape[0]))

    # S = rectify_shearing(h_left, h_right, (imgL.shape[1], imgL.shape[0]))
    # h_left = S.dot(h_left)

    # Apply the rectification transforms to the images
    # camera_matrix = calibrator.camera_matrix
    # distortion = calibrator.distortion_coeff
    # imgsize = (imgL.shape[1], imgL.shape[0])
    # map1x, map1y, map2x, map2y = remap(camera_matrix, distortion, h_left, h_right, imgsize)
    #
    # rectified_left = cv2.remap(imgL, map1x, map1y,
    #                            interpolation=cv2.INTER_LINEAR)
    #
    # rectified_right = cv2.remap(imgR, map2x, map2y,
    #                             interpolation=cv2.INTER_LINEAR)

    rectified_left = cv2.warpPerspective(imgL, h_left, (height, width), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_right = cv2.warpPerspective(imgR, h_right, (height, width), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ## DRAW RECALCULATED EPILINES ##
    F, mask, left_points, right_points = findFundementalMatrix(rectified_left, rectified_right, 0.8)
    #
    # linesL, linesR = calcualteEpilines(left_points, right_points, F)
    # rectified_left, img6 = drawlines(rectified_left.copy(), rectified_right.copy(), linesL, left_points, right_points)
    # rectified_right, img4 = drawlines(rectified_right.copy(), rectified_left.copy(), linesR, right_points, left_points)
    # pyplot.subplot(121), pyplot.imshow(rectified_left)
    # pyplot.subplot(122), pyplot.imshow(rectified_right)
    # pyplot.show()

    ## Display rectified images ##
    cv2.imshow('Left RECTIFIED', rectified_left)
    cv2.imshow('Right RECTIFIED', rectified_right)
    pyplot.show()
    cv2.waitKey(0)

    return rectified_left, rectified_right


def findFundementalMatrix(imgL, imgR, threshold):

    sift = cv2.xfeatures2d.SIFT_create(2000)
    keyPointsL, descriptorL = sift.detectAndCompute(imgL, None)
    keyPointsR, descriptorR = sift.detectAndCompute(imgR, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptorL, descriptorR, k=2)

    good = []
    ptsL = []
    ptsR = []

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)
            ptsR.append(keyPointsR[m.trainIdx].pt)
            ptsL.append(keyPointsL[m.queryIdx].pt)

    ptsL = np.float32(ptsL)
    ptsR = np.float32(ptsR)


    # matchedImg = None
    # matchedImg2 = cv2.drawMatches(imgL, keyPointsL, imgR, keyPointsR, good, flags=2, outImg=matchedImg)
    # pyplot.imshow(matchedImg2), pyplot.show()
    # cv2.waitKey(0)

    # Calculate fundemental matrix from the feature point pair
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.RANSAC, 3, 0.99)

    # Select only inlier points
    ptsL = ptsL[mask.ravel() == 1]
    ptsR = ptsR[mask.ravel() == 1]

    return F, mask, ptsL, ptsR


def calcualteEpilines(left_points, right_points, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    linesL = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F)
    linesL = linesL.reshape(-1, 3)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    linesR = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F)
    linesR = linesR.reshape(-1, 3)

    return linesL, linesR


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2




def remap(camera_matrix, distortion, h_left, h_right, imgsize):
    r_left = la.inv(camera_matrix).dot(h_left).dot(camera_matrix)
    r_right = la.inv(camera_matrix).dot(h_right).dot(camera_matrix)
    height, width = imgsize

    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height), 1, (width, height))

    map1x, map1y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        r_left,
        optimal_camera_matrix,
        imgsize,
        cv2.CV_16SC2
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        r_right,
        optimal_camera_matrix,
        imgsize,
        cv2.CV_16SC2
    )
    return map1x, map1y, map2x, map2y


if __name__ == "__main__":
    imgL = cv2.imread("./input_images/left_rect_test.jpg")
    imgR = cv2.imread("./input_images/right_rect_test.jpg")
    # imgL = cv2.imread("./input_images/aloeL.jpg")
    # imgR = cv2.imread("./input_images/aloeR.jpg")


    l, r = rectify_stereo_pair_uncalibrated(imgL, imgR, 0.80)
    stereo = SGBMTuner(l, r)

    # F, mask, left_points, right_points = findFundementalMatrix(imgL, imgR, 0.68)
    #
    # # Select only inlier points
    # left_points = left_points[mask.ravel() == 1]
    # right_points = right_points[mask.ravel() == 1]
    #
    # linesL, linesR = calcualteEpilines(left_points, right_points, F)
    # img5, img6 = drawlines(imgL.copy(), imgR.copy(), linesL, left_points, right_points)
    # img3, img4 = drawlines(imgR.copy(), imgL.copy(), linesR, right_points, left_points)
    # pyplot.subplot(121), pyplot.imshow(img5)
    # pyplot.subplot(122), pyplot.imshow(img3)
    # pyplot.show()
