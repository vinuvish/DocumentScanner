
# import the necessary packages
import numpy as np
import cv2

def align_points(points):
    # initialzie the coordinates
    # The coordinats are top-left,top-right,bottom-right and bottom-left
    # allocating memory for the four ordered points
    rect = np.zeros((4, 2), dtype = "float32")
    s = points.sum(axis=1)


    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    # smallest x + y sum, and the bottom-right point, which will have the largest x + y sum
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # compute the difference between the points, the
    # top-right and bottom-left points. Here we’ll take the difference (i.e. x – y) between the points using the np.diff
    diff = np.diff(points,axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect




def transform_points(image,points):
    # obtain a consistent order of the points and unpack them
    rect = align_points(points)
    (tl,tr,br,bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped