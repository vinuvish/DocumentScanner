# import the necessary packages
from perspectiveTransform import transform_points
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="inputh the image path")
args = vars(ap.parse_args())

# now we are going to Edge Detection
image = cv2.imread(args["image"])
# scan on the original image rather than the resized image.
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height=500)

# Convert to gray scal image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# using GaussianBlur to noic ereduction
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# using Canny edge detaction for edge detaction
edge = cv2.Canny(gray, 75, 200)

# Finding Contours
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour

cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the counter
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# apply the four point transform to obtain a top-down
# view of the original image
wraper = transform_points(original,screenCnt.reshape(4, 2) * ratio)
#convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
wraper = cv2.cvtColor(wraper,cv2.COLOR_BGR2GRAY)
T = threshold_local(wraper,11,offset=10,method="gaussian")
wraper = (wraper > T).astype("uint8") * 255


cv2.imshow("Original", imutils.resize(original, height = 650))
cv2.imshow("Scanned", imutils.resize(wraper, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
