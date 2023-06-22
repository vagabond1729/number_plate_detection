from chardet import detect
from easyocr import Reader
import cv2
import numpy as np

# load the image and resize it
image = cv2.imread('image.jpg')


"""
image = cv2.resize(image, (800, 600))
cv2.imshow('small image', image)
kernel = np.ones((11,11), np.uint8)
# convert the input image to grayscale,
# blur it, and detect the edges 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('grey', gray)
blur = cv2.GaussianBlur(gray, (11,11), 0) 
cv2.imshow('blur', blur)
dilate = cv2.dilate(blur, kernel, iterations=1)
cv2.imshow('dilate', dilate)
edged = cv2.Canny(dilate, 10, 200) 
cv2.imshow('Canny', edged)
cv2.waitKey(1000)

# find the contours, sort them, and keep only the 5 largest ones
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in contours:
    # approximate each contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the contour has 4 points, we can say
    # that we have found our license plate
    if len(approx) == 4:
        n_plate_cnt = approx
        break        

# get the bounding box of the contour and 
# extract the license plate from the image
(x, y, w, h) = cv2.boundingRect(n_plate_cnt)
license_plate = gray[y:y + h, x:x + w]
"""
# initialize the reader object
reader = Reader(['en'])
# detect the text from the license plate

detection = reader.readtext(image)
# image = cv2.rectangle(image,(50,100),(80,125),(0,0,255),4)
# print(detection(4))
det = detection[0][1]
cv2.putText(image, det, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
cv2.imshow('original', image)
cv2.waitKey(5000000)
print("len: ", len(detection), " ", detection)



if len(detection) == 0:
    # if the text couldn't be read, show a custom message
    text = "Impossible to read the text from the license plate"
    cv2.putText(image, detection)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
else:
    # # draw the contour and write the detected text on the image
    # cv2.drawContours(image, [n_plate_cnt], -1, (0, 255, 0), 3)
    # text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
    # cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # # display the license plate and the output image
    # cv2.imshow('license plate', license_plate)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    pass
cv2.destroyAllWindows()