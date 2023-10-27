import cv2
import numpy as np
import imutils
import pytesseract

# read from the web camera
cap = cv2.VideoCapture(3)

while True:
    # Read a frameq
    _, image = cap.read()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps between object edges
    edged = cv2.Canny(img, 40, 90)
    dilate = cv2.dilate(edged, None, iterations=2)

    # create an empty mask
    mask = np.ones(img.shape[:2], dtype="uint8") * 255

    # find the contours
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        cv2.imshow('Original image', cv2.resize(image, (640, 480)))
        cv2.imshow('Dilated', cv2.resize(image, (640, 480)))
        cv2.imshow('New Image', cv2.resize(image, (640, 480)))
        cv2.imshow('Inverted Threshold', cv2.resize(image, (640, 480)))
        # break
    else:
        cnts = cnts[0] if len(cnts) == 2 else cnts[0]
        orig = img.copy()
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 200:
                cv2.drawContours(mask, [c], -1, 0, -1)

            x, y, w, h = cv2.boundingRect(c)

            # Filter and remove more contours according to your need
            if w > h:
                cv2.drawContours(mask, [c], -1, 0, -1)

        # Remove ignored contours
        newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
        # Dilate again if necessary
        img2 = cv2.dilate(newimage, None, iterations=1)
        # Invert threshold for better results
        ret2, th1 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Tesseract on the filtered image
        temp = pytesseract.image_to_string(dilate)
        # Write text on the image
        cv2.putText(image, temp, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Show the results
        cv2.imshow('Original image', cv2.resize(image, (640, 480)))
        cv2.imshow('Dilated', cv2.resize(dilate, (640, 480)))
        cv2.imshow('New Image', cv2.resize(newimage, (640, 480)))
        cv2.imshow('Inverted Threshold', cv2.resize(th1, (640, 480)))
        print(temp)
    # Exit condition
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
