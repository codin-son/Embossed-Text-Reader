import cv2
import numpy as np
import imutils
import pytesseract
import re

# read from the web camera
cap = cv2.VideoCapture(3)

while True:
    # Read a frameq
    _, image = cap.read()
    cropped_image = cv2.medianBlur(image, 3)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((3, 3), np.uint8)
    erodee = cv2.erode(thresh, kernel, iterations=1)
    dilatee = cv2.dilate(erodee, kernel, iterations=2)
    # dilatee = cv2.erode(dilatee, kernel, iterations=1)
    bitwise_notee = cv2.bitwise_not(dilatee)
    thresh_text = pytesseract.image_to_string(thresh, config="6")
    eroded_text = pytesseract.image_to_string(erodee, config="6")
    dilatee_text = pytesseract.image_to_string(dilatee, config="6")
    bitwise_notee_text = pytesseract.image_to_string(bitwise_notee, config="6")
    thresh_text = re.sub(r'[^\w\s.]', '', thresh_text)
    eroded_text = re.sub(r'[^\w\s.]', '', eroded_text)
    dilatee_text = re.sub(r'[^\w\s.]', '', dilatee_text)
    bitwise_notee_text = re.sub(r'[^\w\s.]', '', bitwise_notee_text)
    bitwise_notee_text = bitwise_notee_text.encode('utf-8').decode('utf-8')

    
    # Write text on the image
    cv2.putText(image, bitwise_notee_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    # Show the results
    cv2.imshow('Original image', cv2.resize(image, (640, 480)))
    cv2.imshow('cropped_image', cv2.resize(cropped_image, (640, 480)))
    cv2.imshow('New gray', cv2.resize(gray, (640, 480)))
    cv2.imshow('New thresh', cv2.resize(thresh, (640, 480)))
    cv2.imshow('New erodee', cv2.resize(erodee, (640, 480)))
    cv2.imshow('New dilatee', cv2.resize(dilatee, (640, 480)))
    cv2.imshow('New bitwise_notee', cv2.resize(bitwise_notee, (640, 480)))
    print("thresh text:",thresh_text)
    print("eroded text:",eroded_text)
    print("dilatee text:",dilatee_text)
    print("bitwise_notee text:",bitwise_notee_text)
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
