import cv2
import numpy as np
import imutils
import pytesseract
from pytesseract import Output

# read from the web camera
cap = cv2.VideoCapture(1)

while True:
    # Read a frameq
    _, image = cap.read()
    image = cv2.medianBlur(image, 3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((3, 3), np.uint8)
    erodee = cv2.erode(thresh, kernel, iterations=1)
    dilatee = cv2.dilate(erodee, kernel, iterations=1)
    bitwise_notee = cv2.bitwise_not(dilatee)
    # gray_text = pytesseract.image_to_data(gray, lang='eng',
    #                                       config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789 ', output_type=Output.DICT)

    # thresh_text = pytesseract.image_to_data(thresh, lang='eng',
    #                                         config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789 ', output_type=Output.DICT)

    # eroded_text = pytesseract.image_to_data(erodee, lang='eng',
    #                                         config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789 ', output_type=Output.DICT)

    # dilatee_text = pytesseract.image_to_data(dilatee, lang='eng',
    #                                          config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789 ', output_type=Output.DICT)

    bitwise_notee_text = pytesseract.image_to_data(bitwise_notee, lang='eng',
                                                                       config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789./- ', output_type=Output.DICT)
    image_bitwise = image.copy()
    n_boxes = len(bitwise_notee_text['level'])
    # Replace with your image's resolution in DPI (dots per inch)

    for i in range(n_boxes):
        text = bitwise_notee_text['text'][i]
        width_pixels = bitwise_notee_text['width'][i]
        height_pixels = bitwise_notee_text['height'][i]

        # Check if text contains more than two periods ('.')
        if len(text) > 3 and len(text) < 7:
            last_char = text[-1]  # Get the last character of the text

            if (
                (text.count('.') <= 2 or text.count(
                    '-') <= 2 or text.count('/') <= 2)
                and last_char not in ['.', '-']
            ):

                (x, y, w, h) = (bitwise_notee_text['left'][i], bitwise_notee_text['top'][i],
                                width_pixels, height_pixels)
                cv2.rectangle(image_bitwise, (x, y),
                              (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image_bitwise, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # image_gray = image.copy()
    # n_boxes = len(gray_text['level'])
    # for i in range(n_boxes):
    #     text = gray_text['text'][i]
    #     if len(text) > 3:
    #         (x, y, w, h) = (gray_text['left'][i], gray_text['top'][i],
    #                         gray_text['width'][i], gray_text['height'][i])
    #         cv2.rectangle(image_gray, (x, y),
    #                       (x + w, y + h), (0, 0, 255), 2)
    #         cv2.putText(image_gray, text, (x, y-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # image_thresh = image.copy()
    # n_boxes = len(thresh_text['level'])
    # for i in range(n_boxes):
    #     text = thresh_text['text'][i]
    #     if len(text) > 3:
    #         (x, y, w, h) = (thresh_text['left'][i], thresh_text['top'][i],
    #                         thresh_text['width'][i], thresh_text['height'][i])
    #         cv2.rectangle(image_thresh, (x, y),
    #                       (x + w, y + h), (0, 0, 255), 2)
    #         cv2.putText(image_thresh, text, (x, y-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # image_delate = image.copy()
    # n_boxes = len(dilatee_text['level'])
    # for i in range(n_boxes):
    #     text = dilatee_text['text'][i]
    #     if len(text) > 3:
    #         (x, y, w, h) = (dilatee_text['left'][i], dilatee_text['top'][i],
    #                         dilatee_text['width'][i], dilatee_text['height'][i])
    #         cv2.rectangle(image_delate, (x, y),
    #                       (x + w, y + h), (0, 0, 255), 2)
    #         cv2.putText(image_delate, text, (x, y-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # image_erodee = image.copy()
    # n_boxes = len(eroded_text['level'])
    # for i in range(n_boxes):
    #     text = eroded_text['text'][i]
    #     if len(text) > 3:
    #         (x, y, w, h) = (eroded_text['left'][i], eroded_text['top'][i],
    #                         eroded_text['width'][i], eroded_text['height'][i])
    #         cv2.rectangle(image_erodee, (x, y),
    #                       (x + w, y + h), (0, 0, 255), 2)
    #         cv2.putText(image_erodee, text, (x, y-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.imshow("image Image", image)
    # cv2.imshow("gray Image", image_gray)
    cv2.imshow("thresh Image", thresh)
    cv2.imshow("erodee Image", erodee)
    cv2.imshow("dilatee Image", dilatee)
    cv2.imshow("bitwise_notee ", bitwise_notee)
    cv2.imshow("bitwise_notee Image", image_bitwise)
    # Exit condition
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
