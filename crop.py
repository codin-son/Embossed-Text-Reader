import cv2
import numpy as np
import pytesseract

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]


def bright_img(img):
    cropped_image = cv2.fastNlMeansDenoising(img, None, 5, 4, 14)

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    edges = cv2.Canny(cropped_image, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    ret2,th1 = cv2.threshold(eroded ,0, 500, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("thresh Image", thresh)
    cv2.imshow("edges Image", edges)
    cv2.imshow("dilated Image", dilated)
    cv2.imshow("eroded Image", eroded)
    cv2.imshow("inverted Image", th1)
    thresh_text = pytesseract.image_to_string(thresh, config=6)
    print("tresh text:",thresh_text)
    dilated_text = pytesseract.image_to_string(dilated, config=6)
    print("dilated text:",dilated_text)
    eroded_text = pytesseract.image_to_string(eroded, config=6)
    print("eroded text:",eroded_text)
    crop_text = pytesseract.image_to_string(cropped_image, config=6)
    print("cropped_image text:",crop_text)
    th1_text = pytesseract.image_to_string(th1, config=6)
    print("invert text:",th1_text)
    
    if thresh_text == "" and dilated_text == "" and eroded_text == "" and crop_text == "" and th1_text == "":
        return True

def main():
    # Load the image
    image_path = "nice/IMG_1431.JPG"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to open the image.")
        return
    target_size = (1280, 1000)  # Replace with your desired size
    image = cv2.resize(image, target_size)
    

    while True:
        # Display the original image
        cv2.imshow("Original Image", image)
        key = cv2.waitKey(0)

        if key == ord('c'):
            # Prompt the user to select a region for cropping
            x, y, width, height = cv2.selectROI("Select ROI", image)
            cv2.destroyAllWindows()

            if width > 0 and height > 0:
                cropped_image = crop_image(image, x, y, width, height)

                # Display the cropped image
                cv2.imshow("Cropped Image", cropped_image)
                key = cv2.waitKey(0)

                if key == ord('p'):
                    # got_text = bright_img(cropped_image)
                    
                    # if(got_text):
                    #     print("no text found")
                    #     continue
                    # cropped_image = 255 - cropped_image 
                    # cropped_image = cv2.fastNlMeansDenoising(cropped_image, None, 5, 4, 14)
                    # cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)
                    cropped_image = cv2.medianBlur(cropped_image, 3)

                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
                    kernel = np.ones((3, 3), np.uint8)
                    erodee = cv2.erode(thresh, kernel, iterations=1)
                    dilatee = cv2.dilate(erodee, kernel, iterations=1)
                    bitwise_notee = cv2.bitwise_not(dilatee)
                    
                    cv2.imshow("gray Image", gray)
                    cv2.imshow("thresh Image", thresh)
                    cv2.imshow("erodee Image", erodee)
                    cv2.imshow("dilatee Image", dilatee)
                    cv2.imshow("bitwise_notee Image", bitwise_notee)
                    
                    gray_text = pytesseract.image_to_string(gray)
                    print("tresh text:",gray_text)
                    thresh_text = pytesseract.image_to_string(thresh)
                    print("thresh text:",thresh_text)
                    eroded_text = pytesseract.image_to_string(erodee)
                    print("eroded text:",eroded_text)
                    dilatee_text = pytesseract.image_to_string(dilatee)
                    print("dilatee text:",dilatee_text)
                    bitwise_notee_text = pytesseract.image_to_string(bitwise_notee)
                    print("bitwise_notee text:",bitwise_notee_text)
                    


                    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                    # edges = cv2.Canny(cropped_image, 100, 200)
                    # kernel = np.ones((3, 3), np.uint8)
                    # dilated = cv2.dilate(thresh, kernel, iterations=1)
                    # eroded = cv2.erode(dilated, kernel, iterations=1)
                    # ret2,th1 = cv2.threshold(eroded ,0, 500, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    # cv2.imshow("thresh Image", thresh)
                    # cv2.imshow("edges Image", edges)
                    # cv2.imshow("dilated Image", dilated)
                    # cv2.imshow("eroded Image", eroded)
                    # cv2.imshow("inverted Image", th1)
                    # thresh_text = pytesseract.image_to_string(thresh, config="6")
                    # print("tresh text:",thresh_text)
                    # dilated_text = pytesseract.image_to_string(dilated, config="6")
                    # print("dilated text:",dilated_text)
                    # eroded_text = pytesseract.image_to_string(eroded, config="6")
                    # print("eroded text:",eroded_text)
                    # crop_text = pytesseract.image_to_string(cropped_image, config="6")
                    # print("cropped_image text:",crop_text)
                    # th1_text = pytesseract.image_to_string(th1, config="6")
                    # print("invert text:",th1_text)
                    key = cv2.waitKey(0)

                    if key == ord('s'):
                        # Save the preprocessed image to a file
                        cv2.imwrite("preprocessed_image.jpg", cropped_image)
                        print("Preprocessed image saved.")
                cv2.destroyAllWindows()
        elif key == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
