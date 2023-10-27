import cv2
import numpy as np
import pytesseract

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def main():
    # Load the image
    image_path = "Data/images/test10.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to open the image.")
        return
    # target_size = (1280, 1000)  # Replace with your desired size
    # image = cv2.resize(image, target_size)

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

#apply noise removel
                    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    # img = cv2.fastNlMeansDenoising(img, None, 5, 4, 14)
                    # denoisimg = cv2.fastNlMeansDenoising(img, None, 5, 4, 14)
                    # img = cv2.medianBlur(img, 5)
                    # denoisimg = cv2.medianBlur(img, 5)
                    
                    # blur it to remove noise
                    img = cv2.GaussianBlur(img, (7,7), 0)
                    
                    # perform edge detection, then perform a dilation + erosion to
                    # close gaps in between object edges
                    edged = cv2.Canny(img, 40, 90)
                    dilate = cv2.dilate(edged, None, iterations=2)
                    # perform erosion if necessay, it completely depends on the image
                    # erode = cv2.erode(dilate, None, iterations=1)
                    
                    # create an empty masks
                    mask = np.ones(img.shape[:2], dtype="uint8") * 255
                    
                    # find contours
                    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
                    cnts,hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    
                    
                    orig = img.copy()
                    for c in cnts:
                        # if the contour is not sufficiently large, ignore it
                        if cv2.contourArea(c) < 300:
                            cv2.drawContours(mask, [c], -1, 0, -1)
                        
                        x,y,w,h = cv2.boundingRect(c)
                        
                        # filter more contours if nessesary
                        if(w>h):
                            cv2.drawContours(mask, [c], -1, 0, -1)
                        
                    newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
                    img2 = cv2.dilate(newimage, None, iterations=1)
                    ret2,th1 = cv2.threshold(img2 ,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    
                    # Tesseract OCR on the image
                    temp = pytesseract.image_to_string(dilate)
                    
                    # Write results on the image
                    cv2.putText(image, temp, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 3)
                    
                    # show the outputs
                    cv2.imshow('Original image', cv2.resize(image,(640,480)))
                    cv2.imshow('Dilated', cv2.resize(dilate,(640,480)))
                    cv2.imshow('New Image', cv2.resize(newimage,(640,480)))
                    cv2.imshow('Inverted Threshold', cv2.resize(th1,(640,480)))
                    # cv2.imshow('Denois img', cv2.resize(denoisimg,(640,480)))
                    print(temp)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        elif key == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
