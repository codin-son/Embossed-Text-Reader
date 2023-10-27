import cv2
import numpy as np

# Load an image
image = cv2.imread('preprocessed_image.jpg')

# Check if the image was loaded successfully
if image is None:
    print('Image not found!')
else:
    # Invert the image by subtracting it from the maximum intensity value (255 for 8-bit)
    negative_image = 255 - image

    # Display the original and negative images
    cv2.imshow('Original Image', image)
    cv2.imshow('Negative Image', negative_image)

    # Wait for a key press and then close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
