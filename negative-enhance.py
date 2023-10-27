import cv2

# Load the image
img = cv2.imread('nice/IMG_3770.jpg')

# downscale the image
img = cv2.resize(img, (0, 0), fx=0.9, fy=0.9)


# Convert the image to its negative
negative_img = 255 - img

cv2.imshow('Original Image', img)
cv2.imshow('negative Image', negative_img)

cv2.waitKey(0)
# Save the negative image
# cv2.imwrite('negative_image.jpg', negative_img)
