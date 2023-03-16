import cv2
import numpy as np
from random import randint
 
# Read the original image
img = cv2.imread('image.png') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilated = cv2.dilate(edges, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# grab every contour that has a parent contour, and put it in a list called hole_contours
hole_contours = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) < 1000]

# Sort the contours in descending order based on their area and only keep the ones larger than 100 pixels
leaf_contours = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) > 1000]

# Create a blank image to draw the filled contours onto
filled = np.zeros_like(img)
hole_filled = np.zeros_like(img)

# Iterate over the contours and fill them with green
for contour in leaf_contours:
    cv2.fillPoly(filled, pts=[contour], color=(0, 255, 0))
# Iterate over the hole contours and fill them with red
print("hole_contours: ", hole_contours)
for contour in hole_contours:
    cv2.fillPoly(hole_filled, pts=[contour], color=(0, 0, 255))
# Display the result
cv2.imshow('Filled Contours', filled)
cv2.waitKey(0)
cv2.imshow('Hole Filled Contours', hole_filled)
cv2.waitKey(0)
# save the image to output.png
cv2.imwrite('output.png', filled)
 
cv2.destroyAllWindows()
