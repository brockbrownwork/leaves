import cv2
import numpy as np
from random import randint
import numpy as np
from PIL import Image
 
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

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# grab every contour that has a parent contour, and put it in a list called hole_contours
hole_contours = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) < 1000]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilated = cv2.dilate(edges, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours in descending order based on their area and only keep the ones larger than 100 pixels
leaf_contours = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) > 1000]

# Create a blank image to draw the filled contours onto
filled = np.zeros_like(img)

# Iterate over the contours and fill them with green
for contour in leaf_contours:
    # plot the convex hull of the contour
    hull = cv2.convexHull(contour)
    cv2.fillPoly(filled, pts=[hull], color=(0, 0, 255))
    cv2.fillPoly(filled, pts=[contour], color=(0, 255, 0))
# Iterate over the hole contours and fill them with red
# print("hole_contours: ", hole_contours)
# for contour in hole_contours:
#     cv2.fillPoly(filled, pts=[contour], color=(0, 0, 255))
# Display the result
cv2.imshow('Filled Contours', filled)
cv2.waitKey(0)
# save the image to output.png
cv2.imwrite('output.png', filled)
 
cv2.destroyAllWindows()

def count_pixels(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Define the green and red colors in RGB
    green_color = np.array([0, 255, 0])
    red_color = np.array([255, 0, 0])

    # Count the number of green and red pixels
    green_pixels = np.sum(np.all(img_array == green_color, axis=-1))
    red_pixels = np.sum(np.all(img_array == red_color, axis=-1))

    # Get the total number of pixels in the image
    total_pixels = img_array.shape[0] * img_array.shape[1]

    # Print the results
    print(f"Green pixels: {green_pixels}")
    print(f"Red pixels: {red_pixels}")
    # calculate the amount of fraying
    fraying = (red_pixels / (red_pixels + green_pixels)) * 100
    print(f"Fraying: {fraying}%")

# Call the function with the image path as the argument
count_pixels("output.png")
