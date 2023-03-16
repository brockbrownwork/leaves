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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilated = cv2.dilate(edges, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours in descending order based on their area and only keep the ones larger than 100 pixels
leaf_contours = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) > 1000]

# Create a blank image to draw the filled contours onto
filled = np.zeros_like(img)

# Create a blank image for masking later
mask = np.zeros_like(img_gray)

# Iterate over the contours and fill them with green
for contour in leaf_contours:
    # plot the convex hull of the contour
    hull = cv2.convexHull(contour)
    cv2.fillPoly(filled, pts=[hull], color=(0, 0, 255))
    cv2.fillPoly(filled, pts=[contour], color=(0, 255, 0))
    cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
cv2.imshow('Filled Contours', filled)
cv2.waitKey(0)
# Display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)

# apply the mask to the original image
masked = cv2.bitwise_and(img, img, mask=mask)
# save the masked image
cv2.imwrite('masked.png', masked)

# save the image to output.png
cv2.imwrite('output.png', filled)

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

    # calculate the amount of fraying
    fraying = (red_pixels / (red_pixels + green_pixels)) * 100
    print(f"Fraying: {round(fraying, 3)}%")

# figure out the amount of holes

# Call the function with the image path as the argument
count_pixels("output.png")

# Load the image as a NumPy array
img = np.array(Image.open('masked.png'))

# Define the white and black thresholds
white_threshold = 210  # pixels with values above this threshold are considered "white"
black_threshold = 10   # pixels with values below this threshold are considered "black"

# Count the number of pixels close to white and non-black pixels
num_white_pixels = np.sum(img > white_threshold)
num_non_black_pixels = np.sum(img > black_threshold)

# Calculate the percentage of white pixels
percentage_white = num_white_pixels / num_non_black_pixels
print(f"Percentage of non-fray holes: {round(percentage_white * 100, 3)}%")

# Load the masked image
masked_img = cv2.imread('masked.png')

# Define the color range for white pixels
lower_white = np.array([white_threshold, white_threshold, white_threshold])
upper_white = np.array([255, 255, 255])

# Create a mask for the white pixels
white_mask = cv2.inRange(masked_img, lower_white, upper_white)

# Replace the white pixels with red pixels
masked_img[white_mask == 255] = (0, 0, 255)

# Save the image with the white pixels converted to red
cv2.imwrite('masked_red.png', masked_img)

# Display the image with the white pixels converted to red
cv2.imshow('Masked Red', masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
