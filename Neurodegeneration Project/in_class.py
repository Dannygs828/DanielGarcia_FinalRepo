from termcolor import colored
import cv2
import numpy as np

# Image filename and depth (closest to interpolated point)
filename = r"data/MASK_Sk658 Llobe ch010033.jpg"
depth = 3350  # micrometers

# Load image in grayscale
img = cv2.imread(filename, 0)

# Threshold the image to binary (black & white)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Count white and black pixels
white = np.sum(binary == 255)
black = np.sum(binary == 0)

# Compute percentage of white pixels
white_percent = 100 * (white / (white + black))

# Print results
print(colored("Analysis of interpolated-depth image", "yellow"))
print(colored(f"Filename: {filename}", "red"))
print(f"Depth: {depth} µm")
print(f"White pixels: {white}")
print(f"Black pixels: {black}")
print(colored(f"Percentage of white pixels: {white_percent:.4f}%", "green"))
