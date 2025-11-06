"""This program can be used to create percentage blur. 
The user inputs the percentage they would like to blur the images by
Used to create Modifiers B-D"""
import os
from PIL import Image, ImageFilter

INPUT_FOLDER = 'New Images'
OUTPUT_FOLDER = 'output_images_B_2'
BLUR_PERCENTAGE = 30

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

def blur_percent(tempimg, percent):
    """Helper function to blur stuff"""
    max_radius = min(img.size) / 10
    radius = (percent / 100) * max_radius
    return tempimg.filter(ImageFilter.GaussianBlur(radius))

for filename in os.listdir(INPUT_FOLDER):
    full_path = os.path.join(INPUT_FOLDER, filename)
    print(full_path[:full_path.index(".")])
    img = Image.open(full_path)
    blurred_image = blur_percent(img, BLUR_PERCENTAGE)
    blurred_image.save(f"{OUTPUT_FOLDER}/{filename[:filename.index(".")]}-{BLUR_PERCENTAGE}PercentBlur.png")
