import cv2
import os

input_path = r"C:\recipe_project\test_images\dosa.jpg"
output_path = r"C:\recipe_project\test_images\dosa_opencv_resized.jpg"

# Read the original image
image = cv2.imread(input_path)

if image is None:
    print("ERROR: Image not found!")
    exit()

print("Original image size:", image.shape)

# Resize for OpenCLIP
resized = cv2.resize(image, (224, 224))

# Save OpenCV output
cv2.imwrite(output_path, resized)

print("OpenCV preprocessing completed!")
print("Resized image size:", resized.shape)
print("Output saved at:", output_path)