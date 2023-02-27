import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load color image
img_color = cv2.imread('C:\\python_projects\\CollegeProject_SignLanguageRecognition\\data\\100.jpg')

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Resize to (128, 128) while preserving aspect ratio
h, w = img_gray.shape[:2]
if h > w:
    new_h, new_w = 128, int(w / h * 128)
else:
    new_h, new_w = int(h / w * 128), 128
img_gray_resized = cv2.resize(img_gray, (new_w, new_h))

# Pad the image to size (128, 128)
pad_w = (128 - new_w) // 2
pad_h = (128 - new_h) // 2
img_gray_padded = cv2.copyMakeBorder(img_gray_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

# Add a new axis to create a single-channel image with shape (height, width, 1)
img_gray_reshaped = np.expand_dims(img_gray_padded, axis=-1)

dimension = img_gray_reshaped.shape
print(dimension)

#plt.imshow(img_gray_reshaped)
#plt.show()
