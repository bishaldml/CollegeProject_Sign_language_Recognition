#importing the modules cv2 and numpy
import cv2
import numpy as np
#reading the image to be normalized using imread() function
imageread = cv2.imread('C:\\python_projects\\CollegeProject_SignLanguageRecognition\\data\\test\\A\\0.jpg')
#setting the array for resulting image after normalization
resultimage = np.zeros((800, 800))
#normalizing the given image using normalize() function
normalizedimage = cv2.normalize(imageread,resultimage, 0, 100, cv.NORM_MINMAX)
#displaying the normalized image as the output on the screen
cv2.imshow('Normalized_image', normalizedimage)
cv2.waitKey(0)
cv2.destroyAllWindows()