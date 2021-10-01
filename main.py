#Imports
import os
import cv2 as cv
import numpy as np
from pathlib import Path    
from matplotlib import pyplot as plt


"""The max rgb filter considers the pixels which have green values above 80 and turns other pixels to black
then checks if green(G) is the maximum and  if not turns that pixel is turned to (0,0,0) i.e black
"""

def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv.split(image)
	R[G<80]=0
	G[G<80]=0
	B[G<80]=0
	M = np.maximum(np.maximum(B, G), R)
	R[G < M] = 0
	G[G < M] = 0
	B[G < M] = 0

	# merge the channels back together and return the image
	return cv.merge([B, G, R])


"""The remove_blue_filter function calculates the difference between Green and Red(GR) , Green and Blue(GB)
If GB>GR it means the pixel is not related to plant and is a background pixel 
"""

def remove_blue_filter (image):
  # split the image into its BGR components
  (B,G,R)=cv.split(image)
  GR=G-R
  GB=G-B
  R[GR>GB]=0
  B[GR>GB]=0
  G[GR>GB]=0
  # merge the channels back together and return the image
  return cv.merge([B,G,R])

"""The increase_brightness function increases the brightness of the image so that plants in the dark area 
can be recognized properly"""
def increase_brightness(img, value=50):
    # split the image into its HSV components
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    #Adding value to every pixel and chekimg overflow
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    # merge the channels back together and return the image
    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

Input_path = "Exam pictures"
Output_path = "Results"

for file in os.listdir(Input_path):
  file_path = f"{Input_path}/{file}"
  
  #Input the image
  img = cv.imread(file_path)

  #increase brightness 
  img = increase_brightness(img)

  #Using Bilateral filter to remove noise while preserving the edges of the plant
  blur = cv.bilateralFilter(img,10,150,150)


  rgb_max_image = max_rgb_filter(blur)
  reduced_blue_image= remove_blue_filter(rgb_max_image)

  cv.imwrite('reduced_blue_image.png',reduced_blue_image)

  #Read the reduced_blue_image in grayscale
  img = cv.imread('reduced_blue_image.png',0)

  #Applying ostu thresholding with Inverse binary thresholding to get the final B/W image
  ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

  #Write the output file in .png format
  #Isolate the file name
  file=Path(file).stem  
  output_file_path = f"{Output_path}/{file}.png"
  cv.imwrite(output_file_path, th2)