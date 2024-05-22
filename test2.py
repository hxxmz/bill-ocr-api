
# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddleocr
from paddleocr import PaddleOCR
import easyocr
from datetime import datetime


# ## ROI taken as input from user

#image_path
img_path="C:/Users/xc/Downloads/IMG_1.jpg"

#read image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# dictionary to store cropped coordinates
itemList = {
  "image_path": img_path,
  "meter_reading": None,
  "meter_id": None
}

# dictionary to store extracted meter reading & id via OCR
Tokens = {
    "meter_reading": [],
    "meter_id": []
}

#saving roi coordinates
roi_coordinates = []

#select ROI function for two different objects in a loop

for item in itemList.keys():
  if item != "image_path":
    print("Select the coordinates for", item)
    
    #select ROI function
    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    roi = cv2.selectROI(img_res)
    
    #print rectangle points of selected roi
    roi_coordinates.append(roi)
    print(roi)
    
    #Crop selected roi from raw image
    roi_cropped = img_res[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    scale_percent = 200
    width = int(roi_cropped.shape[1] * scale_percent / 100)
    height = int(roi_cropped.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_res = cv2.resize(roi_cropped, dim, interpolation = cv2.INTER_AREA)
    #show cropped image
    cv2.imshow(item, img_res)
    
    #save the cropped image
    itemList[item] = img_res
    
    # Get the current date and
    # time from system
    # and use strftime function
    # to format the date and time.
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
  
    # Split the picture path
    # into root and extension
    newfileName = item + ".jpg"
    splitted_path = os.path.splitext(newfileName)
  
    # Add the current date time
    # between root and extension
    modified_picture_path = splitted_path[0] + curr_datetime + splitted_path[1]
    
    cv2.imwrite(modified_picture_path, img_res)
  
    #hold window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if item == 'meter_reading':
        # Initialize PaddleOCR
        ocr = PaddleOCR()

        # Perform OCR on the image
        result = ocr.ocr(img_res)

        print(result)
        output_text = ' '.join(item[1][0] for item in result[0])

        # Print the output text
        text = output_text
        Tokens['meter_reading'].append(text)
    else:
#         img = cv2.imread(img_res)
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(thresh)
        Tokens['meter_id'].append(result[0][1])
        print(result[0][1])

print(itemList)
print(Tokens)


# ## ROI coordinates fixed for further readings

# In[ ]:


#image_path
img_path="C:/Users/sheri/Meter-Reading-master/input/IMG_6.jpg"

#read image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# dictionary to store cropped coordinates
itemList = {
  "image_path": img_path,
  "meter_reading": None,
}

# dictionary to store extracted content via OCR
Tokens = {
    "meter_reading": [],
}

#select ROI function for two different objects in a loop

def func(coordinates):
    for item in itemList.keys():
      if item != "image_path":
        print("Select the coordinates for", item)

        #select ROI function
        scale_percent = 30
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #     roi = cv2.selectROI(img_res)

        #rectangle points of roi
        roi = coordinates
    #     print(roi)

        #Crop selected roi from raw image
        roi_cropped = img_res[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        scale_percent = 200
        width = int(roi_cropped.shape[1] * scale_percent / 100)
        height = int(roi_cropped.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_res = cv2.resize(roi_cropped, dim, interpolation = cv2.INTER_AREA)
        #show cropped image
        cv2.imshow(item, img_res)

        #save the cropped image
        itemList[item] = img_res

        # Get the current date and
        # time from system
        # and use strftime function
        # to format the date and time.
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        # Split the picture path
        # into root and extension
        newfileName = item + ".jpg"
        splitted_path = os.path.splitext(newfileName)

        # Add the current date time
        # between root and extension
        modified_picture_path = splitted_path[0] + curr_datetime + splitted_path[1]

        cv2.imwrite(modified_picture_path, img_res)

        #hold window
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

        if item == 'meter_reading':
            # Initialize PaddleOCR
            ocr = PaddleOCR()

            # Perform OCR on the image
            result = ocr.ocr(img_res)

            print(result)
            output_text = ' '.join(item[1][0] for item in result[0])

            # Print the output text
            text = output_text
            Tokens['meter_reading'].append(text)

func(roi_coordinates[0])

print(itemList)
print(Tokens)

