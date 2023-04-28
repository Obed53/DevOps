import numpy as np
import cv2  #required for camera access
from copy import deepcopy
import pytesseract.pytesseract
from PIL import Image
import pytesseract as tess
pytesseract.pytesseract.tesseract_cmd= r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

def preprocess(img):
    cv2.imshow("Input",img)
    #imshow is a method which we use to display an image in a window
    imgBlurred = cv2.GaussianBlur(img,(5,5),0)
    #it returns us an blurred image we have to pass height and width
    gray = cv2.cvtColor(imgBlurred,cv2.COLOR_BGR2GRAY)
    #changes color

    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
    #it helps us to compute an approx of the gradient of the image
    #we can get the edges
    ret2, threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshold_img

def cleanPlate(plate):
    #this function clears/makes the number plate clear
    print("Processing...")
    gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    num_plate, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #helps us to detect the outer edges of the plate

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        #trying to access maximum area

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x,y,w,h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea,w,h):
            return plate,None
        cleaned_final = thresh[y:y+h, x:x+w]
        return cleaned_final,[x,y,w,h]
    else:
        return plate,None

def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17,3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img,op=cv2.MORPH_CLOSE,kernel=element,dst=morph_img_threshold)
    cv2.imshow("Morphed",morph_img_threshold)
    cv2.waitKey(0)


    contours, hierarchy = cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    return contours

def ratioCheck(area,width,height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 15 * aspect * 15  # minimum area
    max = 125 * aspect * 125  # maximum area
    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True


def isMaxWhite(plate):
    avg = np.mean(plate)
    if (avg >= 115):
        return True
    else:
        return False

def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect

    if (width > height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True


def cleanAndRead(img, contours):
    count=0
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)

        if validateRotationAndRatio(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]

            if (isMaxWhite(plate_img)):
                count+=1
                clean_plate, rect = cleanPlate(plate_img)

                if rect:
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    cv2.imshow("Cleaned Plate", clean_plate)
                    cv2.waitKey(0)
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    print("Detected Text : ", text)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Detected Plate", img)
                    cv2.waitKey(0)

                print ("No. of final cont : " , count)


if __name__ == '__main__':
    print("DETECTING PLATE . . .")

    # img = cv2.imread("testData/Final.JPG")
    img = cv2.imread("1.jpeg")
    #img = cv2.VideoCapture('50.mp4')
    threshold_img = preprocess(img)
    print("1 Done")
    contours = extract_contours(threshold_img)
    print("2 Done")
    if len(contours)!=0:
        #print (f"Total length of characters in the number plate {len(contours)}")
        #print(contours)
        #length of the number plate
        cv2.drawContours(img, contours, -1, (0,255,0), 1)
        cv2.imshow("Contours",img)
        cv2.waitKey(0)

    cleanAndRead(img, contours)
