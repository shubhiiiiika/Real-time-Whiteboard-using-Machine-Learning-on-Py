import cv2
import numpy as np
import time
import os
import HandTrackingM as htm

######################
brushThickness = 5
eraserThickness = 70

######################
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlaylist = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)
print(len(overlaylist))   # to see whether we've imported all images correctly or not shows numbers of images
header = overlaylist[0]   # calling over image as header which overlays on each other
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1200)  # frame size
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.05)  # def value is 0.5
while True:
    # 1. Import image
    success, img = cap.read()
    # for importing image we've to flip
    # flip horizontally because mirror image
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks: done thru hand tracking module
    img = detector.findHands(img)
    # getting all landmark positions
    lnlist = detector.findPosition(img, draw=False)

    #check
    if len(lnlist)!=0:

       # print(lnlist)

        # tip of index and middle finger points
        x1, y1 = lnlist[8][1:]
        x2, y2 = lnlist[12][1:]
    # 3. Checking which fingers are up index, middle

        fingers = detector.fingersUp()
        #print(fingers)
    # To draw only one finger i.e index finger should be up to ot both fingers
    # 4. if selection mode - two fingers are up: we've to select not draw
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)
            print('Selection Mode')
           # checking for the click
           # if we're at the top of the image
        if y1 < 125:
            if 250 < x1 < 450:
                header = overlaylist[0]
                drawColor= (255, 0, 255)
            elif 550 < x1 < 750:
                header = overlaylist[1]
                drawColor = (255, 255, 255)
            elif 800 < x1 < 950:
                header = overlaylist[2]
                drawColor = (0, 255, 0)
            elif 1050 < x1 < 1200:
                header = overlaylist[3]
                drawColor = (0, 0, 0)


    # 5. if drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing Mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            #as we're taking start point and end point
            #here previous point = new point there it will draw as point now not line

            cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
            # the points will keep updating
            # previous position to new position
    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    #CONVERTING INTO BINARY IMAGE and inversing it
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  # adding to og image
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)




# Setting the header image
# because our image is matrix we just need to define the location of this new image overlaying, so will slice it
    img[0:125, 0:1280] = header  # img[height,width]
 #   img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

