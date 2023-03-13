import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from datetime import datetime

#When webcam is open there is live edge and corner tracking
def liveTracking(frame):
    mask = np.ones((5, 5), np.uint8)
    blank_doc = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, mask, iterations=5)
    blank_doc = cv2.cvtColor(blank_doc, cv2.COLOR_BGR2GRAY)
    blank_doc = cv2.GaussianBlur(blank_doc, (5, 5), 0) #smooth the image, but don't lose edges
    edge = cv2.Canny(blank_doc, 150, 200)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        for con in contours:
            cv2.drawContours(frame, con, -1, (0, 255, 0), 3)
    except:
        print("no contours")
    
    return frame

def croppedCord(corners):
    (tl, tr, br, bl) = corners
    widthBottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthTop= np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightRight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightLeft = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxW = max(int(widthBottom), int(widthTop))
    maxH = max(int(heightRight), int(heightLeft))
    return [[0, 0], [maxW, 0], [maxW, maxH], [0, maxH]]


#process an image after it has been taken
def scan(frame):
    mask = np.ones((5, 5), np.uint8)
    blank_doc = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, mask, iterations=5)
    blank_doc = cv2.cvtColor(blank_doc, cv2.COLOR_BGR2GRAY)
    blank_doc = cv2.GaussianBlur(blank_doc, (5, 5), 0) #smooth the image, but don't lose edges
    edge = cv2.Canny(blank_doc, 150, 200)

    #find and draw edges 
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #handle for errors by making sure we save and use the largest contour
    maxArea = 0
    maxContour = None
    for con in contours:
        area = cv2.contourArea(con)
        if area > maxArea:
            maxContour = con
            maxArea = area


    #find 4 best corners and circle them on original image
    corners = cv2.goodFeaturesToTrack(image=edge, maxCorners=4, qualityLevel=0.2, minDistance=30)
    corners = np.int0(corners)
    

    rect = np.zeros((4, 2), dtype='float32')
    pts = corners.reshape(-1, 2)
    sumCord = corners.reshape(-1, 2).sum(axis=1)
    diffCord = np.diff(corners.reshape(-1, 2), axis=1)
    rect[0] = pts[np.argmin(sumCord)]
    rect[2] = pts[np.argmax(sumCord)]
    rect[1] = pts[np.argmin(diffCord)]
    rect[3] = pts[np.argmax(diffCord)]

    finalCrop = croppedCord(rect)

    M = cv2.getPerspectiveTransform(np.float32(rect), np.float32(finalCrop))
    cropped = cv2.warpPerspective(frame, M, (finalCrop[2][0], finalCrop[2][1]), flags=cv2.INTER_LINEAR)
    return cropped


def main():
    # s = 0
    # if len(sys.argv) > 1:
    #     s = sys.argv[1]
    # source = cv2.VideoCapture(s)

    # # Show image preview 
    # win_name = 'Image Preview'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # #take a "picture" when the escape key is pressed, the frame is the picture taken
    # while cv2.waitKey(1) != 27: # Escape
    #     has_frame, frame = source.read()
    #     if not has_frame:
    #         break
    #     edgeFrame = liveTracking(frame)
    #     cv2.imshow(win_name, edgeFrame)
    # #release all resources and close camera preview
    # source.release()
    # cv2.destroyWindow(win_name)

    frame = cv2.imread("jubes.png", cv2.IMREAD_COLOR)

    #resize the image if over a certain size
    dim_limit = 1080
    max_dim = max(frame.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)


    #process image taken and show
    processedFrame = scan(frame)

    now = datetime.now()
    fileName = now.strftime("Scanned%d_%m_%Y_%H_%M.png")
    print(fileName)
    cv2.imwrite(fileName, processedFrame)
    cv2.imshow("Processed Image", processedFrame)
    cv2.waitKey(0)
    cv2.destroyWindow("Processed Image")

if __name__ == "__main__":
    main()





