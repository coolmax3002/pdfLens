import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# source = cv2.VideoCapture(s)


# # Show image preview 
# win_name = 'Image Preview'
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
# #take a "picture" when the escape key is pressed, the frame is the picture taken
# while cv2.waitKey(1) != 27: # Escape
#     has_frame, frame = source.read()
#     if not has_frame:
#         break
#     cv2.imshow(win_name, frame)




# #release all resources and close camera preview
# source.release()
# cv2.destroyWindow(win_name)





# #show the picture taken for 10 seconds then close 
# cv2.imshow("picture", frame)
# cv2.waitKey(10000)
# cv2.destroyWindow("picture")



testDoc = cv2.imread("MidtermJulia.jpg", cv2.IMREAD_COLOR)
#thres_doc = cv2.adaptiveThreshold(testDoc, 255 ,cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY_INV, 11, 7)
#blurred = cv2.blur(thres_doc, (5, 5))
mask = np.ones((3, 3), np.uint8)
mask5 = np.ones((5, 5), np.uint8)
blank_doc = cv2.morphologyEx(testDoc, cv2.MORPH_CLOSE, mask, iterations=5)
blank_doc = cv2.cvtColor(blank_doc, cv2.COLOR_BGR2GRAY)
blank_doc = cv2.GaussianBlur(blank_doc, (5,5), 0)
edge = cv2.Canny(blank_doc, 150, 200)



#find and draw edges 
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(testDoc, contours, -1, (0, 255, 0), 3)

#find 4 best corners and circle them on original image
corners = cv2.goodFeaturesToTrack(image=edge, maxCorners=4, qualityLevel=0.2, minDistance=30)
corners = np.int0(corners)
for cor in corners:
    cv2.circle(testDoc, tuple(cor.flatten()), 25, (0,255,0), 3)



cv2.imshow("page", testDoc)
cv2.waitKey(0)
cv2.destroyAllWindows()









