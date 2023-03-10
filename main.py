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
blank_doc = cv2.morphologyEx(testDoc, cv2.MORPH_CLOSE, mask, iterations=5)
blank_doc = cv2.cvtColor(blank_doc, cv2.COLOR_BGR2GRAY)
blank_doc = cv2.blur(blank_doc, (5,5))
edge = cv2.Canny(blank_doc, 0, 200)
edge = cv2.dilate(edge, mask, iterations=3)

# contours, hierarchy = cv2.findContours(blank_doc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# edges = cv2.Canny(blank_doc, 80, 150)
# contours_poly = [None]*len(contours)
# boundRect = [None]*len(contours)
# for i, c in enumerate(contours):
#     contours_poly[i] = cv2.approxPolyDP(c, 1, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])  
# drawing = np.zeros((blank_doc.shape[0], blank_doc.shape[1], 3), dtype=np.uint8) 
# for i in range(len(contours)):
#         color = (0,255,0)
#         #cv2.drawContours(drawing, contours_poly, i, color)
#         cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#           (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
plt.imshow(edge, cmap='gray')
plt.title("Masked doc")
plt.show()






