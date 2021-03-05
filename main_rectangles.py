# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:39:14 2021

@author: Linda.LUDOVISI
@since: Tue Feb 16 12:39:14 2021
@version: 1.0.0
History: 
    1.0.0 - Main program for hough transform and shape detector
            For transparent products use:   L-H     0-180
                                            L-S     241-255
                                            L-V     0-255
                                            U-H     180-180
                                            U-S     255-255
                                            U-V     255-255
            
            For black products use:         L-H     0
                                            L-S     250
                                            L-V     56
                                            U-H     180
                                            U-S     255
                                            U-V     255
"""


from hough_transform import HoughLineDetector

import numpy as np
import cv2 as cv
import os

INDEX_GLOBAL = 0

def nothing(x):
    pass

def detect_color_scheme(img):
    # Create trackbars to find the mask 
    cv.namedWindow("Trackbars")
    cv.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
    cv.createTrackbar("L-S", "Trackbars", 250, 255, nothing)
    cv.createTrackbar("L-V", "Trackbars", 56 , 255, nothing)
    cv.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
    cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
    cv.createTrackbar("U-V", "Trackbars", 255, 255, nothing)
    
    loop = True 
    
    while loop:
        frame = img.copy()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
        l_h = cv.getTrackbarPos("L-H", "Trackbars")
        l_s = cv.getTrackbarPos("L-S", "Trackbars")
        l_v = cv.getTrackbarPos("L-V", "Trackbars")
        u_h = cv.getTrackbarPos("U-H", "Trackbars")
        u_s = cv.getTrackbarPos("U-S", "Trackbars")
        u_v = cv.getTrackbarPos("U-V", "Trackbars")
    
        lower_blue = np.array([l_h, l_s, l_v]) 
        upper_blue = np.array([u_h, u_s, u_v])
        
        # Create mask based on color detected and erase 5x5 pixel areas (avoid noise)
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel)
        
        loop = False
        
        '''
        # Show the result of the mask
        cv.imshow("Mask", mask)
    
        key = cv.waitKey(1) # Press 'ESC' to exit
        if key == 27:
            break
        '''
        
    cv.destroyAllWindows()
    
    return mask
  


def preprocess_image(img, lines, index, out_loc):
    
    height, width = img.shape[:2]
    
    # Compute m, q of the two detected lines (lines are sorted by rho)
    rho1, theta1, rho2, theta2 = lines[0][0], lines[0][1], lines[1][0], lines[1][1]
    m1, q1 = -np.cos(theta1)/np.sin(theta1), rho1/np.sin(theta1)
    m2, q2 = -np.cos(theta2)/np.sin(theta2), rho2/np.sin(theta2)
    
    # Create polygons that are going to be filled with black
    rect = np.array([[0,0], [width,0], [width, int(m1*width + q1)], [0, int(m1*0 + q1)]])
    tri =  np.array([[0, height], [0, int(m2*0 + q2)], [int((height-q2) / m2), height]])
    
    # Draw polygons
    for pts in [rect, tri]:
        pts = pts.reshape((-1,1,2))
        #cv2.polylines(img,[pts],True,(0,0,0)) 
        cv.fillPoly(img, [pts], (255,0,0))
    
    # Black 2/5 of image
    img[0:height, int(6*width/11):width] = [255,0,0]     
    #cv.imshow("Blue background", img)
    #cv.waitKey()
 
    '''
        SHAPE DETECTOR PHASE 
    '''
    # Retrieve mask
    mask = detect_color_scheme(img)
    
    # Detect contours, given the mask as input
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Prepare the frame to draw countours on
    frame = img.copy()
    
    coord_cnt = None # Use this to store coordinates of the rectangle
    check = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
        # Take the first coordinate to decide where to put the text
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        # Filter out areas which are small
        if area > 400:
            cv.drawContours(frame, [approx], 0, (255, 255, 0), 5)

            if len(approx) == 4:
                cv.putText(frame, "Rectangle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                if 125000 < area < 133000:   # Store rectangles which have a certain area 
                    coord_cnt = approx
                    check = 1

            #cv.imshow("Final Frame", frame)  
        #cv.waitKey()
    
    if check == 1: # If a rectangle has been detected

        print("shape of cnt: {}".format(coord_cnt.shape))
        rect = cv.minAreaRect(coord_cnt)
        print("rect: {}".format(rect))
    
        # the order of the box points: bottom left, top left, top right,
        # bottom right
        box = cv.boxPoints(rect)
        box = np.int0(box)
    
        print("bounding box: {}".format(box))
        #cv.drawContours(img, [box], 0, (0, 0, 255), 2)
    
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
    
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
    
        # The perspective transformation matrix
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
    
        # Directly warp the rotated rectangle to get the straightened rectangle
        warped = cv.warpPerspective(img, M, (width, height))
        
        # Write image in memory
        cv.imwrite(out_loc + "crop1_8_" + str(index) + ".jpg", warped)
        index += 1
        
        #cv.imshow('IMG',warped)
        #cv.waitKey(0)
        
    return index




if __name__ == "__main__":
    
    first_frame = 'frame0.jpg'
    # Apply Hough Transform on 'frame0.jpg' and obtain lines
    hough = HoughLineDetector()
    lines = hough(first_frame)
    
    in_loc = 'dataset/1/8_video/'
    out_loc = 'data_final/2/old8_video/'
    
    os.makedirs(out_loc)
    n_frames = len(os.listdir(in_loc))
    
    index = 0
    # For each image, preprocess it and detect rectangles
    for i in range(0, n_frames, 10):
        img_path = in_loc + 'frame' + str(i) + '.jpg'
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        index = preprocess_image(img, lines, index, out_loc)
    
    
    
   

        
        
     