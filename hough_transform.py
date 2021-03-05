# -*- coding: utf-8 -*-

"""
Created on Mon Feb  8 15:30:07 2021

@author: Linda.LUDOVISI
@since: Mon Feb  8 15:30:07 2021
@version: 1.0.1
History:
    1.0.0 - Hough transform 
    1.0.1 - Implementation of clusterization to detect the main lines
            added self._get_centroids(), self._get_main_lines()
 
"""



from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np


class HoughLineDetector():
    
    MINVAL_CANNY = 10
    MAXVAL_CANNY = 90

    RHO         = 1
    THETA       = np.pi/180
    THRESHOLD   = 200

    NUM_CLUSTERS = 7

    def __init__(self, rho_acc = RHO, theta_acc = THETA, thresh = THRESHOLD, 
                 min_canny = MINVAL_CANNY, max_canny = MAXVAL_CANNY, n_clusters = NUM_CLUSTERS):
        self.rho_acc    = rho_acc
        self.theta_acc  = theta_acc
        self.thresh     = thresh
        
        self.min_canny = min_canny
        self.max_canny = max_canny
        
        self.n_clusters = n_clusters

    
    def __call__(self, image):
        # Process image for edge detection
        self._image = image
        self._edges = self._preprocess_image()
        
        # Get hough lines
        self._lines = self._get_hough_lines()

        # Get hough lines in clusters (get the centroids)
        self._centroids = self._get_centroids()

        # Get hough lines referring to conveyor belt
        return self._get_main_lines()

    
    def _get_hough_lines(self):    
        lines = cv.HoughLines(self._edges, self.rho_acc, self.theta_acc, self.thresh)
        # Draw Hough Lines on the image
        self._draw_hough_lines(lines)
        
        return lines

    
    def _draw_hough_lines(self, lines):
        #print(lines)
        hough_line_output = self._image.copy()
        # Convert (rho, theta) in a line and draw lines on the image
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            n = 2000
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))
            cv.line(hough_line_output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Show and save the image
        #self._show_image(hough_line_output)       
        #cv.imwrite('hough_line.jpg', hough_line_output)
    
    
    def _get_centroids(self):
        k = self.n_clusters       
        X = np.array([[line[0][0], line[0][1]] for line in self._lines])
        # Apply KMeans clustering to (rho, theta)
        kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = 100, 
                        n_init = 10, random_state = 0).fit(X)    
        clusters = np.expand_dims(kmeans.cluster_centers_, axis=1)  
        # Draw Hough Lines of the clusters
        self._draw_hough_lines(clusters)
        
        return clusters
    
    
    def _get_main_lines(self):
        # Here the idea is to order the centroids based on rho
        centroids = np.array([[line[0][0], line[0][1]] for line in self._centroids])
        sorted_centroids = centroids[np.argsort(centroids[ :, 0])]
        # Take the 3rd and the 6th
        sorted_centroids = np.delete(sorted_centroids,[0, 1, 3, 4, 6] , 0)
        # Draw Hough Lines
        sorted_centroids = np.expand_dims(sorted_centroids, axis=1)  
        self._draw_hough_lines(sorted_centroids)
        
        return sorted_centroids[0][0], sorted_centroids[1][0]
            

        
    def _preprocess_image(self):
        self._image = cv.imread(self._image, cv.IMREAD_COLOR) 
        # Create Edge Map using Canny Detector
        image_gray = cv.cvtColor(self._image.copy(), cv.COLOR_BGR2GRAY)
        edges = cv.Canny(image_gray, self.min_canny, self.max_canny)
        # Show image 
        #self._show_image(edges)
        
        return edges
    
    
    def _show_image(self, image):     
        cv.imshow("Hough transform", image)
        cv.waitKey()
    
    
    def _get_image(self):
        return self._image
    
    

    

