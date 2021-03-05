# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:09:51 2021

@author: Linda.LUDOVISI
@since: Fri Feb  5 14:09:51 2021
@version: 1.0.0
History:
    1.0.0 - Extract frames from videos and save them creating new directories.
"""

import os
import cv2


def video_to_frames(in_loc, out_loc):    
    os.makedirs(out_loc)    
    vidcap = cv2.VideoCapture(in_loc)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(out_loc,"frame{:d}.jpg".format(count)), image)     
        count += 1
    print("{} images are extacted in {}.".format(count,out_loc))


if __name__=="__main__":

    #input_loc = 'videos/1/'    # Videos from the first folder
    #output_loc = 'dataset/1/'
    input_loc = 'videos/2/'    # Videos from the second folder
    output_loc = 'dataset/2/'
    
    for i in range(1,31):
        in_loc = input_loc + 'new_' + str(i) + '.h264'  # Videos from the second folder
        #in_loc = input_loc + str(i) + '.h264'          # Videos from the first folder
        out_loc = output_loc + str(i) + '_video/' 
        print(f"Converting video {in_loc} into directory {out_loc}")
        video_to_frames(in_loc, out_loc)
        
        
        