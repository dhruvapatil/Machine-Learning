import cv2
import cv2.cv as cv
import numpy as np
import os

'''
Input: Video
Output: Frames written in specified folder path
'''

def get_video_frames(path, folder_name = 'Frames'):
    #make destination folder if it doesnt exist
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    des_path = folder_name + '/'

    cap = cv2.VideoCapture(path)
    success, frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)  # handle of the Video Capture is required for obtaining frame.

    count = 1  #initialize count to write frame number
    while success:
        image_path = des_path + str(count) + '.jpg'
        cv2.imwrite(image_path, frame)  # save frame as JPEG file
        count += 1
        success, frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)  # to read the last frame

    cap.release()

