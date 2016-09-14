import cv2, cv2.cv as cv

'''
Input: Format for writing to video
Output: Codec for video
'''

def get_codec(format):
    if(format == 'avi' or '.avi'):
        codec = cv.CV_FOURCC('D','I','V','X')
    elif(format == 'ogg' or '.ogv' or 'ogg' or '.ogg'):
        codec = cv.CV_FOURCC('t', 'h', 'e', 'o')

    return codec
