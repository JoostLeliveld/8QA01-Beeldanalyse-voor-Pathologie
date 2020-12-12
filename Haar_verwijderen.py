# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:58:01 2020

@author: 20203167
"""

import cv2
def haarverwijderen(src):
    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(src,thresh2, 1, cv2.INPAINT_TELEA)
    return dst
    

 
