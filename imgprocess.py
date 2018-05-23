import SimpleITK as sitk
from myshow import myshow, myshow3d
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from utils import process_config,frame_factory,frame_diff,thresh_otsu,concatenate,resize_and_gray,show_img,open_op,PIL_filter,\
plt_show,shape_filter
from ipywidgets import interact, FloatSlider
params=process_config('..\\config.cfg')
frames=frame_factory(params)
fgbg = cv2.createBackgroundSubtractorMOG2()
for i in range(200):
    img,gray=resize_and_gray(frames[i],True)
    #gray=filter_img.filter(gray)
    fgmask = fgbg.apply(gray)
    #absdiff=cv2.absdiff(gray,fgbg.getBackgroundImage())
    #cv2.imshow('frame',concatenate(img,fgmask))
    #cv2.imshow('bs',np.concatenate([gray,absdiff],axis=1))
    #k = cv2.waitKey(100) 
    #if k == 27:
    #    break
    #else:
    #    continue
#cv2.destroyAllWindows()
img,gray=resize_and_gray(frames[200],True)
fgmask = fgbg.apply(gray)
plt_show(fgmask)
plt_show(cv2.absdiff(gray,fgbg.getBackgroundImage()))
absdiff=cv2.absdiff(gray,fgbg.getBackgroundImage())
ret, bg = cv2.threshold(fgmask,126,255,cv2.THRESH_BINARY)
ret, fg = cv2.threshold(fgmask,128,255,cv2.THRESH_BINARY)
plt_show(bg)
plt_show(fg)