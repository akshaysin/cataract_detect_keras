import cv2 as cv
import os

dir_list = ['neg','info','pos']

# This loop will resize all imaes to 64*64*3
for img_dir in dir_list:
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            filefullpath=os.path.join(root,file)
            if (filefullpath.endswith('.jpg') or filefullpath.endswith('.JPG')):
                try:
                    print ("proccessing : {0}".format(filefullpath))
                    im=cv.imread(filefullpath)
                except :
                    print ("Error reading file : {0}".format(filefullpath))
