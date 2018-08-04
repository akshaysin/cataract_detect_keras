import cv2 as cv
import os
import Augmentor
import shutil
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def resize_img(im):
    resized_img = cv.resize(im, (128, 128))
    return resized_img
# https://github.com/mdbloice/Augmentor
def create_samples(dir):
    p = Augmentor.Pipeline(dir)
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    # p.ground_truth("/path/to/ground_truth_images")
    # Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.sample(1000)

if __name__ == '__main__':
    dir_list = ['neg','pos']
    dest="input_path"
    # lets create a sample of 500 negative and 500 positive Images
    for img_dir in dir_list:
        print ("************************Proccesseing {0} Directory*****************************".format(img_dir))
        create_samples(img_dir)
        # Now lets move files created to input_path folder
        source=img_dir+"/output/"
        files = os.listdir(source)
        for f in files:
            shutil.move(source+f, dest)
        print ("********************************************************************************")

    # # This loop will resize all imaes to 64*64*3
    # for img_dir in dir_list:
    #     # create_samples(img_dir)
    print ("************************Starting the resize proccess*****************************".format(img_dir))
    for root, dirs, files in os.walk(dest):
        for file in files:
            filefullpath=os.path.join(root,file)
            if (filefullpath.endswith('.jpg') or filefullpath.endswith('.JPG')):
                im_read=cv.imread(filefullpath)
                resized_img = resize_img(im_read)
                print ("[INFO] Writing converted {0} file".format(filefullpath))
                cv.imwrite(filefullpath,resized_img)
