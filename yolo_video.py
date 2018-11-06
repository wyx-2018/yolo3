import sys
import argparse
import os
from yolo import YOLO, detect_video
from PIL import Image

def detect_img():
    #while True:
    # xml_path = 'F:/vehicle-detection/Annotations/xmls/'
    img_path = 'images/in/'
    yolo=YOLO(**vars())
    imge_files=os.listdir(img_path)
    for img in imge_files:
        path=os.path.join(img_path,img)
        # xmlname=img.replace('.jpg','.xml')
        try:
            image = Image.open(path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            # xmlpath=os.path.join(xml_path,xmlname)
            # if not os.path.exists(xmlpath):
            print(image.filename.split('/')[-1])
            r_image = yolo.detect_image(image)
            #r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    print("Image detection mode")
    detect_img()