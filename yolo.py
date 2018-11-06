# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
# import os
# from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
import re
from keras.utils import multi_gpu_model
from xml.dom.minidom import Document

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolov3.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        # classes_path = os.path.expanduser(self.classes_path)
        classes_path = 'model_data/coco_classes.txt'
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        # anchors_path = os.path.expanduser(self.anchors_path)
        anchors_path = 'model_data/yolo_anchors.txt'
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # model_path = os.path.expanduser(self.model_path)
        model_path = 'model_data/yolov3.h5'
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        #start = timer()
        #缩放图像为固定尺寸
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        coco={}
        file_name=image.filename.split('/')[-1]
        image_draw=np.zeros(image.size,np.uint8)
        image_draw=image.copy()
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image_draw)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            
            
            # nodename=self.get_class_name_from_filename(file_name)
            # xmlname = file_name.replace('.jpg','.xml')
            vehicle=['car','truck']
            if predicted_class in vehicle and score>0.65 and ((right-left)>=1/4*image.width and (bottom-top)>=1/4*image.height):
                image1=image.crop((left,top,right,bottom))
                newname=file_name.replace('.jpg','%s.jpg'%('-'+str(i)))
                image1.save('images/crop/%s' % newname)
                # My kingdom for a good redistributable image drawing library.
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
	            # if xmlname in coco:
	            # # object
	            #     Createnode = coco[xmlname]
	            #     object_node=Createnode.createElement('object')
	            #     Root.appendChild(object_node)
	                
	            #     node=Createnode.createElement('name')
	            #     node.appendChild(Createnode.createTextNode(nodename))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('pose')
	            #     node.appendChild(Createnode.createTextNode('Unspecified'))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('truncated')
	            #     node.appendChild(Createnode.createTextNode('0'))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('difficult')
	            #     node.appendChild(Createnode.createTextNode('0'))
	            #     object_node.appendChild(node)
	                
	            #     bndbox_node=Createnode.createElement('bndbox')
	            #     object_node.appendChild(bndbox_node)
	                
	            #     node=Createnode.createElement('xmin')
	            #     node.appendChild(Createnode.createTextNode(str(left)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('ymin')
	            #     node.appendChild(Createnode.createTextNode(str(top)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('xmax')
	            #     node.appendChild(Createnode.createTextNode(str(right)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('ymax')
	            #     node.appendChild(Createnode.createTextNode(str(bottom)))
	            #     bndbox_node.appendChild(node)
	            # else:
	            #     #Produce xml for each image
	            #     Createnode=Document()  #创建DOM文档对象
	                    
	            #     Root=Createnode.createElement('annotation') #创建根元素
	            #     Createnode.appendChild(Root)
	                
	            #     # folder
	            #     folder=Createnode.createElement('folder')
	            #     folder.appendChild(Createnode.createTextNode('images'))
	            #     Root.appendChild(folder)
	                
	            #     # filename
	            #     filename = Createnode.createElement('filename')
	            #     filename.appendChild(Createnode.createTextNode(file_name))
	            #     Root.appendChild(filename)

	            #     # path
	            #     path = Createnode.createElement('path')
	            #     path.appendChild(Createnode.createTextNode(image.filename))
	            #     Root.appendChild(path)
	                
	            #     # source
	            #     source_node = Createnode.createElement('source')
	            #     Root.appendChild(source_node)
	                
	            #     node = Createnode.createElement('database')
	            #     node.appendChild(Createnode.createTextNode('Unknown'))
	            #     source_node.appendChild(node)
	                
	            #     # size
	            #     size_node=Createnode.createElement('size')
	            #     Root.appendChild(size_node)
	                
	            #     node=Createnode.createElement('width')
	            #     node.appendChild(Createnode.createTextNode(str(image.size[0])))
	            #     size_node.appendChild(node)
	                
	            #     node=Createnode.createElement('height');
	            #     node.appendChild(Createnode.createTextNode(str(image.size[1])))
	            #     size_node.appendChild(node)
	                
	            #     node=Createnode.createElement('depth')
	            #     node.appendChild(Createnode.createTextNode('3'))
	            #     size_node.appendChild(node)
	                
	            #     # segmented
	            #     node=Createnode.createElement('segmented')
	            #     node.appendChild(Createnode.createTextNode('0'))
	            #     Root.appendChild(node)
	                
	            #     # object
	            #     object_node=Createnode.createElement('object')
	            #     Root.appendChild(object_node)
	                
	            #     node=Createnode.createElement('name')
	            #     node.appendChild(Createnode.createTextNode(nodename))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('pose')
	            #     node.appendChild(Createnode.createTextNode('Unspecified'))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('truncated')
	            #     node.appendChild(Createnode.createTextNode('0'))
	            #     object_node.appendChild(node)
	                
	            #     node=Createnode.createElement('difficult')
	            #     node.appendChild(Createnode.createTextNode('0'))
	            #     object_node.appendChild(node)
	                
	            #     bndbox_node=Createnode.createElement('bndbox')
	            #     object_node.appendChild(bndbox_node)
	                
	            #     node=Createnode.createElement('xmin')
	            #     node.appendChild(Createnode.createTextNode(str(left)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('ymin')
	            #     node.appendChild(Createnode.createTextNode(str(top)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('xmax')
	            #     node.appendChild(Createnode.createTextNode(str(right)))
	            #     bndbox_node.appendChild(node)
	                
	            #     node=Createnode.createElement('ymax')
	            #     node.appendChild(Createnode.createTextNode(str(bottom)))
	            #     bndbox_node.appendChild(node)
	            #     coco[xmlname] = Createnode
        image_out_path=os.path.join('images/out/',file_name)
        image_draw.save(image_out_path,quality=90)
        # xml_path='images/xmls/'
        # if coco:
        #     with open(xml_path+xmlname,'w') as f:
        #         f.write(coco[xmlname].toprettyxml(indent = '\t'))
        # end = timer()
        # print(end - start)
        return image
    def get_class_name_from_filename(self,filename):
        labels_path='images/labels.txt'
        with open(labels_path,encoding='utf-8') as f:
            label_names = f.readlines()
        label_names = [c.strip() for c in label_names]
        match = re.match(r'([a-zA-Z0-9\_]+[\_])([0-9]+)(\.jpg)', filename, re.I)
        nodenum=match.groups()[1]
        for label_name in label_names:
            if nodenum==label_name.split(':')[0]:
                return label_name.split(':')[0]

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

