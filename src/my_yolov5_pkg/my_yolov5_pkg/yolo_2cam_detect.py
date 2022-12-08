import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn


from my_yolov5_pkg.config import cfg
#from my_yolov5_pkg.models.common import DetectMultiBackend
from my_yolov5_pkg.models.common import DetectMultiBackend

from my_yolov5_pkg.utils.datasets import IMG_FORMATS, VID_FORMATS
from my_yolov5_pkg.utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from my_yolov5_pkg.utils.plots import Annotator, colors
from my_yolov5_pkg.utils.torch_utils import select_device, time_sync
from my_yolov5_pkg.util import read_class_names


from my_yolov5_pkg.utils.datasets import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bbox_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Header
from cv_bridge import CvBridge

from multiprocessing import Process



class yolo5PreProcess():
    def __init__(self,  weights,
                        data,
                        imagez_height,
                        imagez_width,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        classes,
                        agnostic_nms,
                        line_thickness,
                        half,
                        dnn
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        self.s = str()

        self.load_model()
        
    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = True
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
    
    def image_callback_pp(self, image_raw, window_name):
        # TODO: with varaible classes 
        class_list = []
        confidence_list = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []

        # im is  NDArray[_SCT@ascontiguousarray
        # im = im.transpose(2, 0, 1)
        self.stride = 32  # stride
        self.img_size = 640
        img = letterbox(image_raw, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        save_dir = "runs/detect/exp7"
        path = ['0']
        ## custum class selection code 
        
        class_names = read_class_names(cfg.YOLO.CLASSES)
        classes = list(class_names.values())
        # all clases are selected by default
        # allowed_classes = class_names.values()
        allowed_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'] 
        cls_idx = []

        for allowed_class in allowed_classes:
            #print(i, allowed_classes[i])
            # print(f"Detect_class: {allowed_class}")
            a = classes.index(allowed_class)
            # print(a)
            cls_idx.append(a)

        # custom class selection code end
        
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, cls_idx, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):
            im0 = image_raw
            self.s += f'{i}: '

            # p = Path(str(p))  # to Path
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    save_conf = False
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(xyxy, label)
                    class_list.append(self.names[c])
                    confidence_list.append(conf)
                    # tensor to float
                    x_min_list.append(xyxy[0].item())
                    y_min_list.append(xyxy[1].item())
                    x_max_list.append(xyxy[2].item())
                    y_max_list.append(xyxy[3].item())

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow(f"{window_name}", im0)
                cv2.waitKey(1)  # 1 millisecond

            return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list
        
class Yolo5Ros(Node):
    def __init__(self):
        super().__init__('yolov5_2cam')

        self.bridge = CvBridge()
        self.pub_bbox = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 1)
        self.pub_image = self.create_publisher(Image, 'yolov5/image_raw', 1)

        self.sub_image_1 = self.create_subscription(Image, 'img_1', self.image_callback_1,1)
        self.sub_image_2 = self.create_subscription(Image, 'img_2', self.image_callback_2,1)
        # p1= Process(target= self.create_subscription , args=(Image, 'img_1', self.image_callback_1, 1))
        # p2= Process(target= self.create_subscription , args=(Image, 'img_2', self.image_callback_2, 1))
        # p1.start()
        # p2.start()
        # p1.join()
        # p2.join()        
        
        #  TODO : 1. get parameters from yaml file 
        # parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter('weights', str(ROOT) + '/yolov5s.pt')
        self.declare_parameter('data', str(ROOT) + '/data/coco128.yaml')
        self.declare_parameter('imagez_height', 640)
        self.declare_parameter('imagez_width', 640)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('view_img', True)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)

        self.weights = self.get_parameter('weights').value
        self.data = self.get_parameter('data').value
        self.imagez_height = self.get_parameter('imagez_height').value
        self.imagez_width = self.get_parameter('imagez_width').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value

        self.yolov5 = yolo5PreProcess(self.weights,
                                            self.data,
                                            self.imagez_height,
                                            self.imagez_width,
                                            self.conf_thres,
                                            self.iou_thres,
                                            self.max_det,
                                            self.device,
                                            self.view_img,
                                            self.classes,
                                            self.agnostic_nms,
                                            self.line_thickness,
                                            self.half,
                                            self.dnn)
       
    def yolovFive2bboxes_msgs(self, bboxes:list, scores:list, cls:list, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        print(bboxes)
        # print(bbox[0][0])
        i = 0
        for score in scores:
            one_box = BoundingBox()
            one_box.xmin = int(bboxes[0][i])
            one_box.ymin = int(bboxes[1][i])
            one_box.xmax = int(bboxes[2][i])
            one_box.ymax = int(bboxes[3][i])
            one_box.probability = float(score)
            one_box.class_id = cls[i]
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1
        
        return bboxes_msg

    def image_callback_1(self, image:Image):
        #changes to color format BGR
        # image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        print('running for first camera image')
        image_raw = self.bridge.imgmsg_to_cv2(image)
        
        # return (class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = self.yolov5.image_callback_pp(image_raw, "src_1")

        msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)
        self.pub_bbox.publish(msg)

        self.pub_image.publish(image)

        print("start ==================")
        # print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        self.get_logger().info(f"class list: {class_list} conf_list:{confidence_list}")
        self.get_logger().info(f"loc: {x_min_list, y_min_list, x_max_list, y_max_list}")
        print("end ====================")
        
    def image_callback_2(self, image:Image):
        #changes to color format BGR
        # image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        print('running for second camera image')
        image_raw = self.bridge.imgmsg_to_cv2(image)
        
        # return (class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = self.yolov5.image_callback_pp(image_raw, "src_2")

        msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)
        self.pub_bbox.publish(msg)

        self.pub_image.publish(image)

        print("start ==================")
        self.get_logger().info(f"class list: {class_list} conf_list:{confidence_list}")
        self.get_logger().info(f"loc: {x_min_list, y_min_list, x_max_list, y_max_list}")
        self.get_logger().info(f"class is: {msg.bounding_boxes[0].class_id}")
        # print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        print("end ====================")
        
        
def main(args=None):
    rclpy.init(args=args)
    yolov5_node = Yolo5Ros()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()