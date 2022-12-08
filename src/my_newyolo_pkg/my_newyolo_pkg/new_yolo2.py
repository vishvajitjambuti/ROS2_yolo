import argparse
import os
import sys
from pathlib import Path
import platform
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from my_newyolo_pkg.models.common import DetectMultiBackend
from my_newyolo_pkg.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from my_newyolo_pkg.utils.plots import Annotator, colors, save_one_box
from my_newyolo_pkg.utils.torch_utils import select_device, smart_inference_mode
from my_newyolo_pkg.util import read_class_names
from my_newyolo_pkg.config import cfg


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bbox_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Header
from cv_bridge import CvBridge



class YoloPreProcecess():
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
                        dnn,
                        
                        save_txt = False, 
                        augment=False,
                        vid_stride = 5, # check video framrate  stride
                        ):
        self.weights = weights
        # source is added this testcase 
        
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
        self.augment=augment
        self.vid_stride = vid_stride
        self.save_txt = save_txt
        self.exist_ok = False
        self.hide_labels=False,  # hide labels
        self.hide_conf=False,  # hide confidences
        self.s = str()
        self.save_conf = False
        self.project=ROOT / 'runs/detect',  # save results to project/name
        self.nosave=True, 
        self.view_img=False,
        self.save_crop=False,
        self.name='exp',
        self.load_model()

    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        self.half &= (self.pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Directories
        # save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        # (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.save_dir = "/home/vishvajit/Workspace/yolov5_ultralytics-/runs/detect/"
        
        
        
        bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        # model warmup
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *imgsz))  # warmup
        # dt is is used in callback fun if it does not work check orignal code and make changes accordingly
        self.dt, self.windows, self.seen = (Profile(), Profile(), Profile()), [], 0
        
    def image_callback_pp(self, image_raw):
        source = image_raw
        imgsz = (self.imagez_height, self.imagez_width)
        # Dataloader
        self.webcam = True
        if self.webcam:
            # assign to view_img varibale and check_imshow function
            self.view_img = check_imshow()
            self.dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
            bs = len(self.dataset)  # batch_size
            print(f"image shape:{self.dataset.imgs[0].shape}")
            print(f"datatype of the dataset: {type(self.dataset)}")
            print(f"batch size is {bs}")
            cudnn.benchmark = True
            
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        
        
        
        
        class_list = []
        confidence_list = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []
        
        save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
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

        for path, im, im0s, vid_cap, s in self.dataset:
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.model(im, augment=self.augment, visualize=False)
                
            # Apply NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=cls_idx, agnostic=self.agnostic_nms, max_det=self.max_det)
                
            # Process Predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                
                p = Path(p)  # to Path
                # save_path = str(self.save_dir / p.name)  # im.jpg
                # txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        #     with open(f'{txt_path}.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # if save_img or self.save_crop or self.view_img:  # Add bbox to image
                        #     c = int(cls)  # integer class
                        #     label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        #     annotator.box_label(xyxy, label, color=colors(c, True))
                        # if self.save_crop:
                        #     save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

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
                    if platform.system() == 'Linux' and p not in self.windows:
                        self.windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
            #     if save_img:
            #         if self.dataset.mode == 'image':
            #             cv2.imwrite(save_path, im0)
            #         else:  # 'video' or 'stream'
            #             if self.vid_path[i] != save_path:  # new video
            #                 self.vid_path[i] = save_path
            #                 if isinstance(self.vid_writer[i], cv2.VideoWriter):
            #                     self.vid_writer[i].release()  # release previous video writer
            #                 if vid_cap:  # video
            #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #                 else:  # stream
            #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #                 self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #             self.vid_writer[i].write(im0)

            # # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")
            
            
            return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list


class Yolo5Ros2(Node):
    def __init__(self):
        super().__init__('new_yolo_two')

        self.bridge = CvBridge()
        self.pub_bbox = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 10)
        self.pub_image = self.create_publisher(Image, 'yolov5/image_raw', 10)

        
        
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
        #source = '0'
        self.yolov5 = YoloPreProcecess(self.weights,
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
        
        # create subscriber
        self.sub_image = self.create_subscription(Image, 'image_raw', self.image_callback,10)
        
        
        
        
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

    def image_callback(self, image:Image):
        #changes to color format BGR
        # image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        
        image_raw = self.bridge.imgmsg_to_cv2(image)
        
        # return (class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = self.yolov5.image_callback_pp(str(image_raw))

        msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)
        self.pub_bbox.publish(msg)

        self.pub_image.publish(image)

        print("start ==================")
        print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        print("end ====================")
        
def main(args=None):
    rclpy.init(args=args)
    yolov5_node = Yolo5Ros2()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()