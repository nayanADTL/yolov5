# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
import json
from pathlib import Path
import time


import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print(ROOT)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


#--## setting path
sys.path.append('../tools')
from tools.autopilot import Autopilot
from tools.vector_math import Position4D
from tools.geodesy import compute_gps
from tools.networking import createUDPLink


class Classifier:  # Classifier class used for initializing aerial human detection model and all other necessary dependencies.

    fov_x = 73.387
    fov_y = 45.462

    def __init__(self, id, simulation):
        self.id = int(id)
        # self.size = size
        # self.fps = fps

        if simulation:
            sim_id = id
        else:
            sim_id = 1

        self.ap_addr = ("127.0.0.1", 14553 + (sim_id-1)*10)
        self.swarm_tx_addr = ("127.0.0.1", 10002 + (sim_id-1)*10) #(10002)

    @torch.no_grad()
    def run( self,
            event,
            source, #=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            weights, #=ROOT / 'yolov5s.pt',  # model.pt path(s)
            imgsz,  # =(640, 640),  # inference size (height, width)
            view_img, #=True,  # show results
            save_txt, #=False,  # save results to *.txt
            data, #=ROOT / 'data/data.yaml',  # dataset.yaml path
            showPred, #=True, #For printing the predictions
            sendPred, #=True, #For sending the tags
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            save_conf=True,  # save confidences in --save-txt labels
    ):

        #--## Code Setup
        swarm_tx_sock = createUDPLink(self.swarm_tx_addr, is_server = None)
        ap = Autopilot(address=self.ap_addr)
        center_pos = Position4D([0,0,0,0])

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url)
        # print(webcam)
        #if is_url:
        #    source = check_file(source)  # download
        

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # print(imgsz)


        # Dataloader
        if webcam:
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    
        while True:
            try:
                for path, im, im0s, vid_cap, s in dataset:

                    if not event.is_set():
                        print("......................waiting to be set again.............................................")
                        event.wait()
                    
                    pos1 = ap.pos_4d
                    pos2 = ap.pos_4d
                    avg_pos = (pos1 + pos2)/2


                    #@#t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    #@#t2 = time_sync()
                    #@#dt[0] += t2 - t1

                    # Inference
                    # print('Reading image3:',im.shape)
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    print("Performing prediction")
                    start = time.time()
                    pred = model(im, augment=augment, visualize=visualize)
                    #@#t3 = time_sync()
                    #@#dt[1] += t3 - t2

                    # NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    # print(pred)
                    #@#dt[2] += time_sync() - t3
                    print("fps = ", 1/(time.time()-start))
                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                    
                    print(seen)
                    tag_list=[]
                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        # print('Reading image4:',im0.shape)
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        # print('Reading image5:',im.shape)

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            # This function is saving the predictions one by one
                            for *xyxy, conf, cls in reversed(det):
                                # print('xyxy',xyxy)
                                # print(type(xyxy)) 
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                                if showPred:
                                    print('Printing the predictions:')
                                    print(line)
                                    # print(cls.item())
                                    # print(conf.item())
                                    
                                if save_txt:  # Write to file 
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        # File stores the class, x-center, y-center, width, height, confidence 
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                


                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                                # Preparing the tag_list to send the predictions
                                x1 = xywh[0] - xywh[2] / 2  # top left x
                                y1 = xywh[1] - xywh[3] / 2  # top left y
                                x2 = xywh[0] + xywh[2] / 2  # bottom right x
                                y2 = xywh[1] + xywh[3] / 2  # bottom right y
                                
                                lat, lon = compute_gps(xywh[0], xywh[1], Classifier.fov_x, Classifier.fov_y, avg_pos, im0.shape)
                                # print('lat and lon',lat, lon)

                                
                                # Generating the json using the x-center and y-center with width and height of the bounding box
                                # tag_json = {'lbl':int(cls.item()), 'Acc.': conf.item(), 'x-center': xywh[0], 'y-center': xywh[1], 'w': xywh[2], 'h': xywh[3]}
                                # print(tag_json)
                                # Generating the json using the x1,y1 (top left) and x2,y2 (bottom right) of the bounding box
                                # To map the x1,y1 and x2,y2 values (between 0 and 1) to absolute coodinators using: x1=x1*im0.shape[1] or y1=y1*im0.shape[0]
                                tag_json = {'lbl':int(cls.item()), 'Acc.': int(conf.item()*100), 'loc':[lat, lon], 'x1,y1': [x1,y1], 'x2,y2': [x2,y2]}
                                # tag_json = {'lbl':int(cls.item()), 'Acc.': int(conf.item()*100), 'x1,y1': [x1,y1], 'x2,y2': [x2,y2]}
                                print(tag_json)
                                tag_list.append(tag_json)

                        # Stream results
                        im0 = annotator.result()
                        if view_img:
                            if platform.system() == 'Linux' and p not in windows:
                                windows.append(p)
                                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)
                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer[i].write(im0)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
                    if sendPred:
                        try:
                            # "tag":{[
                            #         "label":str,    // object detected: human/tank etc.
                            #         "accuracy":int, // confidence level
                            #         "pos":(float, float)    // (lat, lon)
                            #     ],
                            #     [],
                            #       }
                            if len(tag_list) !=0:
                                js= {}
                                js["tag"]=tag_list
                                pkt = json.dumps(js).encode("UTF-8")
                                # print('Printing the tags')
                                # print(pkt)
                                swarm_tx_sock.sendto(pkt, self.swarm_tx_addr) #10002
                        except Exception as err:
                            print("[Swarm][Error] Data not sent to Swarm: " + str(err))


            except Exception as err:
                print("error in run",err)
                raise

            except KeyboardInterrupt:
                break

            # Print time (inference-only)
            #@#LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            #@#t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            #@#LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
            break

