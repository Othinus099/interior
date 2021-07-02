# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
# Modified by Xingyi Zhou: reset metadata.thing_classes using loaded label space
import os
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import json

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder
from classes import furniture_class
#

        
#

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


bed_sofa_label = ['Double_Bed', 'Queen_Bed', 'Sofa001', 'Sofa002']
cabinet_shelf_label = ['BookShelf', 'Cabinet001', 'Cabinet002', 'Glassed_Cabinet', 'Kitchen_Cabine', 'Kitchen_Shelf', 'Living_room_Cabinet', 'Office_Shelf', 'bookcase001', 'bookcase002', 'bookcase003']
chair_label = ['Dining_Chair', 'chair01', 'chair02', 'chair03', 'chair04', 'chair05', 'chair06', 'chair07', 'chair08', 'chair09']
table_label = ['Antique_Table', 'Conference_Table001', 'Conference_Table002', 'Office_Tea_Table', 'Tea_Table', 'table001', 'table002', 'table003', 'table004', 'table005', 'table006', 'table007']
closet_label = ['Stand_Closet', 'bedroom_closet', 'closet03']
drawer_label = ['Drawer001', 'Drawer002', 'Office_Drawer002']
desk_label = ['Computer_Desk001', 'Computer_Desk002', 'Office_Desk', 'Office_Desk004']
tv_label = ['TV_Stand', 'TV_Stand002']

#model_bed_sofa
bs_net = models.resnet50(pretrained=True)
bs_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

bs_net.fc = nn.Linear(2048, 4)  #分成几类
bs_net = bs_net.to(device)
bs_net.load_state_dict(torch.load('./models/model_bs/19.pth'))
bs_net = bs_net.eval()

#model_cf
cf_net = models.resnet50(pretrained=True)
cf_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

cf_net.fc = nn.Linear(2048, 11)  #分成几类
cf_net = cf_net.to(device)
cf_net.load_state_dict(torch.load('./models/model_cf/19.pth'))
cf_net = cf_net.eval()

#model_chair
ch_net = models.resnet50(pretrained=True)
ch_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

ch_net.fc = nn.Linear(2048, 10)  #分成几类
ch_net = ch_net.to(device)
ch_net.load_state_dict(torch.load('./models/model_chair/19.pth'))
ch_net = ch_net.eval()

#model_table
tb_net = models.resnet50(pretrained=True)
tb_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

tb_net.fc = nn.Linear(2048, 12)  #分成几类
tb_net = tb_net.to(device)
tb_net.load_state_dict(torch.load('./models/model_table/19.pth'))
tb_net = tb_net.eval()

#model_closet
cl_net = models.resnet50(pretrained=True)
cl_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

cl_net.fc = nn.Linear(2048, 3)  #分成几类
cl_net = cl_net.to(device)
cl_net.load_state_dict(torch.load('./models/model_closet/19.pth'))
cl_net = cl_net.eval()

#model_drawer
dr_net = models.resnet50(pretrained=True)
dr_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

dr_net.fc = nn.Linear(2048, 3)  #分成几类
dr_net = dr_net.to(device)
dr_net.load_state_dict(torch.load('./models/model_drawer/19.pth'))
dr_net = dr_net.eval()


#model_desk
dk_net = models.resnet50(pretrained=True)
dk_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

dk_net.fc = nn.Linear(2048, 4)  #分成几类
dk_net = dk_net.to(device)
dk_net.load_state_dict(torch.load('./models/model_desk/19.pth'))
dk_net = dk_net.eval()

#model_tv
tv_net = models.resnet50(pretrained=True)
tv_net.conv1 = nn.Conv2d(1,64,kernel_size =7,stride =2, padding=3, bias = False) #for grayscale training

tv_net.fc = nn.Linear(2048, 2)  #分成几类
tv_net = tv_net.to(device)
tv_net.load_state_dict(torch.load('./models/model_tvstand/19.pth'))
tv_net = tv_net.eval()



train_tf = tfs.Compose([
    tfs.Grayscale(1), #for grayscale
    tfs.RandomResizedCrop(224), 
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, ], [0.229, ]) # for grayscale
    #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #使用ImageNet的均值和方差
])


valid_tf = tfs.Compose([
    tfs.Grayscale(1),
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize([0.485, ], [0.229, ])
    #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# train_set = ImageFolder('./database/tv_stand/train/', train_tf)

# print(train_set.classes)



class UnifiedVisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get("__unused")
        unified_label_file = json.load(open(cfg.MULTI_DATASET.UNIFIED_LABEL_FILE))
        self.metadata.thing_classes = [
            '{}'.format([xx for xx in x['name'].split('_') if xx != ''][0]) \
                for x in unified_label_file['categories']]
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                furnitures = []
                instances = predictions["instances"].to(self.cpu_device)
                pred_classes = predictions['instances'].pred_classes.cpu().tolist()
                #######
                class_names = self.metadata.thing_classes
                pred_box_list = predictions['instances'].pred_boxes
                pred_class_names = list(map(lambda x: class_names[x], pred_classes))


                idx = len(predictions["instances"])
                print(f"Number is {idx}")

                save_file = "cur_test"
                # if not os.path.exists(save_file):
	            #     os.mkdir(save_file)
                for box, class_name in zip(pred_box_list, pred_class_names):
                    if(class_name == "cabinet/shelf"):
                            class_name =  "cabinet_shelf"
                    #print(box.cpu().numpy())
                    x0 = int(box.cpu().numpy()[0])
                    y0 = int(box.cpu().numpy()[1])
                    x1 = int(box.cpu().numpy()[2])
                    y1 = int(box.cpu().numpy()[3])
                    crop = image[y0:y1, x0:x1]
                    
                    
                    pil_image = Image.fromarray(crop)
                    net_input = valid_tf(pil_image)
                    


                    out_label = "Not in the dataset"

                    #BED_SOFA
                    if(class_name == "Couch" or 
                        class_name == "Sofa bed" or  
                        class_name == "Loveseat" or
                        class_name == "Bed" or
                        class_name == "Infant bed" or
                        class_name == "studio couch"):
                        out = bs_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = bed_sofa_label[out.max(1)[1].item()]

                    #CABINET_SHELF
                    if(class_name == "cabinet_shelf" or 
                        class_name == "Cabinetry" or  
                        class_name == "Wine rack" or
                        class_name == "Wardrobe" or
                        class_name == "Infant bed" or
                        class_name == "Bookcase" or
                        class_name == "Filing cabinet" or
                        class_name == "Cupboard" or
                        class_name == "Shelf" or
                        class_name == "Bathroom cabinet"):
                        out = cf_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = cabinet_shelf_label[out.max(1)[1].item()]

                    #CHAIR
                    if(class_name == "Chair" or 
                        class_name == "Stool"):
                        out = ch_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = chair_label[out.max(1)[1].item()]

                    #TABLE
                    if(class_name == "Table" or 
                        class_name == "Coffee table" or
                        class_name == "Kitchen & dining room table"):
                        out = tb_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = table_label[out.max(1)[1].item()]

                    #CLOSET
                    if(class_name == "Closet"):
                        out = cl_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = closet_label[out.max(1)[1].item()]

                    #DRAWER
                    if(class_name == "Chest of drawers" or 
                        class_name == "Drawer"):
                        out = dr_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = drawer_label[out.max(1)[1].item()]


                    #DESK
                    if(class_name == "Desk"):
                        out = dk_net(Variable(net_input.unsqueeze(0)).cuda())
                        out_label = desk_label[out.max(1)[1].item()]

                    # #TV_STAND
                    # if(class_name == "Desk"):
                    #     out = tv_net(Variable(net_input.unsqueeze(0)).cuda())
                    #     out_label = desk_label[out.max(1)[1].item()]



                    if(out_label != "Not in the dataset"): 
                        w = image.shape[1]
                        h = image.shape[0]
                        
                        cx = (x0 + x1)/(2 * w)
                        cy = (y0 + y1)/(2 * h)
                        f = furniture_class(out_label,cx,cy)
                        furnitures.append(f)
                        print("\n\n")

                    #print(out.max(1))
                    #print(train_set.classes)
                    #print(class_name)
                    #cv2.imshow('crop', crop)


                    # if not os.path.exists(save_file + "/" + out_label):
	                #     os.mkdir(save_file + "/" + out_label)
                    # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(save_file + "/" + out_label + "/" + str(idx)  + '.jpg', crop)
                    # idx += 1

                
                #print(pred_class_names)
                ###########
                print(f"Total Number of instances : {len(furnitures)}")
                furnitures2 = []
                for furniture in furnitures:
                    furnitures2.append({
                        'class':furniture.label,
                        'cx':furniture.x,
                        'cy':furniture.y
                    })
                    print(f"{furniture.label}`s Center X : {furniture.x}, Y : {furniture.y}")
                jsonTemp = {
                    'num':len(furnitures),
                    'furnitures':furnitures2
                }

        return predictions, jsonTemp

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
