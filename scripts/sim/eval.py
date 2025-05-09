import socket
from unrealcv import Client  
import cv2
import numpy as np  
import io
import time
from datetime import datetime
import math
import os
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
from model.prismatic import PrismaticVLM
from model.overwatch import initialize_overwatch
from model.action_tokenizer import ActionTokenizer
from model.vision_backbone import DinoSigLIPViTBackbone, DinoSigLIPImageTransform
from model.llm_backbone import LLaMa2LLMBackbone
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)
logging.basicConfig(filename="./save_result", level=logging.INFO, force=True)

# Load Processor & VLA
model_name_or_path="IPEC-COMMUNITY/openfly-agent-7b"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_name_or_path, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
).to("cuda:0")

class UE5CameraCenter:  
    def __init__(self):  
        self._client = Client(('192.168.31.56', 9000))  
        # self._client = Client(('', 9000))  
        # self._client = Client(('localhost', 9000))  
        self._connection_check()  
        self._camera_init()  
  
        self._lit_image = LitImage()  
        self._object_mask = ObjectMaskImage()  
        self._depth_mask = DepthImage()

    def __del__(self):  
        self._client.disconnect()  
  
    def _connection_check(self):  

        '''检查是否连接'''  
        if self._client.connect():  
            logging.info('UnrealCV connected successfully')  
        else:  
            logging.info('UnrealCV is not connected')  
            exit()
  
    def set_camera_pose(self, x, y, z, pitch, yaw, roll):  
        '''设置摄像头位置'''  
        x = x * 100
        y = y * 100
        z = z * 100
        camera_settings = {  
            'location': {'x': x, 'y': y, 'z': z},  
            'rotation': {'pitch': pitch, 'yaw': yaw, 'roll': roll}  
        }  
        logging.info(camera_settings)

        # 设置相机的位置  
        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))  
        # 设置相机的旋转  
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))  
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))  
        # time.sleep(0.5)


    def _camera_init(self):  
        '''摄像头初始化'''  
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 2560 1440')
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # 初始位置
        time.sleep(1)

    def get_camera_data(self, camera_type):  
        valid_types = {'lit', 'object_mask', 'depth'}  
        if camera_type not in valid_types:  
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")  
        if camera_type == 'lit':  
            return self._lit_image.get_image(self._client)  
        elif camera_type == 'object_mask':  
            return self._object_mask.get_image(self._client)  
        elif camera_type == 'depth':  
            return self._depth_mask.get_image(self._client)
    
    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, camera_type, file_path):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)


class Image:  
    def get_image(self, client):  
        pass


class LitImage(Image):  
    def get_image(self, client):  
        data = client.request('vget /camera/1/lit png')  
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class ObjectMaskImage(Image):  
    def get_image(self, client):  
        data = client.request('vget /camera/1/object_mask png')  
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class DepthImage(Image):  
    def get_image(self, client):  
        data = client.request('vget /camera/1/depth npy')  
        depth_np = np.load(io.BytesIO(data))
        return depth_np  # 返回深度数据

# TCP 服务器类，用于接收相机的位姿



def load_jsonl_data(json_data):
    """
    Parse a string containing JSONL data (multiple JSON objects, one per line).
    
    Args:
    - json_data (str): A string containing multiple JSON objects, separated by new lines.
    
    Returns:
    - list: A list of parsed data, each item being a dictionary extracted from a JSON object.
    """
    # Split the input string into individual JSON objects
    ans_list = []
    with open(json_data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('{"name":'):
                continue
            json_object = json.loads(line.strip())
            ans_list.append(json_object)
    return ans_list


def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 0:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]
        elif step == 1:
            if len(lst) == 2:
                return [lst[-2], lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[-1], lst[-1], lst[-1]]
            
def get_action(image_list, text, his, if_his=False,his_step=0):
    # action_dict = {
    #     "1": "move forward",
    #     "2": "turn left",
    #     "3": "turn right",
    #     "4": "go up",
    #     "5": "go down",
    #     "6": "move left",
    #     "7": "move right",
    # }
    
    prompt = f"In: What action should the robot take to {text}?\nOut:"
    image_list = get_images(image_list,if_his,his_step)
    inputs = processor(prompt, image_list).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="vln_norm", do_sample=False)
    action = action.tolist()
    max_value = max(action)
    return int(action.index(max_value))


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)


def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose
    step_size = 5.0
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        x -= step_size * math.cos(yaw)
        y -= step_size * math.sin(yaw)
    elif action == 4:
        # 向左平移 5 单位（假设是沿着机体系的左方，x 负向）
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 5:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 2:
        yaw += math.radians(10)
    elif action == 3:
        yaw -= math.radians(10)

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]



def get_jsonl_files_in_subfolders(directory):
    jsonl_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"): 
                jsonl_files.append(os.path.join(root, file))
    
    return jsonl_files


def calyaw_rad(start, end):
    vec = np.array(end) - np.array(start)
    start_yaw = math.atan2(vec[1], vec[0])
    if start_yaw > math.pi:
        start_yaw -= 2 * math.pi
    if start_yaw < -math.pi:
        start_yaw += 2 * math.pi
    return start_yaw


def main():
    eval_data_directory = "test.jsonl"
    f = open(eval_data_directory, 'r')
    json_file = json.loads(f.read())
    acc = 0
    stop = 0
    acts = []
    MAX_STEP = 100


    for idx in range(len(json_file)):    

        obj_list = json_file[idx]['pos']
        start_yaw = 0
        text = json_file[idx]['gpt_instrucion']
        start_postion = obj_list[0]
        second_position = obj_list[1]
        start_yaw = calyaw_rad(start_postion, second_position)
        end_position = obj_list[-1]
        # print(start_postion, "   : ", end_position,"  : ", start_yaw)
        stop_error = 1
        image_error = False
        ue5_cam_center = UE5CameraCenter()

        new_pose = UE5CameraCenter._camera_init()

        image_list = []
        step = 0
        act_num = 0

        while step < MAX_STEP:
            image_list.append(ue5_cam_center.get_camera_data('lit'))
            model_action = get_action(image_list, text, acts, if_his=True)
            acts.append(model_action)
            new_pose = getPoseAfterMakeAction(new_pose, model_action)

            act_num += 1
            if model_action == 0:
                stop_error = 0
                break
            step += 1

        if image_error:
            continue    
        model_end_position = new_pose
        dis = calculate_distance(end_position, model_end_position)

        if dis <= 15:
            acc += 1
        acc_ = acc / (data_num + 1)

        stop += stop_error
        f = stop / (data_num + 1)
        data_num += 1
        logging.info(f'num_of_step:{act_num}')
        logging.info(f'data:{idx}, success_rate:{acc_}.')

if __name__ == '__main__':
    main()
