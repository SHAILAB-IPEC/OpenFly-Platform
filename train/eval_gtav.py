#!/usr/bin/env python3 
import numpy as np  
import cv2
from collections import deque
import io
import time
import subprocess, threading
from datetime import datetime
import math
from tqdm import tqdm
import random
import os
import json
import torch
from PIL import Image as Image_
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  

from utils.Constants import IMG_WIDTH, IMG_HEIGHT
from PIL import ImageGrab
from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed
from deepgtav.client import Client
from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_meta_data, getRunCount, generateNewTargetLocation
from scipy.spatial.transform import Rotation as R
import argparse
from PIL import Image
import pyautogui
import shutil
import select
import socket
from datetime import datetime
import threading
import yaml
from pathlib import Path

class GTAVBridge:
    def __init__(self, host='127.0.0.1', port=8000, save_dir='./output'):
        self.host = '127.0.0.1'
        self.port = 8000
        self.save_dir = os.path.normpath(save_dir)
        self.client = Client(ip=self.host, port=self.port)
        scenario = Scenario(drivingMode=[786603,0], vehicle="voltic", location=[245.23306274414062, -998.244140625, 29.205352783203125])
        dataset = Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=5000)   
        self.client.sendMessage(Start(scenario=scenario, dataset=dataset))  

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        pitch = 0.0 
        roll = 0.0

        rotation = R.from_euler('y', 90, degrees=True)  # Inverse rotation
        rotation_matrix = rotation.as_matrix()

        out_x = -y
        out_y = x
        position = np.array([out_x, out_y, z])
        rotated_position = np.dot(rotation_matrix, position)

        rotation_angles = np.array([roll, pitch, yaw])  # [pitch, yaw, roll]
        rotated_angles = rotation.inv().apply(rotation_angles)

        # print(f"Original Position: x: {x}, y: {y}, z: {z}")
        # print(f"Rotated Position: x: {rotated_position[0]}, y: {rotated_position[1]}, z: {rotated_position[2]}")
        # print(f"Original Angles: pitch: {pitch}, yaw: {yaw}, roll: {roll}")
        # print(f"Rotated Angles: pitch: {rotated_angles[0]}, yaw: {rotated_angles[1]}, roll: {rotated_angles[2]}")
        
        self.client.sendMessage(SetClockTime(12))
        message = self.client.recvMessage() 
        pos =  message["location"]
        target_position = (position[0], position[1], position[2])
        current_position = (pos[0], pos[1], pos[2])
        distance = self.calculate_distance_3d(current_position, target_position)
        cccc=0
        while distance > 1 and cccc<10:
            # 重设位置
            cccc+=1
            self.client.sendMessage(SetCameraPositionAndRotation(
                0, 0, -10,
                rotation_angles[0], rotation_angles[1], -rotation_angles[2]
            ))
            self.client.sendMessage(TeleportToLocation(position[0], position[1], position[2]))
            
            # 等待0.1秒
            time.sleep(0.05)
            
            # 获取新位置
            message = self.client.recvMessage()
            pos = message["location"]
            current_position = (pos[0], pos[1], pos[2])
            
            # 重新计算距离
            distance = self.calculate_distance_3d(current_position, target_position)
       

            self.client.sendMessage(SetClockTime(12))
        print(message['CameraAngle'])
        time.sleep(0.02)


    def calculate_distance_3d(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + 
                        (point2[1] - point1[1])**2 + 
                        (point2[2] - point1[2])**2)

    def get_camera_data(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 定义截取区域（左上角x, 左上角y, 右下角x, 右下角y）
        bbox = (320, 171, 2239, 1250)
        screenshot = ImageGrab.grab(bbox=bbox)
        
        screenshot.save(file_path)
        return cv2.imread(file_path)

    # def get_camera_data(self):
    #     output_path = "C:/Program Files/Epic Games/GTAVEnhanced/color.raw"
    #     self.press_key_l()
    #     output_filename = output_path.replace(".raw", ".jpg")
    #     return cv2.imread(output_path)

    def press_key_l(self):
        """Simulate pressing the 'l' key"""
        pyautogui.press('l')

    def move_raw_image(self, source, destination):
        """Move raw image file to the destination path"""
        shutil.move(source, destination)

    def stop_client(self):
        """Stop the client and disconnect"""
        self.client.sendMessage(Stop())
        self.client.close()

class TCPServer:
    def __init__(self, host='0.0.0.0', port=9999, timeout=80):
        self.server_address = (host, port)
        self.timeout = timeout  # Timeout in seconds
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(1)
        print(f"Listening on {host}:{port}")
    
    def accept_connection(self):
        conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")
        return conn
    
    def receive_pose(self, conn):
        ready_to_read, _, _ = select.select([conn], [], [], self.timeout)
        
        if ready_to_read:
            data = conn.recv(1024).decode()
            return data
        else:
            print(f"No data received within {self.timeout} seconds.")
            return 'timeout'


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        # print(config)
    return config['thread_params']

def find_pose_json_folders(root_folder):
    """递归查找有且只有pose.json文件的最深层文件夹"""
    pose_folders = []
    
    for root, dirs, files in os.walk(root_folder):
        dirs.sort() 
        if not dirs:
            if len(files) == 2 and files[1] == 'pose.json':
                pose_folders.append(root)
    
    return pose_folders



# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result

def get_images(lst, if_his, step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def get_action(policy, processor, image_list, text, if_his=False, his_step=0):

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    return cur_action

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]



def main():

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = 100

    eval_data_directory = "gtav.json"
    f = open(eval_data_directory, 'r')
    eval_info = json.load(f)

    env_name = "env_game_gtav"
        
    env_bridge = GTAVBridge()
    output_file = f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    pos_ratio = 1.0

    message = env_bridge.client.recvMessage()

    # Load model
    model_name_or_path="IPEC-COMMUNITY/openfly-agent-7b"
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    policy = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path, 
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
    ).to("cuda:0")

    for sample_idx, item in enumerate(eval_info):
        if not "image_path" in item.keys():
            continue
        if not item['image_path'].__contains__(env_name):
            continue

        step = 0
        flag_osr = 0
        image_list = []
        env_bridge.pass_len = 1e-3
        os.makedirs(f"{env_name}/{sample_idx}", exist_ok=True)

        pos_list = item['pos']
        text = item['gpt_instruction']
        start_postion = pos_list[0]
        start_yaw = item['yaw'][0]
        new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
        old_pose = new_pose
        end_position = pos_list[-1]
        print(f"Sample {sample_idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")
        
        stop_error = 1
        image_error = False
        
        # Set camera pose
        pitch = -45.0 if 'high' in item['image_path'] else 0.0
        env_bridge.set_camera_pose(
            start_postion[0]/pos_ratio, 
            start_postion[1]/pos_ratio, 
            start_postion[2]/pos_ratio, 
            pitch, 
            np.rad2deg(start_yaw), 
            0
        )

        while step < MAX_STEP:    
            
            image = env_bridge.get_camera_data(f"{env_name}/{sample_idx}/{str(step).zfill(3)}.png")

            image_list.append(image)

            model_action = get_action(policy, processor, image_list, text, if_his=True, his_step=2)
            print(model_action)
            new_pose = getPoseAfterMakeAction(new_pose, model_action)

            # print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
            env_bridge.set_camera_pose(
                new_pose[0]/pos_ratio, 
                new_pose[1]/pos_ratio, 
                new_pose[2]/pos_ratio, 
                0.0, 
                np.rad2deg(new_pose[3]),
                0
            )

            env_bridge.pass_len += calculate_distance(old_pose, new_pose)
            dis = calculate_distance(end_position, new_pose)
            if dis < 40 and flag_osr != 2:
                flag_osr = 2
                env_bridge.osr.append(1)
            old_pose = new_pose

            if model_action == 0:
                stop_error = 0
                break
            step += 1

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            env_bridge.distance_to_goal.append(dis)
            if dis < 20:
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(0)

            if image_error:
                continue
                
            # Save individual sample statistics to jsonl
            sample_stats = {
                "env_name": env_name,
                "sample_idx": sample_idx,
                "instruction": text,
                "start_position": start_postion,
                "end_position": end_position,
                "final_position": new_pose[:3],
                "distance_to_goal": dis,
                "success": 1 if dis < 20 else 0,
                "spl": env_bridge.spl[-1],
                "osr": env_bridge.osr[-1],
                "steps_taken": step,
                "trajectory_length": env_bridge.traj_len,
                "path_length": env_bridge.pass_len,
                "stop_error": stop_error,
                "image_error": image_error
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample_stats, ensure_ascii=False) + '\n')
            
            data_num += 1
            if dis < 20:
                acc += 1
            stop += stop_error
                
        # Save environment summary statistics
        env_summary = {
            "type": "environment_summary",
            "env_name": env_name,
            "total_samples": len(eval_info),
            "success_rate": sum(env_bridge.success) / len(env_bridge.success) if env_bridge.success else 0,
            "spl": sum(env_bridge.spl) / len(env_bridge.spl) if env_bridge.spl else 0,
            "osr": sum(env_bridge.osr) / len(env_bridge.osr) if env_bridge.osr else 0,
            "avg_distance_to_goal": sum(env_bridge.distance_to_goal) / len(env_bridge.distance_to_goal) if env_bridge.distance_to_goal else 0,
            "success_count": sum(env_bridge.success),
            "total_distance_to_goal": sum(env_bridge.distance_to_goal)
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(env_summary, ensure_ascii=False) + '\n')
        
        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        f.close()
        

if __name__ == '__main__':
    main()
