#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT

from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_meta_data, getRunCount, generateNewTargetLocation
from scipy.spatial.transform import Rotation as R
import argparse
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from random import uniform
from math import sqrt
import numpy as np
import pyautogui
import os
import base64
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

        self.test_x = 0.0
        self.test_y = 0.0
        self.test_z = 120.0

        self.test_roll = 0.0
        self.test_pitch = 0.0
        self.test_yaw = 0.0

    def setup_scenario(self):
        """Set up the scenario, dataset, and send start command"""
        scenario = Scenario(drivingMode=[786603,0], vehicle="voltic", location=[245.23306274414062, -998.244140625, 29.205352783203125])
        dataset = Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=5000)   
        self.client.sendMessage(Start(scenario=scenario, dataset=dataset))

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

        print(f"Original Position: x: {x}, y: {y}, z: {z}")
        print(f"Rotated Position: x: {rotated_position[0]}, y: {rotated_position[1]}, z: {rotated_position[2]}")
        print(f"Original Angles: pitch: {pitch}, yaw: {yaw}, roll: {roll}")
        print(f"Rotated Angles: pitch: {rotated_angles[0]}, yaw: {rotated_angles[1]}, roll: {rotated_angles[2]}")
        
        self.client.sendMessage(SetCameraPositionAndRotation(
            0, 0, -10,
            rotation_angles[0], rotation_angles[1], rotation_angles[2]
        ))
        self.client.sendMessage(TeleportToLocation(position[0], position[1], position[2]))

        time.sleep(0.02)

    def process_camera_data(self, file_path):
        output_path = file_path
        output_filename = output_path.replace(".raw", ".jpg")
        print(f"Converted to {output_filename}")
        shutil.move("./color.raw", file_path)

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

def handle_planner(config_params):
    tcp_server = TCPServer(port=config_params['aim_port'])
    print("config_params['ue_ip']:", config_params['ue_ip'])
    print("config_params['ue_port']:", config_params['ue_port'])
    gtav_bridge = GTAVBridge(host=config_params['ue_ip'], port=config_params['ue_port'])
    file_path = "./data_gen/tmp"

    image_id = 0
    while True:
        print("ready to connect")
        conn = tcp_server.accept_connection()
        print("Connection established!")
        i = 0
        
        while True:
            data = tcp_server.receive_pose(conn)
            if not data:
                break
            if "path:" in data:
                base_path = "./data_gen/color_file/"
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                image_id = 0
                file_path = os.path.join(base_path, current_time)
                
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    print(f"Directory {file_path} created successfully.")
                print("file_path:", file_path)
                continue

            pose = list(map(float, data.split(',')))
            print(f"Received pose: {pose}")

            gtav_bridge.set_camera_pose(*pose)

            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_")

            gtav_bridge.process_camera_data(file_path + "\\" + current_time_str + str(image_id) + '.png')
            time.sleep(0.1)
            image_id += 1
            print("Image saved!")

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['date_gen_config']['thread_params']

def main():
    parser = argparse.ArgumentParser(description="env name")
    print("test !!")
    parser.add_argument('--env', type=str, default='env_game_gtav', help="input env name")

    args = parser.parse_args()

    print(args.env)
    cur_path = Path(__file__)
    ros_dir = cur_path.parent.parent
    config_file = args.env + ".yaml"

    planner_configs = load_config(config_file)

    threads = []
    for config in planner_configs:
        thread = threading.Thread(target=handle_planner, args=(config,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    print("test")
    main()