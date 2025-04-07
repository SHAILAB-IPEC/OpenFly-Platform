import subprocess
import os
import requests
import argparse
import psutil
import socket 
import numpy as np  
import time
import threading
import yaml
from common import *

class GSBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)
  
    def _init_gs_sim(self):
        dataset_dir = "envs/gs/" + self.env_name  
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw
    
    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text) 
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass

class TCPServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.server_address = (host, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(1)
        print(f"Listening on {host}:{port}")
    
    def accept_connection(self):
        conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")
        return conn
    
    def receive_pose(self, conn):
        data = conn.recv(1024).decode()
        print("data", data)
        return data

def handle_planner(config_params, env_name):
    gs_bridge = GSBridge(env_name)
    tcp_server = TCPServer(port=config_params['aim_port'])
    file_path = "uav_vln_data/test"
    image_id = 0
    while True:
        print("ready connet")
        conn = tcp_server.accept_connection()
        print("Connection established!")
        while True:
            data = tcp_server.receive_pose(conn)
            if not data:
                break
            if "path:" in data:
                file_path = data.split("path:")[1]
                image_id = 0
                print("file_path:", file_path)
                continue
            pose = list(map(float, data.split(',')))
            if len(pose) != 6:
                print("Invalid pose data!")
                continue
            print(f"Received pose: {pose}")
            gs_bridge.set_camera_pose(*pose, file_path)
            time.sleep(0.1)
            image_id += 1
            print("Image saved!")

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['thread_params']

def main():
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_gs_sjtu02', help="input env name")
    args = parser.parse_args()
    config_file = "configs/" + args.env + ".yaml"
    planner_configs = load_config(config_file)
    threads = []
    for config in planner_configs:
        thread = threading.Thread(target=handle_planner, args=(config, args.env))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()