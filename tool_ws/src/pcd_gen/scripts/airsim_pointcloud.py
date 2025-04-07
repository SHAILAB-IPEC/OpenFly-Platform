import airsim
import numpy as np  
import time
from datetime import datetime
import threading
import yaml
from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parents[4]))
print(str(Path(__file__).parents[4]))
from scripts.sim.common import *
# from common import *
import json
import re
import argparse
import subprocess

def get_next_sequence_number(save_dir, env_name):
    files = os.listdir(save_dir + '/' + env_name)
    pattern = re.compile(rf"{env_name}_(\d{{5}})_")
    max_seq_num = 0

    for file in files:
        match = pattern.search(file)
        if match:
            seq_num = int(match.group(1))
            if seq_num > max_seq_num:
                max_seq_num = seq_num

    return max_seq_num + 1

class AirsimBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self.global_point_cnt = 0
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def __del__(self):  
        print("end")
    
    def _connection_check(self):  
        if self._client.confirmConnection():  
            print('Airsim connected successfully')  
            self._client.enableApiControl(True)
            self._client.armDisarm(True)
        else:  
            print('Airsim is not connected')  
            exit()
  
    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(roll, pitch, yaw)
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)

    def get_images(self, camera_names, update_frequency=10):
        responses = []

        for camera_name in camera_names:
            simGetCameraInfo = self._client.simGetCameraInfo(camera_name)
            camera_responses = self._client.simGetImages([
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.Scene),
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.DepthPlanar, True),
                airsim.ImageRequest(f"{camera_name}", airsim.ImageType.Segmentation)
            ])
            responses.extend(camera_responses)

        time.sleep(1 / update_frequency)
        return responses

    def process_images(self, camera_names, save_dir, env_name):
        seq_num = get_next_sequence_number(save_dir, env_name)
        images = self.get_images(camera_names)
        print("images len", len(images))
        print("save images")
        
        for i, response in enumerate(images):
            seq_str = f"{seq_num:05d}"
            image_types = {0: 'color', 1: 'depth', 5: 'object_mask'}
            filename = f"{save_dir}/{env_name}/{env_name}_{seq_str}_{response.camera_name}_{image_types[response.image_type]}"
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        
            if i == 0:
                pose_info = {
                    "id": f"{env_name}_{seq_str}_{response.camera_name}",
                    "pos": {
                        "x": response.camera_position.x_val,
                        "y": response.camera_position.y_val,
                        "z": response.camera_position.z_val
                    },
                    "orient": {
                        "w": response.camera_orientation.w_val,
                        "x": response.camera_orientation.x_val,
                        "y": response.camera_orientation.y_val,
                        "z": response.camera_orientation.z_val
                    }
                }
                jsonl_filename = f"{save_dir}/{env_name}/{env_name}.jsonl"
                with open(jsonl_filename, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(pose_info) + '\n')

    def get_lidar_data(self, update_frequency=400):
        accumulated_points = []

        for _ in range(int(0.005 * update_frequency)):
            lidar_data_horizontal = self._client.getLidarData(lidar_name='LidarSensor1')
            lidar_data_vertical = self._client.getLidarData(lidar_name='LidarSensor2')

            if len(lidar_data_horizontal.point_cloud) < 3 and len(lidar_data_vertical.point_cloud) < 3:
                print("\tNo points received from Lidar data")
                time.sleep(1.0 / update_frequency)
                continue

            if len(lidar_data_horizontal.point_cloud) >= 3:
                points_horizontal = np.array(lidar_data_horizontal.point_cloud, dtype=np.float32)
                points_horizontal = points_horizontal.reshape(-1, 3)
                self.global_point_cnt += points_horizontal.shape[0]
                print(f"\tReceived {points_horizontal.shape[0]} horizontal Lidar points, global count: {self.global_point_cnt}")

                pos_horizontal = lidar_data_horizontal.pose.position
                orientation_horizontal = lidar_data_horizontal.pose.orientation
                rotation_matrix_horizontal = quaternion_to_rotation_matrix(orientation_horizontal.w_val, orientation_horizontal.x_val, orientation_horizontal.y_val, orientation_horizontal.z_val)
                world_points_horizontal = np.dot(points_horizontal, rotation_matrix_horizontal.T) + np.array([pos_horizontal.x_val, pos_horizontal.y_val, pos_horizontal.z_val])
                final_points = world_points_horizontal.copy()
                final_points[:, 1] = -world_points_horizontal[:, 1]
                final_points[:, 2] = -world_points_horizontal[:, 2]
                accumulated_points.append(final_points)

            if len(lidar_data_vertical.point_cloud) >= 3:
                points_vertical = np.array(lidar_data_vertical.point_cloud, dtype=np.float32)
                points_vertical = points_vertical.reshape(-1, 3)
                self.global_point_cnt += points_vertical.shape[0]
                print(f"\tReceived {points_vertical.shape[0]} vertical Lidar points, global count: {self.global_point_cnt}")

                pos_vertical = lidar_data_vertical.pose.position
                orientation_vertical = lidar_data_vertical.pose.orientation
                rotation_matrix_vertical = quaternion_to_rotation_matrix(orientation_vertical.w_val, orientation_vertical.x_val, orientation_vertical.y_val, orientation_vertical.z_val)
                world_points_vertical = np.dot(points_vertical, rotation_matrix_vertical.T) + np.array([pos_vertical.x_val, pos_vertical.y_val, pos_vertical.z_val])
                final_points = world_points_vertical.copy()
                final_points[:, 1] = -world_points_vertical[:, 1]
                final_points[:, 2] = -world_points_vertical[:, 2]
                accumulated_points.append(final_points)

            time.sleep(1.0 / update_frequency)

        if accumulated_points:
            return np.vstack(accumulated_points)
        else:
            return np.array([])

    def save_point_cloud(self, point_cloud, file_path):
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a' if file_exists else 'w') as f:
            if not file_exists:
                f.write('# .PCD v0.7 - Point Cloud Data file format\n')
                f.write('VERSION 0.7\n')
                f.write('FIELDS x y z\n')
                f.write('SIZE 4 4 4\n')
                f.write('TYPE F F F\n')
                f.write('COUNT 1 1 1\n')
                f.write(f'WIDTH {point_cloud.shape[0]}\n')
                f.write('HEIGHT 1\n')
                f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
                f.write(f'POINTS {point_cloud.shape[0]}\n')
                f.write('DATA ascii\n')
            else:
                with open(file_path, 'r') as original_file:
                    lines = original_file.readlines()
                width_line_index = next(i for i, line in enumerate(lines) if line.startswith('WIDTH'))
                points_line_index = next(i for i, line in enumerate(lines) if line.startswith('POINTS'))
                original_width = int(lines[width_line_index].split()[1])
                original_points = int(lines[points_line_index].split()[1])
                new_width = original_width + point_cloud.shape[0]
                new_points = original_points + point_cloud.shape[0]
                lines[width_line_index] = f'WIDTH {new_width}\n'
                lines[points_line_index] = f'POINTS {new_points}\n'
                with open(file_path, 'w') as original_file:
                    original_file.writelines(lines)
            
            np.savetxt(f, point_cloud, fmt='%f %f %f')
        print(f"Point cloud saved to {file_path}")

    def process_lidar_data(self, file_path):
        point_cloud = self.get_lidar_data()
        if point_cloud.size > 0:
            self.save_point_cloud(point_cloud, file_path)

def handle_planner(config_params, global_configs, type):
    env_name = global_configs['datagen']['env']
    airsim_bridge = AirsimBridge(env_name)
    file_path = "tool_ws/src/pcd_gen/tmp_pcd_map/" if type == 'lidar' else 'image/'
    

    map_bound = global_configs['traj_map']['MapBound']
    lidar_delta = global_configs['traj_map']['LidarDelta']

    x_min, x_max, y_min, y_max, z_min, z_max = map_bound
    dx, dy, dz = lidar_delta

    for z in range(z_min, z_max, dz):
        for x in range(x_min, x_max, dx):
            for y in range(y_min, y_max, dy):

                airsim_bridge.set_drone_pos(x, y, z, 0, 0, 0)
                print(f"Current drone position: ({x}, {y}, {z})")
                time.sleep(0.05)
                
                if type == 'image':
                    time.sleep(0.5)
                    camera_names = ["bottom_custom"]
                    airsim_bridge.process_images(camera_names, file_path, env_name)

                elif type == 'lidar':
                    airsim_bridge.process_lidar_data(file_path + env_name + '.pcd')

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="env name")
    
    parser.add_argument('--env', type=str, default='env_airsim_16', help="input env name")
    parser.add_argument('--type', type=str, default='lidar', help="data type")

    args = parser.parse_args()

    config_file = "configs/" + args.env + ".yaml"

    global_configs = load_config(config_file)
    planner_configs = global_configs['thread_params']

    threads = []
    for config in planner_configs:
        thread = threading.Thread(target=handle_planner, args=(config, global_configs, args.type))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()