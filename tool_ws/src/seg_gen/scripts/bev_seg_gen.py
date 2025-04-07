import cv2
import numpy as np
import open3d as o3d
import json
import yaml
import os
from pathlib import Path
import argparse

def find_black_regions_centers(pgm_file):

    img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)


    _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((3, 3), np.uint8)  
    eroded_img = cv2.erode(binary_img, kernel, iterations=2)


    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    centers = []
    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:  

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    return img, centers


def load_map_config(yaml_file):
    """
    从YAML文件加载地图配置，提取resolution和origin，并将origin转化为二维坐标。
    
    :param yaml_file: 配置文件路径
    :return: (resolution, origin) 元组
    """
    with open(yaml_file, 'r') as file:

        config = yaml.safe_load(file)
    

    resolution = config.get('resolution', 4.0) 
    origin = config.get('origin', [0, 0, 0.0])  
    
    origin_2d = origin[:2]  
    
    return resolution, origin_2d

def visualize_point_cloud_with_real_coordinates(ply_file, real_coordinates, height=0.0):
    """
    可视化点云与通过栅格地图获得的真实坐标点，并修改真实坐标点的高度(z坐标)。

    :param ply_file: PLY格式的点云文件路径
    :param real_coordinates: 真实坐标点 [(real_x, real_y), ...]
    :param height: 修改后的z坐标值（默认值为0.0）
    """

    # print("wait pcd read")
    pcd = o3d.io.read_point_cloud(ply_file)
    

    real_points = np.array(real_coordinates)
    

    real_points[:, 2] += 1 

    # print("wait pcd show")


    real_pcd = o3d.geometry.PointCloud()
    real_pcd.points = o3d.utility.Vector3dVector(real_points)
    

    real_pcd.paint_uniform_color([1, 0, 0])  

    o3d.visualization.draw_geometries([pcd, real_pcd], window_name="Point Cloud with Real Coordinates")

def convert_to_jsonl_format(real_coordinates, output_filename="real_coordinates.jsonl"):


    with open(output_filename, 'w') as outfile:
        for (real_x, real_y, real_z) in real_coordinates:

            filename = f"X={real_x}Y={real_y}Z={real_z}.png"

            record = {
                "type": "building",
                "color": "test",
                "size": "test",
                "shape": "test",
                "feature": "test",
                "filename": filename
            }
            

            json.dump(record, outfile)
            outfile.write("\n") 

    print(f"JSONL文件已保存为 {output_filename}")


def find_highest_point_in_point_cloud(points, target_x, target_y, tolerance=5.0):

    points = np.array(points)
    
    x_diff = np.abs(points[:, 0] - target_x)
    y_diff = np.abs(points[:, 1] - target_y)
    
    mask = (x_diff <= tolerance) & (y_diff <= tolerance)
    
    target_points = points[mask]
    
    if target_points.size == 0:
        return 0.0  
    
    highest_point_z = np.max(target_points[:, 2])
    return highest_point_z
    

def get_real_z_from_point_cloud(pcds, x, y):

    
    points = np.asarray(pcds.points)
    
    highest_z = find_highest_point_in_point_cloud(points, x, y)
    return highest_z

def read_ply(file_path):

    point_cloud = o3d.io.read_point_cloud(file_path)
    

    print(f"点云包含 {len(point_cloud.points)} 个点")
    
    return point_cloud


def convert_to_real_coordinates(img, centers, resolution, origin, ply_file):

    points = read_ply(ply_file)

    image_width = img.shape[0]
    print("image_width", image_width)
    real_coordinates = []
    for (cx, cy) in centers:
        real_x = origin[0] + cx * resolution
        real_y = (image_width-cy) * resolution + origin[1]
        real_z = get_real_z_from_point_cloud(points, real_x, real_y)

        real_coordinates.append((real_x, real_y, real_z))

    return real_coordinates


def load_yaml_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_airsim_16', help="input env name")
    args = parser.parse_args()
    env_ = args.env

    cur_path = Path(__file__).resolve()
    pro_path_ = cur_path.parent.parent.parent.parent.parent

    seg_data_file = str(pro_path_) + "/tool_ws/src/seg_gen/env_seg_info/"
    scene_data_file = str(pro_path_) + "/scene_data/"

    pcd_file = scene_data_file +  "/pcd_map/" + env_ + ".pcd"  
    pgm_file = seg_data_file + env_ + "/" + env_ + ".pgm"
    yaml_file = seg_data_file + env_ + "/" + env_ + ".yaml"
    jsonl_file = seg_data_file + env_ + "/" + env_ + ".jsonl"
    jsonl_file_in_data = scene_data_file + "/seg_map/" + env_ + ".jsonl"
    
    print("pgm_file", pgm_file)
    img, centers = find_black_regions_centers(pgm_file)
    resolution , origin = load_map_config(yaml_file)

    real_coordinates = convert_to_real_coordinates(img, centers, resolution, origin, pcd_file)

    convert_to_jsonl_format(real_coordinates, output_filename = jsonl_file)
    convert_to_jsonl_format(real_coordinates, output_filename = jsonl_file_in_data)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (cx, cy) in centers:
        cv2.circle(img_color, (cx, cy), 2, (0, 0, 255), -1) 


    cv2.imwrite(seg_data_file + env_ + "/" + env_ + 'output_with_centers.jpg', img_color)

    print("show img")
    visualize_point_cloud_with_real_coordinates(pcd_file, real_coordinates)
    print("after show")


