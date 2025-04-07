import math
import numpy as np
from math import atan2, asin, degrees, cos, sin, sqrt

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.
    The order of Euler angles is yaw-pitch-roll (Z-Y-X axis rotation order).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * cr + cy * sp * sr
    z = sy * cp * sr - cy * sp * cr
    
    return [x, y, z, w]

def quaternion_to_rotation_matrix(QW, QX, QY, QZ):
    """
    Convert a quaternion to a rotation matrix.
    """
    R = np.array([
        [1 - 2*(QY**2 + QZ**2), 2*(QX*QY - QZ*QW), 2*(QX*QZ + QY*QW)],
        [2*(QX*QY + QZ*QW), 1 - 2*(QX**2 + QZ**2), 2*(QY*QZ - QX*QW)],
        [2*(QX*QZ - QY*QW), 2*(QY*QZ + QX*QW), 1 - 2*(QX**2 + QY**2)]
    ])
    return R

def calculate_camera_position(QW, QX, QY, QZ, TX, TY, TZ):
    """
    Calculate the camera position in the world coordinate system.
    """
    R = quaternion_to_rotation_matrix(QW, QX, QY, QZ)
    R_t = R.T
    T = np.array([TX, TY, TZ])
    camera_position_world = -R_t.dot(T)
    return camera_position_world

def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).
    """
    pitch = asin(-R[2, 0])
    if abs(R[2, 0]) != 1:
        roll = atan2(R[2, 1], R[2, 2])
        yaw = atan2(R[1, 0], R[0, 0])
    else:
        roll = atan2(-R[1, 2], R[1, 1])
        yaw = 0
    
    roll = degrees(roll)
    pitch = degrees(pitch)
    yaw = degrees(yaw)
    
    return roll, pitch, yaw

def cam2world(QW, QX, QY, QZ, TX, TY, TZ):
    camera_position = calculate_camera_position(QW, QX, QY, QZ, TX, TY, TZ)
    formatted_position = [f"{x:.8f}" for x in camera_position]
    print(f"Camera position in world coordinates: {', '.join(formatted_position)}")

    R = quaternion_to_rotation_matrix(QW, QX, QY, QZ)
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    print(f"Camera's orientation (Roll, Pitch, Yaw): {roll}, {pitch}, {yaw}")
    rpy = [roll, pitch, yaw]

    return camera_position, rpy

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    R_x = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    
    R_y = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    
    R_z = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (QW, QX, QY, QZ).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = sqrt(trace + 1.0) * 2
        QW = 0.25 * S
        QX = (R[2, 1] - R[1, 2]) / S
        QY = (R[0, 2] - R[2, 0]) / S
        QZ = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            S = sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            QW = (R[2, 1] - R[1, 2]) / S
            QX = 0.25 * S
            QY = (R[0, 1] + R[1, 0]) / S
            QZ = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            QW = (R[0, 2] - R[2, 0]) / S
            QX = (R[0, 1] + R[1, 0]) / S
            QY = 0.25 * S
            QZ = (R[1, 2] + R[2, 1]) / S
        else:
            S = sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            QW = (R[1, 0] - R[0, 1]) / S
            QX = (R[0, 2] + R[2, 0]) / S
            QY = (R[1, 2] + R[2, 1]) / S
            QZ = 0.25 * S
    
    return QW, QX, QY, QZ

def calculate_camera_position_from_world(R_t, camera_position):
    """
    Given camera_position_world and the transpose of the rotation matrix R_t,
    calculate the translation vector T.
    """
    T = -R_t.dot(camera_position)
    return T

def world2cam(x, y, z, roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    QW, QX, QY, QZ = rotation_matrix_to_quaternion(R)
    print(f"Quaternion (QW, QX, QY, QZ): {QW}, {QX}, {QY}, {QZ}")

    T = calculate_camera_position_from_world(R, np.array([x, y, z]))
    formatted_T = [f"{t:.8f}" for t in T]
    print(f"Camera position in camera coordinates (TX, TY, TZ): {', '.join(formatted_T)}")

    return QW, QX, QY, QZ, T[0], T[1], T[2]

def world2cam_WXYZ(x, y, z, QW, QX, QY, QZ):
    """
    Converts world coordinates and quaternion to camera coordinates (position and orientation).
    """
    R = quaternion_to_rotation_matrix(QW, QX, QY, QZ)
    position_world = np.array([x, y, z])
    camera_position = calculate_camera_position_from_world(R, position_world)
    formatted_camera_position = [f"{t:.8f}" for t in camera_position]
    print(f"Camera position in camera coordinates (TX, TY, TZ): {', '.join(formatted_camera_position)}")

    return camera_position