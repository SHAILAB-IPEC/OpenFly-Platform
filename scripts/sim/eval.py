import socket
from unrealcv import Client  
import cv2  # OpenCV  
import numpy as np  
import io
import time
from datetime import datetime
import math
import os
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("/cpfs01/user/gaoyunpeng/weights/weight_f_history_1ep", trust_remote_code=True)
base_vla = AutoModelForVision2Seq.from_pretrained(
    "/cpfs01/user/gaoyunpeng/weights/weight_f_history_1ep", 
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
            print('UnrealCV connected successfully')  
        else:  
            print('UnrealCV is not connected')  
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
        print(camera_settings)

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
            # 每一行都是一个 JSON 对象
            # print(line)
            if not line.startswith('{"name":'):
                continue
            json_object = json.loads(line.strip())
            # 你可以在这里处理每个 JSON 对象
            # print(json_object)
            ans_list.append(json_object)
    # return parsed_data
    # print(ans_list)
    return ans_list



# def eval_progress(directory_path):
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(".jsonl"):
#                 file_path = os.path.join(root, file)
#                 with open(file_path, 'r') as f:
#                     json_data = f.read()
                    
#                     # Parse the JSON data
#                     parsed_data = load_test_data(json_data)
#                     #init
#                     print(parsed_data)

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
                return [lst[-111], lst[-2], lst[-1]]
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
    print(text)
    inputs = processor(prompt, image_list).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="vln_norm", do_sample=False)
    action = action.tolist()
    max_value = max(action)
    # print("Action:",action)
    # print('index:',int(action.index(max_value)))
    return int(action.index(max_value))


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

#  将openvla的输出

def getPoseAfterMakeAction(new_pose, action):
    # 解构 new_pose 数组，获取 x, y, z, yaw（pitch 和 roll 都是 0）
    x, y, z, yaw = new_pose

    # 定义步长
    step_size = 5.0  # 平移的步长（单位可以根据需要调整）

    # 根据 action 的值更新 new_pose
    if action == 0:
        # 不变
        pass
    elif action == 1:
        # 向前平移 5 单位（假设是沿着机体系的前方，y 正向）
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        # 向后平移 5 单位（假设是沿着机体系的后方，y 负向）
        x -= step_size * math.cos(yaw)
        y -= step_size * math.sin(yaw)
    elif action == 4:
        # 向左平移 5 单位（假设是沿着机体系的左方，x 负向）
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 5:
        # 向右平移 5 单位（假设是沿着机体系的右方，x 正向）
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 2:
        # 顺时针旋转 10°
        yaw += math.radians(10)
    elif action == 3:
        # 逆时针旋转 10°
        yaw -= math.radians(10)

    # 保证 yaw 在 -π 到 π 之间
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    # 返回更新后的位姿
    return [x, y, z, yaw]



def get_jsonl_files_in_subfolders(directory):
    # 存储所有的jsonl文件路径
    jsonl_files = []
    
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):  # 只获取.jsonl文件
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
    # 启动 TCP 服务器，接收相机位姿
    eval_data_directory = "/home/pjlab/workspace/lch/data_gen/UAV_VLN_Data_Gen/ros_ws/scripts/tmp_data/test1/test1.jsonl"
    # 列表包含多个测试数据，每条数据对应一个json文件
    # test_path = ["test_path1", "test_path2", "..."]
    f = open(eval_data_directory, 'r')
    json_file = json.loads(f.read())
    print("test_path" , json_file)
    # 测试指标
    acc = 0
    stop = 0
    acts = []
    MAX_STEP = 100


    for idx in range(len(json_file)):    

        obj_list = json_file[idx]['pos']
        # print(obj_list)
        start_yaw = 0

        # acts = []
        # vec = obj_list[1]["pos"] - [0]["pos"]
        # start_yaw = math.atan2(vec[1], vec)
        # print(start_yaw)
        text = json_file[idx]['gpt_instrucion']
        start_postion = obj_list[0]
        second_position = obj_list[1]
        start_yaw = calyaw_rad(start_postion, second_position)
        # start_rotation = test_path[idx]['start_rotation']
        end_position = obj_list[-1]
        print(start_postion, "   : ", end_position,"  : ", start_yaw)
        # continue
        stop_error = 1
        image_error = False
        # 实例化 UE5 相机控制类
        ue5_cam_center = UE5CameraCenter()

        new_pose = UE5CameraCenter._camera_init()

        image_list = []
        step = 0

        while step < MAX_STEP:
            #  从ue5中获取当前位置对饮的图片
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


if __name__ == '__main__':
    main()

