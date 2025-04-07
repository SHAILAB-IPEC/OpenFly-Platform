import json
import os


def extract_number(s):
    # 移除 `.png` 后缀部分
    if s.endswith(".png"):
        s = s.rsplit('.', 1)[0]  # 删除文件扩展名

    # 查找最后一个 "_" 后的数字部分
    last_underscore_pos = s.rfind("_")
    if last_underscore_pos == -1:
        return float("inf")  # 没有找到 "_" 时，返回一个较大的值，表示该项排最后

    number_part = s[last_underscore_pos + 1:]

    # 尝试将数字部分转换为整数，如果失败则返回 float('inf')
    try:
        return int(number_part)
    except ValueError:
        return float("inf")

def sort_strings(strings):
    # 使用 sorted 排序，根据数字部分排序
    return sorted(strings, key=extract_number)



def count_png_images(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                count += 1
    return count

def get_png_images(folder_path):
    png_paths = []
    img_id = []
    for root, dirs, files in os.walk(folder_path):
        # 对子目录和文件列表排序
        dirs.sort()  # 按子目录名排序
        files.sort()  # 按文件名排序
        for file in files:
            if file.lower().endswith('.png'):
                # 获取文件的完整路径
                png_paths.append(os.path.join(root, file))
                img_id.append(file.replace('.png',''))
            if file.lower().endswith('.jpg'):
                # 获取文件的完整路径
                png_paths.append(os.path.join(root, file))
                img_id.append(file.replace('.jpg',''))
    
    # 如果需要整体排序，可以基于路径排序
    png_paths = sort_strings(png_paths)
    img_id = sort_strings(img_id)
    return png_paths,img_id



def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 用来存储完整的 JSON 对象
    json_objects = []

    # 变量用于跟踪大括号的嵌套层级
    brace_count = 0
    start = 0

    # 遍历文件内容
    for idx, char in enumerate(content):
        if char == '{':
            # 增加嵌套层级
            if brace_count == 0:
                start = idx  # 记录开始位置
            brace_count += 1
        elif char == '}':
            # 减少嵌套层级
            brace_count -= 1
            if brace_count == 0:
                # 当嵌套层级归零时，意味着我们找到了一个完整的 JSON 对象
                json_str = content[start:idx+1]
                try:
                    # 将字符串解析为字典
                    json_objects.append(json.loads(json_str))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue
    action_dicts = [d for d in json_objects if 'action' in d]
    aim_landmark = [d for d in json_objects if 'aim_landmark' in d]

    if len(aim_landmark) == 0:
        aim_landmark = [{'aim_landmark':{"error":None}}] 
    return action_dicts,aim_landmark[-1]




def merge_adjacent(data):
    """
    根据 'name' 合并相邻的相同对象，并对 'value' 求和。
    
    :param data: 输入数据列表，格式为 [{'name': ..., 'value': ...}, ...]
    :return: 合并后的数据列表
    """
    if not data:
        return []

    merged_data = [data[0]]  # 初始化结果列表，包含第一个元素

    for item in data[1:]:
        # 如果当前对象的 name 与前一个合并对象的 name 相同
        if item['name'] == merged_data[-1]['name']:
            # 合并 value
            merged_data[-1]['value'] += item['value']
        else:
            # 不同的 name，作为新的对象加入结果列表
            merged_data.append(item)

    return merged_data

def find_consecutive_turns(actions):
    consecutive_turns = []
    
    for i in range(1, len(actions) - 2):  # 从第二个到倒数第二个元素
        current_action = actions[i]['action']['type']
 
        next_action = actions[i + 1]['action']['type']
        
        # 如果前后动作不同且当前动作和前一个动作相同
        if  current_action != next_action:
            consecutive_turns.append(i)
    
    return consecutive_turns


def check(file_path):
    # 检查并重命名 `pose.json` 为 `pose.jsonl`
    pose_json_path = os.path.join(file_path, 'pose.json')
    pose_jsonl_path = os.path.join(file_path, 'pose.jsonl')
    
    if os.path.exists(pose_json_path):
        os.rename(pose_json_path, pose_jsonl_path)

    # 检查是否存在 `pose.jsonl`
    if os.path.exists(pose_jsonl_path):
        json_object = read_jsonl(pose_jsonl_path)
       
        acts_len = len(json_object)
        imgs_len = count_png_images(file_path)  # 获取 PNG 图片数量

        # 验证动作长度和图片数量的关系
        return acts_len == imgs_len 
    else:
        return False


def process_act(actions):
    # 定义动作名称与数字的对应关系
    action_map = {
        "stop":0,
        "go straight": 1,
        "turn left": 2,
        "turn right": 3,
        "go up": 4,
        "go down": 5,
        "move left": 6,
        "move right": 7,
        'up':-1,
        'down':-2
    }
    
    # 转换列表
    transformed_list = [action_map[action['action']['type']] for action in actions]
    
    return transformed_list



