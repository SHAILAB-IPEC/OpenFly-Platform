import base64

import random
import logging
from mimetypes import guess_type
from openai import OpenAI
from process import get_png_images, read_jsonl, find_consecutive_turns
 
        
class OpenAIClient:
    def __init__(self,api_key,model):
        """
        初始化 Azure OpenAI 客户端
        """
        self.client = OpenAI(
            api_key=api_key,
        )
        self.model = model
        self.token_usage = 0  # 记录 Token 使用量
        self.input_token = 0
        self.output_token = 0
    
    def update_token_usage(self, response):
        """
        更新 Token 使用量
        """
        self.token_usage += response.usage.total_tokens
        self.input_token += response.usage.prompt_tokens
        self.output_token += response.usage.completion_tokens 

    def local_image_to_data_url(self, image_path):
        """
        将本地图像转换为 Base64 编码的 Data URL
        """
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"
    def get_landmark(self, img_path):
        """
        使用 GPT-4 模型识别图像中的地标
        """
        try:
            # 将图片转换为 Base64 数据
            data_url = self.local_image_to_data_url(img_path)

            # 调用 Azure OpenAI 客户端
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant proficient in image recognition. You can accurately identify the object closest to you in the image and its different features from surrounding objects, and reply to me in JSON format."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The target is the nearest prominent landmark to me。Answer me a dictionary like {color:, feature: , size: , type: }."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ]
            )
            self.update_token_usage(response)  # 更新 token 计数
            output = response.choices[0].message.content
            output = output.strip("```json").strip("```")
            return output
        except Exception as e:
            logging.error(f"Error in get_landmark: {e}")
            return None
        
    def get_aim_landmark(self, img_paths,info):
        """
        使用 GPT-4 模型识别图像中的地标
        """
        img_data = []
        for path in img_paths:
            data_url = self.local_image_to_data_url(path)
            img_data.append(data_url)
        if 'dir' in info['aim_landmark']:
            dirs = f"The target building is at {info['aim_landmark']['dir']} of the image."
        else:
            dirs = f"The target building is at center of the image."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who is proficient in image recognition. You can accurately identify the object in the picture and its characteristics that are different from the surrounding objects.I will give you the three final images you will see. Please focus on the last image and tell me the features of the target building and reply to me in the form of JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": dirs +" Answer me a dictionary like {color: **, feature: **, size: **, type: **}."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_data[0]
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_data[1]
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_data[2]
                            }
                        }
                    ]
                }
            ]
        )
        self.update_token_usage(response)  # 更新 token 计数
        output = response.choices[0].message.content
        output = output.strip("```json").strip("```")
        return output

    def get_action(self, action, data):
        """
        根据动作类型生成动作描述
        """
        action_dic = {
            'go straight': ['Proceed straight', 'Move ahead', 'Advance forward', 'Move forward', 'Walk straight', 'Head straight', 'Keep going straight', 'Go directly ahead'],
            'turn left': ['Go left', 'Take a left', 'Make a left turn', 'Turn to the left', 'Shift left', 'Veer left'],
            'turn right': ['Go right', 'Take a right', 'Make a right turn', 'Turn to the right', 'Shift right', 'Veer right'],
            'move left': ['Shift to the left', 'Step left', 'Slide left', 'Move towards the left', 'Lean left', 'Adjust leftward'],
            'move right': ['Shift to the right', 'Step right', 'Slide right', 'Move towards the right', 'Lean right', 'Adjust rightward'],
            'go up': ['Move up', 'Ascend', 'Rise', 'Go upwards', 'Climb', 'Elevate'],
            'go down': ['Move down', 'Descend', 'Fall', 'Go downwards', 'Drop', 'Lower'],
            'up':['Move up', 'Ascend', 'Rise', 'Go upwards', 'Climb', 'Elevate'],
            'down':['Move down', 'Descend', 'Fall', 'Go downwards', 'Drop', 'Lower']
        }

        if action in action_dic:
            straight = random.sample(action_dic['go straight'], 1)
            if action == 'go straight':
                return str(straight) + 'to' + str(data) 
            else:
                act = random.sample(action_dic[action], 1)
                return str(act) + 'to' + str(data) 
        else:
            return "Invalid action key"

    def get_instruction(self, actions):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant proficient in text processing. You need to help me combine these scattered actions and landmarks into a sentence using words with similar meanings and more appropriate words, making them smooth, fluent, and accurate. If the landmarks of adjacent actions are similar or even identical, please use pronouns to refer to them."
                    },
                    {
                        "role": "user",
                        "content": f"Data: {actions}."
                    }
                ]
            )
            self.update_token_usage(response)  # 更新 token 计数
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in get_instruction: {e}")
            return 'None'

    def process(self, file_path):
        """
        同步处理指定文件夹路径的数据
        """
        try:

            actions,aim_landmark = read_jsonl(file_path + '/pose.jsonl')
            imgs_path, img_id = get_png_images(file_path)
            indices = find_consecutive_turns(actions)

            end_landmark = self.get_aim_landmark(imgs_path[-3:],aim_landmark)

            
            return end_landmark, actions, indices, imgs_path, img_id
        except Exception as e:
            logging.error(f"Error in process: {e}")
            logging.error(f"file: {file_path}")
            return None, None, None, None, None
    def get_token_usage(self):
        """
        获取当前 Token 使用量
        """
        return {'total':self.token_usage,'input':self.input_token,'output':self.output_token}