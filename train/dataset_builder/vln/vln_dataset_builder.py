"""bridge_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from pathlib import Path
import glob
import os
from PIL import Image
import numpy as np
import json
import random


# 记录起始帧、拐点帧以及当前帧

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for trajectory dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset of trajectories where each step has the following fields: steps, episode_metadata.",
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    'history': tfds.features.Text(),
                    'is_terminal': tfds.features.Scalar(dtype=tf.bool),
                    'is_last': tfds.features.Scalar(dtype=tf.bool),
                    'language_instruction': tfds.features.Text(),
                    'observation': {
                        'image_1': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                        'image_2': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                        'image_3': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                    },
                    'is_first': tfds.features.Scalar(dtype=tf.bool),
                    'discount': tfds.features.Scalar(dtype=tf.float32),
                    'reward': tfds.features.Scalar(dtype=tf.float32),
                }),
                'episode_metadata': {
                    'has_image_2': tfds.features.Scalar(dtype=tf.bool),
                    'has_image_3': tfds.features.Scalar(dtype=tf.bool),
                    'file_path': tfds.features.Text(),
                    'has_language': tfds.features.Scalar(dtype=tf.bool),
                    'has_image_1': tfds.features.Scalar(dtype=tf.bool),
                    'has_image_0': tfds.features.Scalar(dtype=tf.bool),
                    'episode_id': tfds.features.Scalar(dtype=tf.int32),
                },
            }),
            supervised_keys=None,
            homepage='https://dataset-homepage/',
            citation=r"""@misc{vln_2024, title={VLN Dataset}, year={2024}}""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # path = dl_manager.download_and_extract('https://todo-data-url')
        json_path = "YOUR_JSONDATA_PATH"
        return {
            'train': self._generate_examples(json_path),
        }

    def _generate_examples(self, json_path):
        """Yields examples."""
        f = open(json_path, 'r')
        js = json.load(f)['episodes']
        data_len = len(js)  
                
        action_dict = {
            "0":np.array([1,0,0,0,0,0,0,0]).astype(np.float32), # stop
            "1":np.array([0,3,0,0,0,0,0,0]).astype(np.float32), # move forward
            "2":np.array([0,0,15,0,0,0,0,0]).astype(np.float32), #turn left
            "3":np.array([0,0,0,15,0,0,0,0]).astype(np.float32), # turn right
            "4":np.array([0,0,0,0,2,0,0,0]).astype(np.float32), # go up
            "5":np.array([0,0,0,0,0,2,0,0]).astype(np.float32), # go down
            "6":np.array([0,0,0,0,0,0,5,0]).astype(np.float32), # move left
            "7":np.array([0,0,0,0,0,0,0,5]).astype(np.float32), # move right
            "8":np.array([0,6,0,0,0,0,0,0]).astype(np.float32), # move forward 
            "9":np.array([0,9,0,0,0,0,0,0]).astype(np.float32), # move forward 
        }
                             
        
        def history_recorder(action_list): 
            if not action_list:
                return ""
            # Create a summary of actions with reduced consecutive duplicates
            summary = [action_list[0]]
            for action in action_list[1:]:
                if action != summary[-1]:
                    summary.append(action)

            # Join the actions into a history string
            history = ' then '.join(summary)
            return history
        
        exp_name = "vln_norm"
        for episode_id in range(data_len // 8, data_len // 8 * 2):
            img_path = "YOUR_IMAGE_PATH"
            if not os.path.exists(f"{img_path}/{episode_id}"):
                continue
            historys = [" "]
            data_dict = js[episode_id]
            if len(data_dict["actions"]) > 400:
                continue
            actions = [action_dict[str(x)] for x in data_dict["actions"][:-2]] + [action_dict["0"]] * 2
            instruction = data_dict['instruction']['instruction_text']

            image_array = []
            try:
                for idx in range(len(data_dict["actions"])-2):
                    image_array.append(np.array(Image.open(img_path + "/" + str(episode_id) + "/" + str(idx) + ".png"), dtype=np.uint8))

                image_array.append(np.array(Image.open(img_path + "/" + str(episode_id) + "/" + str(len(data_dict["actions"])-2) + ".png"), dtype=np.uint8))
                image_array.append(np.array(Image.open(img_path + "/" + str(episode_id) + "/" + str(len(data_dict["actions"])-2) + ".png"), dtype=np.uint8))
            except:
                continue
            
            total_steps = len(actions)
            steps = []
            
            actions_mapped = [action_map[str(item)] for item in data_dict["actions"][:-2]]
            for index in range(1, total_steps):
                historys.append(history_recorder(actions_mapped[:index]))
            historys.append(historys[-1])
            historys.append(historys[-1])            
            
            for idx in range(total_steps):
                image_1 = image_array[idx]  # 当前帧
                image_4 = image_array[0]
#                 if idx < 4:
#                     image_3 = image_4 = image_5 = image_2 = image_array[idx-2]
#                 elif idx < 6:
#                     image_2 = image_array[idx-2]
#                     image_3 = image_4 = image_5 = image_array[idx-4]
#                 elif idx < 8:
#                     image_2 = image_array[idx-2]
#                     image_3 = image_array[idx-4]
#                     image_4 = image_5 = image_array[idx-6]
#                 else:
#                     image_2 = image_array[idx-2]
#                     image_3 = image_array[idx-4]
#                     image_4 = image_array[idx-6]
#                     image_5 = image_array[idx-8]
                keypoint = 0
                try:
                    keypoint = next(i for i in range(1, len(data_dict["actions"])) if data_dict["actions"][i] != data_dict["actions"][i-1])
                except:
                    keypoint = 0
    
                if idx == 0:
                    image_2 = image_3 = image_array[0]
                elif idx == 1: 
                    image_2 = image_array[-1]
                    image_3 = image_array[-2]
                    # image_4 = image_array[-2]
                elif keypoint == idx - 1:
                    image_2 = image_array[keypoint]
                    image_3 = image_array[idx-2]
                elif keypoint != 0:
                    image_2 = image_array[idx-1]
                    image_3 = image_array[keypoint]
                else:
                    image_2 = image_array[-2]
                    image_3 = image_array[-3]

                steps.append(
                    {
                      'action': actions[idx],
                      'history': historys[idx],
                      'is_terminal': False if idx < total_steps - 2 else True,
                      'is_last': False if idx < total_steps - 2 else True,
                      'language_instruction': instruction,
                      'observation': {
                          'image_1': image_1,
                          'image_2': image_2,
                          'image_3': image_3,
                      },
                      'is_first': True if idx == 0 else False,
                      'discount': 1.0,
                      'reward': 0.0,
                    }
                )
            
            yield f"{exp_name}/{episode_id}" , {
              'steps': steps,
              'episode_metadata': {
                  'has_image_2': True,
                  'has_image_3': True,
                  'file_path': img_path,
                  'has_language': True,
                  'has_image_1': True,
                  'has_image_0': False,
                  'episode_id': episode_id,
              },
          }

