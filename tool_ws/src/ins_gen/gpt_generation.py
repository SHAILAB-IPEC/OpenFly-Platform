import os
import json
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
from typing import List, Dict
from process import process_act
from gpt import OpenAIClient  



file_lock = asyncio.Lock()  
def get_subdirectories_with_paths(folder_path):
    return [
        (index, name, os.path.join(folder_path, name)) 
        for index, name in enumerate(os.listdir(folder_path)) 
        if os.path.isdir(os.path.join(folder_path, name))
    ]

def get_subfolders(folder_path):
    return [
        os.path.join(folder_path, name) 
        for name in os.listdir(folder_path) 
        if os.path.isdir(os.path.join(folder_path, name))
    ]

async def retry_async(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise e

async def get_data(end_landmark, actions, indices, imgs_path, client: OpenAIClient):
    if len(indices) == 1:
        instructions = await retry_async(client.get_instruction, client.get_action('go straight', end_landmark))
    else:
        actions_list = []
        for i, index in enumerate(indices):
            if i == 0:
                landmark = await retry_async(client.get_landmark, imgs_path[index+1])
                act = actions[index]['action']['type']
                actions_list.extend(client.get_action(act, landmark))
            else:
                if  int(indices[i-1]) == int(indices[i]) -1:
                    act = actions[index]['action']['type']
                    actions_list.extend(f'slightly {act}')
                else:
                    first_landmark = await retry_async(client.get_landmark, imgs_path[index+1])
                    act = actions[index]['action']['type']
                    actions_list.extend(client.get_action(act, first_landmark))
        act = actions[-2]['action']['type']
        actions_list.extend(client.get_action(act, end_landmark))
        instructions = await retry_async(client.get_instruction, actions_list)
    return instructions

async def gpt(folder_path, client: OpenAIClient):
    end_landmark, actions, indices, imgs_path, img_id = await retry_async(client.process, folder_path)
    if end_landmark is None:
        return None

    instructions = await get_data(end_landmark, actions, indices, imgs_path, client)

    data_actions = process_act(actions)
    pos = [item['action']['pos'] for item in actions]
    yaw = [item['action']['yaw'] for item in actions]
    return instructions, data_actions, pos, yaw, img_id

async def process_folder(folder, pool: 'OpenAIPool',base_folder):
    try:
        client = await pool.get_client()
        instructions, data_actions, pos, yaw, img_id = await gpt(folder, client)
        return ('success', {
            'image_path': os.path.relpath(folder, base_folder),
            'gpt_instruction': instructions,
            'action': data_actions,
            'index_list': img_id,
            'pos': pos,
            'yaw': yaw,
        })
    except Exception as e:
        return ('fail', {'path': folder, 'error': str(e)})

class OpenAIPool:
    def __init__(self, configs: List[Dict]):
        self.clients = []
        for conf in configs:
            client = OpenAIClient(
                api_key=conf['key'],
                model=conf['model']
            )
            self.clients.append(client)
        self.index = 0
        self.lock = asyncio.Lock()

    async def get_client(self) -> OpenAIClient:
        async with self.lock:
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client

    def get_tokens(self):
        token_usage = 0  
        input_token = 0
        output_token = 0
        for client in self.clients:
            tokens = client.get_token_usage()
            token_usage += tokens['total']
            input_token += tokens['input']
            output_token += tokens['output']
        return {'total':token_usage,'input':input_token,'output':output_token}
    

    def __init__(self, configs: List[Dict]):
        self.clients = []
        for conf in configs:
            client = OpenAIClient(
                api_key=conf["key"],
                model=conf["model"]
            )
            self.clients.append(client)
        self.index = 0
        self.lock = asyncio.Lock()

    async def get_client(self) -> OpenAIClient:
        async with self.lock:
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client
    def get_tokens(self):
        token_usage = 0  # 记录 Token 使用量
        input_token = 0
        output_token = 0
        for client in self.clients:
            tokens = client.get_token_usage()
            token_usage += tokens['total']
            input_token += tokens['input']
            output_token += tokens['output']
        return {'total':token_usage,'input':input_token,'output':output_token}
        
async def sem_task(task, semaphore):
    async with semaphore:
        return await task



async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Instruction Generation")
    parser.add_argument('-j', '--json', type=str, required=True, help="json_path")
    parser.add_argument('-n', '--type', type=str, required=True, help="data_name")
    args = parser.parse_args()

    json_path = args.json


    with open("tool_ws/src/ins_gen/gpt_api_config.json", "r") as config_file:
        api_configs = json.load(config_file)

    pool = OpenAIPool(api_configs)



    with open(json_path,'r')as f:
        folders = json.load(f)

    
    traj_folder = "Your Trajectory Folders PATH"
    data_path = f"tool_ws/src/ins_gen/instructions/{args.type}.json"


    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    semaphore = asyncio.Semaphore(10)  # Limit the number of concurrent connections, usually set to twice the number of APIs
    
    tasks = [
        sem_task(process_folder(folder, pool, traj_folder), semaphore)
        for folder in folders
    ]
    results = []
    count = 0  

    async with aiofiles.open(data_path, 'w', encoding='utf-8') as file:
        await file.write('[')  
        first = True  

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing Folders:"):
            status, data = await future
            if status == 'success':
                results.append(data)
                count += 1

            if count >= 10:
                if not first:
                    await file.write(',') 
                await file.write(','.join(json.dumps(result, indent=4, ensure_ascii=False) for result in results))
                results.clear()  
                count = 0  
                first = False

        if results:
            if not first:
                await file.write(',')  
            await file.write(','.join(json.dumps(result, indent=4, ensure_ascii=False) for result in results))

        await file.write(']')
    
    token_usage = 0
    input_token = 0
    output_token = 0
    for tokens in [pool.get_tokens()]:
        token_usage += tokens['total']
        input_token += tokens['input']
        output_token += tokens['output']
        
    print('Used token:',{'total':token_usage,'input':input_token,'output':output_token})


if __name__ == "__main__":
    asyncio.run(main())