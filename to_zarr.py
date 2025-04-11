from pathlib import Path

import cv2
import json
import torch
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

base_path = Path('/home/swx/shared_data_link/jaka_s5/blue_can-20241127-21')
store_path = '/home/swx/diffusion_policy/data/jaka_s5/blue_can-20241127-21.zarr'
fps = 30
task = 'Grab a blue can'

episodes_dir = base_path / 'episodes'
videos_dir = base_path / 'videos'

rec_info_path = episodes_dir / 'recording_info.json'
with open(rec_info_path) as f:
    rec_info = json.load(f)
episode_index = rec_info['last_episode_index'] + 1

num_episodes = episode_index


storage = zarr.DirectoryStore(store_path)
root = zarr.group(store=storage, overwrite=True)

replay_buffer = ReplayBuffer.create_empty_zarr()



for i in range(num_episodes):
    episode_path = episodes_dir / f'episode_{i:06d}.pth'
    if not episode_path.exists():
        episode_path = episodes_dir / f'episode_{i}.pth'

    episode_data = torch.load(episode_path)
    actions = episode_data['action']
    states = episode_data['observation.robot.jaka']

    actions = actions.numpy()
    states = states.numpy()
    cam_front_dir = videos_dir / f'observation.image.cam_front_episode_{i:06d}'
    cam_wrist_dir = videos_dir / f'observation.image.cam_wrist_episode_{i:06d}'

    cam_front_array = []
    cam_wrist_array = []

    for frame_index in range(len(actions)):
        action = actions[frame_index]
        state = states[frame_index]

        cam_front_path = cam_front_dir / f'frame_{frame_index:06d}.png'
        cam_wrist_path = cam_wrist_dir / f'frame_{frame_index:06d}.png'
        cam_front_image = cv2.imread(str(cam_front_path))
        cam_wrist_image = cv2.imread(str(cam_wrist_path))
        cam_front_image = cv2.cvtColor(cam_front_image, cv2.COLOR_BGR2RGB)
        cam_wrist_image = cv2.cvtColor(cam_wrist_image, cv2.COLOR_BGR2RGB)
        cam_front_image = np.transpose(cam_front_image, (2, 0, 1))
        cam_wrist_image = np.transpose(cam_wrist_image, (2, 0, 1))

        cam_front_array.append(cam_front_image)
        cam_wrist_array.append(cam_wrist_image)

    cam_front_array = np.stack(cam_front_array)
    cam_wrist_array = np.stack(cam_wrist_array)

    data = {
        'cam_front_img': cam_front_array,
        'cam_wrist_img': cam_wrist_array,
        'state': states,
        'action': actions
    }

    replay_buffer.add_episode(data)

    print(f'Processed episode {i}')

root.attrs['fps'] = fps
root.attrs['task'] = task
replay_buffer.save_to_path(store_path)


        