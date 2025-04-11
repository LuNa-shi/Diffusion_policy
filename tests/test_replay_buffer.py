import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer

def test():
    import numpy as np
    buff = ReplayBuffer.create_empty_numpy()
    buff.add_episode({
        'obs': np.zeros((100,10), dtype=np.float16)
    })
    buff.add_episode({
        'obs': np.ones((50,10)),
        'action': np.ones((50,2))
    })
    # buff.rechunk(256)
    obs = buff.get_episode(0)

    import numpy as np
    buff = ReplayBuffer.create_empty_zarr()
    buff.add_episode({
        'obs': np.zeros((100,10), dtype=np.float16)
    })
    buff.add_episode({
        'obs': np.ones((50,10)),
        'action': np.ones((50,2))
    })
    obs = buff.get_episode(0)
    buff.set_chunks({
        'obs': (100,10),
        'action': (100,2)
    })


def test_real():
    # import os
    # dist_group = zarr.open(
    #     os.path.expanduser('/home/swx/diffusion_policy/data/jaka_s5/blue_can-20241127-21.zarr'), 'r')

    # buff = ReplayBuffer.create_empty_numpy()
    # key, group = next(iter(dist_group.items()))
    # for key, group in dist_group.items():
    #     buff.add_episode(group)

    # out_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht_cchi2_v2_replay.zarr')
    out_path = os.path.expanduser('/home/swx/diffusion_policy/data/jaka_s5/blue_can-20241127-21.zarr')
    # out_store = zarr.DirectoryStore(out_path)
    # buff.save_to_store(out_store)

    buff = ReplayBuffer.copy_from_path(out_path, store=zarr.MemoryStore())
    episode = buff.get_episode(50)
    for key, value in episode.items():
        print(key, value.shape)
    print(buff.n_episodes)


def test_pop():
    buff = ReplayBuffer.create_from_path(
        "/home/swx/diffusion_policy/data/jaka_s5/blue_can-20241127-21.zarr",
        mode='rw')
    episode = buff.pop_episode()
    for key, value in episode.items():
        print(key, value.shape)

if __name__ == '__main__':
    test_real()
    print('done')

