import os
import sys
import numpy as np
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from MARWIL.pure_marwil_train_attack import *

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        'states': list(data['states']),
        'actions': list(data['actions']),
        'returns': list(data['returns'])
    }

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def summarize(il_transition_dict):
    print("Loaded transitions:")
    for k in ('states','actions','returns'):
        v = il_transition_dict.get(k)
        print(f"  {k}: {None if v is None else len(v)} items")
    # sample few entries
    for i in range(min(3, len(il_transition_dict.get('states',[])) )):
        s = il_transition_dict['states'][i]
        a = il_transition_dict['actions'][i] if i < len(il_transition_dict['actions']) else None
        r = il_transition_dict['returns'][i] if i < len(il_transition_dict['returns']) else None
        print(f" sample {i}: state type={type(s)}, state.shape={getattr(np.asarray(s),'shape',None)}; "
              f"action type={type(a)}, returns type={type(r)}")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(cur_dir, "il_transitions.npz")
    pkl_path = os.path.join(cur_dir, "il_transitions.pkl")

    il_transition_dict = None
    if os.path.isfile(npz_path):
        try:
            il_transition_dict = load_npz(npz_path)
            print(f"Loaded npz: {npz_path}")
        except Exception as e:
            print(f"Failed to load npz ({npz_path}): {e}")

    if il_transition_dict is None and os.path.isfile(pkl_path):
        try:
            il_transition_dict = load_pkl(pkl_path)
            print(f"Loaded pkl: {pkl_path}")
        except Exception as e:
            print(f"Failed to load pkl ({pkl_path}): {e}")

    if il_transition_dict is None:
        print("No il_transitions file found in this directory.")
        sys.exit(1)

    summarize(il_transition_dict)

    # 示例：把 states/actions 转为 numpy array（若元素形状一致）
    try:
        states_arr = np.asarray(il_transition_dict['states'])
        actions_arr = np.asarray(il_transition_dict['actions'])
        print("Converted to numpy arrays:", states_arr.shape, actions_arr.shape)
    except Exception:
        print("States/actions could not be converted to regular numpy arrays (mixed shapes/objects).")

    # 第4步：有监督学习teacher动作
    for epoch in range(1000):
        agent.MARWIL_update(il_transition_dict, beta=)
        