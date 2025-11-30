import os
import sys
import numpy as np
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from TrainAndTests.MARWIL.pure_marwil_train_attack import *
from Algorithms.PPOcontinues_std_no_state import *
from Algorithms.PPOcontinues_std_no_state import PPOContinuous

student_agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

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

    # 把 states/actions 转为 numpy array（若元素形状一致）
    try:
        states_arr = np.asarray(il_transition_dict['states'])
        actions_arr = np.asarray(il_transition_dict['actions'])
        print("Converted to numpy arrays:", states_arr.shape, actions_arr.shape)
    except Exception:
        print("States/actions could not be converted to regular numpy arrays (mixed shapes/objects).")

    # --- 新增：预处理 transitions，保证可以转换为 torch.tensor ---
    def preprocess_transitions(trans):
        """把 'states','actions','returns' 转成数值 numpy arrays（float32）。
           - 若元素形状一致则 stack。
           - 否则展平并 pad 到相同长度（右侧补零）。
        """
        for key in ('states','actions','returns'):
            items = trans.get(key, [])
            if len(items) == 0:
                trans[key] = np.array([], dtype=np.float32)
                continue
            arr = np.asarray(items)
            if arr.dtype != object:
                trans[key] = arr.astype(np.float32)
                continue
            nds = [np.asarray(x) for x in items]
            shapes = [n.shape for n in nds]
            if all(s == shapes[0] for s in shapes):
                # 形状一致 -> 直接 stack
                trans[key] = np.stack([n.astype(np.float32) for n in nds])
            else:
                # 形状不一致 -> 展平成 1D 并 pad
                flats = [n.ravel().astype(np.float32) for n in nds]
                maxlen = max(len(f) for f in flats)
                padded = np.zeros((len(flats), maxlen), dtype=np.float32)
                for i, f in enumerate(flats):
                    padded[i, :len(f)] = f
                trans[key] = padded
        return trans

    il_transition_dict = preprocess_transitions(il_transition_dict)
    print("Preprocessed shapes:", {k: np.asarray(il_transition_dict[k]).shape for k in ('states','actions','returns')})

    # 第4步：有监督学习teacher动作
    logs_dir = os.path.join(project_root, "logs/attack")
    mission_name = 'MARWIL'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    for epoch in range(1000):
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(il_transition_dict)
        # 记录损失函数
        if epoch % 10 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)

            print("epoch", epoch)


