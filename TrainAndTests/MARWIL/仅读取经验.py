import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
import pickle

def load_il_and_transitions(folder=None):
    """从指定目录加载 il_transitions.pkl 和 transition_dict.pkl（若存在）。
       返回 (il_transition_dict, transition_dict)，不存在则返回 None 对应项。
    """
    if folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))
    il_path = os.path.join(folder, "il_transitions.pkl")
    trans_path = os.path.join(folder, "transition_dict.pkl")
    il = None
    trans = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
    if os.path.isfile(trans_path):
        with open(trans_path, "rb") as f:
            trans = pickle.load(f)
    return il, trans

if __name__ == "__main__":
    il, trans = load_il_and_transitions(current_dir)
    if il is None and trans is None:
        print("No il_transitions.pkl or transition_dict.pkl found in this directory.")
    else:
        if il is not None:
            print("Loaded il_transitions.pkl -> keys:", list(il.keys()))
        if trans is not None:
            print("Loaded transition_dict.pkl -> keys:", list(trans.keys()))