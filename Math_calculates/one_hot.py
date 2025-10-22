# import numpy as np

# def labels_to_one_hot(labels, n=None):
#     """
#     将 1,2,...,n 风格的标签转为 one-hot numpy 数组。
#     labels: array-like of ints (1-based)
#     n: 可选的类别数（若 None 则取 labels.max()）
#     返回 shape (len(labels), n) 的 int 数组
#     """
#     labels = np.asarray(labels).ravel()
#     if labels.size == 0:
#         return np.zeros((0, 0), dtype=int) if n is None else np.zeros((0, int(n)), dtype=int)
#     if n is None:
#         n = int(labels.max())
#     n = int(n)
#     if labels.min() < 1 or labels.max() > n:
#         raise ValueError(f"labels must be in [1, {n}]")
#     one_hot = np.zeros((labels.size, n), dtype=int)
#     idx = labels.astype(int) - 1
#     one_hot[np.arange(labels.size), idx] = 1
#     return one_hot

# def one_hot_to_labels(one_hot):
#     """
#     将 one-hot numpy 数组（shape (m, n) 或 (n,)）转回 1..n 标签（返回 1D int ndarray）。
#     若输入是一维向量，仍返回 1D 数组（length 1）。
#     """
#     arr = np.asarray(one_hot)
#     if arr.ndim == 1:
#         arr = arr.reshape(1, -1)
#     if arr.size == 0:
#         return np.array([], dtype=int)
#     # 取每行最大值的索引并转为 1-based label
#     labels = arr.argmax(axis=1) + 1
#     return labels.astype(int)

# if __name__ == "__main__":
#     # 简单示例
#     labels = [1, 2, 3, 1, 2]
#     oh = labels_to_one_hot(labels)
#     print("labels -> one-hot:\n", oh)
#     recovered = one_hot_to_labels(oh)
#     print("one-hot -> labels:\n", recovered)


import numpy as np

def labels_to_one_hot(labels, n, min):
    one_hot = np.zeros(n)
    idx = int(labels) - min
    one_hot[idx] = 1
    return one_hot

def one_hot_to_labels(one_hot, min):
    arr = np.asarray(one_hot)
    arr = arr.reshape(1, -1)
    labels = np.argmax(one_hot) + min
    return labels

if __name__ == "__main__":
    # 简单示例
    min = 0
    n = 4
    labels = [1, 2, 3]
    for i in labels:
        oh = labels_to_one_hot(i, n, min)
        print("labels -> one-hot:\n", oh)
        recovered = one_hot_to_labels(oh, min)
        print("one-hot -> labels:\n", recovered)


