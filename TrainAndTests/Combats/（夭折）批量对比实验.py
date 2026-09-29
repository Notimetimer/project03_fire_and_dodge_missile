'''
阻塞等待

import subprocess
import sys
# 启动并等待 worker.py 运行结束
result = subprocess.run([sys.executable, "worker.py"])
# 通过 returncode 判断是否成功运行完毕（0 通常表示正常退出）
if result.returncode == 0:
    print("子脚本运行成功完毕！")
else:
    print(f"子脚本异常退出，错误码: {result.returncode}")

非阻塞等待

import subprocess
import sys
import time

# 非阻塞拉起子脚本
proc = subprocess.Popen([sys.executable, "worker.py"])

print("子脚本已启动，主脚本继续执行其他任务...")

# 轮询监控状态
while True:
    ret_code = proc.poll()
    if ret_code is None:
        print("子脚本还在运行中...")
        time.sleep(1)
    else:
        print(f"子脚本已运行完毕，退出码为: {ret_code}")
        break
'''

import subprocess
import sys

# 启动并等待 worker.py 运行结束
result = subprocess.run([sys.executable, "worker.py"])

# 通过 returncode 判断是否成功运行完毕（0 通常表示正常退出）
if result.returncode == 0:
    print("子脚本运行成功完毕！")
else:
    print(f"子脚本异常退出，错误码: {result.returncode}")
    
    
    