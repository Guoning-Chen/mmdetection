from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import threading
import torch
import psutil
import time
import pynvml
import mmcv
import numpy as np


ready = False


def get_memory_used():
    """获取当前内存的已使用量（MB）"""
    return float(psutil.virtual_memory().used) / 1024 / 1024  # MB


def get_video_memory_used():
    """获取当前显存的已使用量（MB）"""
    return pynvml.nvmlDeviceGetMemoryInfo(
        pynvml.nvmlDeviceGetHandleByIndex(0)).used / 1024 / 1024  # MB


def infer_time(times=10, remove_head=False):
    """
    测试 CPU/GPU 进行单张推理的时间，重复 times 次后取平均值。
    Args:
        times: 一共运行的次数。
        remove_head: 为 True 时去掉第一个值（使用 GPU 时第一次推理时间异常的长）

    Returns:

    """
    # 加载模型和输入
    args = InferParams()
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 运行 times 次取平均值
    record = []
    for _ in range(times):
        start = time.time()
        # ========= 单次运行 =========
        img = mmcv.imread(args.img)
        img = np.asarray([img]*4)
        _ = inference_detector(model, img)
        # ===========================
        end = time.time()
        record.append(1000*(end - start))
    if remove_head:
        result = sum(record[1:]) / (times - 1)
    else:
        result = sum(record) / times
    print('%d 次推理的时间: ' % times, record)
    print('平均值为: %.2f ms' % result)


class InferParams:
    def __init__(self):
        self.img = 'demo.jpg'
        # self.config = 'C:/Users/chengn/Documents/Projects/mmdetection/work_dirs/r50_fpn/r50_fpn.py'
        # self.checkpoint = 'C:/Users/chengn/Documents/Projects/mmdetection/work_dirs/r50_fpn/epoch_12.pth'
        self.config = 'C:/Users/chengn/Documents/Projects/mmdetection/work_dirs/r50pf_fpn/r50pf_fpn.py'
        self.checkpoint = 'C:/Users/chengn/Documents/Projects/mmdetection/work_dirs/r50pf_fpn/epoch_30.pth'
        self.device = 'cuda:0'  # 'cpu'，'cuda:0'
        self.score_thr = 0.3


class MyThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        pynvml.nvmlInit()
        while not ready:
            # 加载模型/推理前的使用情况
            mem_before = get_memory_used()  # 内存
            video_mem_before = get_video_memory_used()  # 显存
        print("[before]\n内存: %d MB, 显存: %d MB" % (
            mem_before, video_mem_before))

        # 记录最大使用量
        mem_max = 0.
        video_mem_max = 0.
        while ready:
            mem_max = max(mem_max, get_memory_used())  # 内存
            video_mem_max = max(video_mem_max, get_video_memory_used())  # 显存

        # 输出差值
        print("[after]\n内存: %d MB, 显存: %d MB" % (
            mem_max, video_mem_max))
        print("[occupation]\n内存: %d MB, 显存: %d MB" % (
            mem_max - mem_before, video_mem_max - video_mem_before))


# child_thread = MyThread(threadID=1, name="child", counter=1)
# child_thread.start()
#
# time.sleep(1)
# ready = True
# args = InferParams()
# # ========== 执行过程 ============
# model = init_detector(args.config, args.checkpoint, device=args.device)
# result = inference_detector(model, args.img)
# # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
# # ===============================
# ready = False

infer_time(times=11, remove_head=True)


"""
【仅加载模型】 
剪枝前：175, 170, 188, 185, 188
剪枝后: 88, 121, 109, 124, 123
【CPU单张内存】
剪枝前：518，516，517，520，513
剪枝后：311，316，317，310，316
【GPU单张内存 & 显存】
剪枝前内存：1938, 1938, 1938, 1938，1938
剪枝前显存：1513，1513，1513，1513,1513
剪枝后内存：
剪枝后显存：
【GPU单张推理时间】
剪枝前：90.48
剪枝后：67.56
【CPU单张推理时间】
剪枝前：2104.71
剪枝后：1720.29
"""
