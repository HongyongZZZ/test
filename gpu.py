#!/usr/bin/env python3

import sys
import subprocess
import time
import argparse
from datetime import datetime


def get_gpu_info():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,nounits,noheader",
            ]
        ).decode("utf-8")
        gpu_lines = result.strip().split("\n")

        gpu_info = {}
        for line in gpu_lines:
            gpu_id, total_memory, used_memory, free_memory, gpu_utilization = (
                line.split(",")
            )
            gpu_info[int(gpu_id)] = {
                "total_memory": int(total_memory),
                "used_memory": int(used_memory),
                "free_memory": int(free_memory),
                "utilization": int(gpu_utilization),
            }

        return gpu_info
    except Exception as e:
        print(f"Error while getting GPU info: {e}", file=sys.stderr)
        return {}


def wait_for_gpus(min_gpu_count, min_free_memory, available_gpu_ids, scan_interval):
    while True:
        gpu_info = get_gpu_info()
        available_gpus = []
        outputs = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
        for gpu_id in available_gpu_ids:
            if gpu_id in gpu_info:
                gpu = gpu_info[gpu_id]
                if gpu["free_memory"] >= min_free_memory:
                    available_gpus.append({"gpu_id": gpu_id} | gpu)
                    outputs += f"{gpu_id}: ✅ "
                else:
                    outputs += f"{gpu_id}: {(gpu['free_memory'] /1024):.2f}GB "
        if len(available_gpus) >= min_gpu_count:
            available_gpus = sorted(
                available_gpus, key=lambda x: x["free_memory"], reverse=True
            )
            return [gpu["gpu_id"] for gpu in available_gpus[:min_gpu_count]]

        print(outputs, file=sys.stderr)
        time.sleep(scan_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wait for GPUs to meet specific conditions."
    )
    parser.add_argument(
        "-n",
        "--min_gpu_count",
        type=int,
        default=1,
        help="Minimum number of GPUs required.",
    )
    parser.add_argument(
        "-mem",
        "--min_free_memory",
        type=float,
        default=None,
        help="Minimum free memory per GPU in GB.",
    )
    parser.add_argument(
        "-ids",
        "--available_gpu_ids",
        type=int,
        nargs="+",
        default=[],
        help="List of available GPU IDs.",
    )
    parser.add_argument(
        "-t", "--scan_interval", type=int, default=600, help="Scan interval in seconds."
    )

    args = parser.parse_args()

    gpu_info = get_gpu_info()
    if not args.available_gpu_ids:
        args.available_gpu_ids = list(gpu_info.keys())

    if args.min_free_memory is None:
        args.min_free_memory = max(
            [i["total_memory"] / 1024 for i in gpu_info.values()]
        )
        delta = 1
        args.min_free_memory -= delta
        # 默认为最大值 - delta，好像驱动会占一些显存
        args.min_free_memory = max(args.min_free_memory, delta)
    args.min_free_memory = int(args.min_free_memory * 1024)

    available_gpus = wait_for_gpus(
        args.min_gpu_count,
        args.min_free_memory,
        args.available_gpu_ids,
        args.scan_interval,
    )
    print(",".join(map(str, available_gpus)))


"""
安装代码：(安装完不要删除源文件)
    将本文件命名为GPU.py
    echo -e "\nfunction GPU() { python $(realpath GPU.py) \"\$@\"; }" >> ~/.bashrc
    source ~/.bashrc

选项:
    -n:  需要多少GPU(默认1)
    -mem: 每个GPU至少剩余多少GB显存(默认为所有GPU中显存最大值-1GB)
    -ids: 可用的GPU id 列表(用空格分隔，默认为全部)
    -t: 多久扫描一次GPU状态(默认600s)

例子:
    CUDA_VISIBLE_DEVICES=$(GPU -n 2 -mem 70 -ids 3 2 -t 600) accelerate launch train.py
    需要两个GPU，每个70GB，只看2、3号GPU，每600秒扫描一次
    CUDA_VISIBLE_DEVICES=$(GPU -n 2) accelerate launch train.py
    需要两个接近全空的GPU
"""
