import subprocess
import sys
import os
import itertools
import datetime


python_exe = sys.executable
# 你想要运行的脚本名称
script_name = "test.py"

# 数据集
datasets = [
    "MNIST-USPS",
    "BDGP",
    "CCV",
    "Fashion",
    "Caltech-2V",
    "Caltech-3V",
    "Caltech-4V",
    "Caltech-5V",
    "Prokaryotic",
    "Synthetic3d",
    "Cifar10",
    "Cifar100"
    "NUS-WIDE",
    "Deep Animal"
]

lambda_cats = [0.1, 0.5, 1.0, 5]
# lambda_cats = [0.05]
# lambda_insts = [0.1, 1.0, 5.0, 10.0]
lambda_insts = [1]
lambda_cons = [1.0,0.7, 0.5, 0.1]
# lambda_cons = [0.1]

# 组合出所有可能的参数对
hyper_grid = list(itertools.product(lambda_cats, lambda_insts, lambda_cons))
total_runs = len(datasets) * len(hyper_grid)

print(f"准备运行 {len(datasets)} 个数据集...")
print(f"每个数据集有 {len(hyper_grid)} 组参数组合。")
print(f"总计需要运行 {total_runs} 次训练。\n")

# 创建一个专门记录调参结果的文件
log_file = "grid_search_log.txt"
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n{'=' * 60}\n")
    f.write(f"Grid Search Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'=' * 60}\n")

run_count = 0

for dataset in datasets:
    print(f"\n{'#' * 50}")
    print(f"🚀 开始调参数据集: {dataset}")
    print(f"{'#' * 50}\n")

    for cat, inst, con in hyper_grid:
        run_count += 1
        print(f"👉 [Progress: {run_count}/{total_runs}] Dataset: {dataset} | cat={cat}, inst={inst}, con={con}")

        # 组装带参数的命令行
        cmd = [
            python_exe, script_name,
            "--dataset", dataset,
            "--lambda_cat", str(cat),
            "--lambda_inst", str(inst),
            "--lambda_con", str(con)
        ]

        # 捕获输出，提取最后的结果
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            print(f"❌ 运行失败！跳过此参数组合。\n")
            # 记录失败日志
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[ERROR] Dataset: {dataset} | cat={cat}, inst={inst}, con={con} | Failed to run.\n")
            continue

        # 从输出中寻找最终分数的标志行
        best_acc_line = ""
        for line in result.stdout.split('\n'):
            if ">>> Finished! Final Best All Acc" in line:
                best_acc_line = line.strip()
                break

        if best_acc_line:
            print(f"✅ 完成！结果: {best_acc_line}\n")
            # 将结果写入调参日志
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Dataset: {dataset:<12} | cat={cat:<4} inst={inst:<4} con={con:<4} | {best_acc_line}\n")
        else:
            print(f"⚠️ 运行完成，但未找到 Best Acc 标识行。\n")

print("\n🎉 所有网格搜索任务已全部执行完毕！请查看 grid_search_log.txt 获取汇总结果。")