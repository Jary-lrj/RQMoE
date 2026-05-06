import json
from collections import defaultdict


def sum_expert_counts_by_group(json_file_path):
    """
    根据group字段统计expert_counts的总和

    参数:
    json_file_path: JSON文件路径

    返回:
    group_sums: 字典，key为group，value为对应group的expert_counts总和列表
    group_counts: 字典，key为group，value为对应group的记录数量
    """
    # 初始化字典来存储每个group的expert_counts总和
    group_sums = defaultdict(lambda: [0, 0, 0, 0])  # 假设有4个专家
    group_counts = defaultdict(int)

    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    group = data.get("group", "")
                    expert_counts = data.get("expert_counts", [])

                    if group and expert_counts:
                        # 将当前记录的expert_counts加到对应group的总和中
                        for i, count in enumerate(expert_counts):
                            if i < len(group_sums[group]):
                                group_sums[group][i] += count

                        group_counts[group] += 1

                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue

    except FileNotFoundError:
        print(f"文件未找到: {json_file_path}")
        return {}, {}
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return {}, {}

    return dict(group_sums), dict(group_counts)


def print_group_statistics(group_sums, group_counts):
    """
    打印分组统计结果
    """
    print("=== 各Group的Expert Counts统计 ===")
    for group, sums in group_sums.items():
        count = group_counts.get(group, 0)
        print(f"\nGroup: {group}")
        print(f"记录数量: {count}")
        print(f"Expert Counts总和: {sums}")
        print(f"Expert Counts平均值: {[round(x/count, 2) if count > 0 else 0 for x in sums]}")
        print(f"总分配次数: {sum(sums)}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file_path = "gate_stats_test_lb=0.01.jsonl"

    # 计算各group的expert_counts总和
    group_sums, group_counts = sum_expert_counts_by_group(json_file_path)

    # 打印结果
    print_group_statistics(group_sums, group_counts)

    # 可选：保存结果到文件
    with open("group_statistics.json", "w", encoding="utf-8") as f:
        result = {"group_sums": group_sums, "group_counts": group_counts}
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n结果已保存到 group_statistics.json")
