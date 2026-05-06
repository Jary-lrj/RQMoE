#!/usr/bin/env python3
"""
计算JSONL文件中所有expert_counts的累加熵

功能：
1. 读取指定JSONL文件
2. 累加所有行的expert_counts
3. 计算累加后的expert_counts的熵
"""

import json
import argparse
import numpy as np
from pathlib import Path


def calculate_entropy(counts):
    """
    计算expert_counts的熵

    Args:
        counts: expert计数列表

    Returns:
        熵值（以2为底的对数）
    """
    counts = np.array(counts, dtype=float)
    total = np.sum(counts)

    if total == 0:
        return 0.0

    # 计算概率
    probabilities = counts / total

    # 避免log(0)的情况
    probabilities = probabilities[probabilities > 0]

    # 计算熵: H = -Σ(p_i * log2(p_i))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))

    return entropy


def process_jsonl_file(file_path):
    """
    处理JSONL文件，累加所有expert_counts并计算熵

    Args:
        file_path: JSONL文件路径

    Returns:
        tuple: (累加后的expert_counts列表, 熵值)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 初始化累加数组
    total_counts = None

    # 读取并处理每一行
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                expert_counts = data.get('expert_counts', [])

                if not expert_counts:
                    print(f"警告: 第 {line_num} 行没有 expert_counts 字段")
                    continue

                # 初始化或累加
                if total_counts is None:
                    total_counts = [0] * len(expert_counts)

                # 确保长度一致
                if len(expert_counts) != len(total_counts):
                    print(f"警告: 第 {line_num} 行的 expert_counts 长度不一致 "
                          f"(期望 {len(total_counts)}, 实际 {len(expert_counts)})")
                    continue

                # 累加
                for i, count in enumerate(expert_counts):
                    total_counts[i] += count

            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                continue

    if total_counts is None:
        raise ValueError("文件中没有找到有效的 expert_counts 数据")

    # 计算熵
    entropy = calculate_entropy(total_counts)

    return total_counts, entropy


def main():
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中所有expert_counts的累加熵'
    )
    parser.add_argument(
        'file',
        type=str,
        help='JSONL文件路径'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )

    args = parser.parse_args()

    try:
        total_counts, entropy = process_jsonl_file(args.file)

        print(f"\n文件: {args.file}")
        print(f"累加后的 expert_counts: {total_counts}")
        print(f"总计数: {sum(total_counts)}")
        print(f"熵值: {entropy:.6f}")

        if args.verbose:
            # 显示每个expert的概率
            total = sum(total_counts)
            probabilities = [c / total for c in total_counts]
            print(f"\n各expert的概率分布:")
            for i, (count, prob) in enumerate(zip(total_counts, probabilities)):
                print(f"  Expert {i}: 计数={count}, 概率={prob:.4f}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
