import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def main():
    # 1️⃣ 设置命令行参数
    parser = argparse.ArgumentParser(description="绘制Gate Entropy柱状图")
    parser.add_argument("--file", type=str, required=True, help="输入的JSONL文件路径")
    args = parser.parse_args()

    input_file = args.file

    # 2️⃣ 生成输出文件名（将输入文件的扩展名从.jsonl替换为.png）
    output_file = os.path.splitext(input_file)[0] + ".png"

    # 3️⃣ 读取 JSONL 文件
    records = [json.loads(line) for line in open(input_file, "r", encoding="utf-8") if line.strip()]

    # 4️⃣ 转成 DataFrame
    df = pd.DataFrame(records)

    # 5️⃣ 计算每个组的平均 entropy
    entropy_df = df.groupby("group", as_index=False)["entropy"].mean()

    # 6️⃣ 固定排序：Hot → Mid → Cold → Ultra-Cold
    order = ["[Hot]", "[Mid]", "[Cold]", "[Ultra-Cold]"]
    entropy_df["group"] = pd.Categorical(entropy_df["group"], categories=order, ordered=True)
    entropy_df = entropy_df.sort_values("group")

    # 7️⃣ 绘制柱状图
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=entropy_df, x="group", y="entropy", order=order, palette="Blues_d")

    plt.title("Gate Entropy for Different Popularity Items", fontsize=14)
    plt.xlabel("Item Group", fontsize=12)
    plt.ylabel("Gate Entropy", fontsize=12)

    # 8️⃣ 在柱子上方添加平均值标签（保留 4 位小数）
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 0.01,  # 标签位置稍微高于柱顶
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存为: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
