import os
import sys
import json
from textwrap import indent


def load_summary(date_str):
    path = os.path.join('outputs', 'reports', f'EXP_{date_str}', 'summary.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pick_best(rows):
    if not rows:
        return None
    return sorted(rows, key=lambda r: (r.get('mAP', 0) or 0), reverse=True)[0]


def top_k_table(rows, k=10):
    rows_sorted = sorted(rows, key=lambda r: (r.get('mAP', 0) or 0), reverse=True)[:k]
    lines = [
        '| exp_id | prompt_name | direction | rank1 | mAP | gate |',
        '| --- | --- | --- | ---: | ---: | --- |'
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r.get('exp_id','')} | {r.get('prompt_name','')} | {r.get('direction','')} | "
            f"{(r.get('rank1',0) or 0):.4f} | {(r.get('mAP',0) or 0):.6f} | {r.get('gate','')} |"
        )
    return '\n'.join(lines)


def generate_md(date_str, summary):
    rows = summary.get('rows', [])
    best = pick_best(rows)
    indir = os.path.join('outputs', 'reports', f'EXP_{date_str}')
    png_path = os.path.join(indir, 'comparison.png')

    total = len(rows)
    md = []
    md.append(f"# 项目实验封存与分析（{date_str}）\n")

    md.append("## 封存与数据筛选\n")
    md.append("- 数据来源：`outputs/experiments/EXP_{date_str}-*`\n")
    md.append("- 筛选规则：剔除 `API_ERROR`/`endpoint为空` 的异常运行；剔除非 `gold` 子集或 `mock` 数据集的 smoke 运行；保留有效运行用于统计。\n")
    md.append(f"- 有效实验数：{total}\n")

    md.append("## CSV 分类整理\n")
    md.append("- 分类维度：根据 `prompt_name` 映射到改进方向：三行结构、两行结构、颜色优先、结构化描述、仅标签、最小化Tokens、Tokens提示、通用描述。\n")
    md.append(f"- 汇总文件：`outputs/reports/EXP_{date_str}/summary.csv` 与 `summary.json`\n")

    md.append("## 数据可视化\n")
    md.append("- 横向比较图：`comparison.png`（按改进方向的 Rank-1 与 mAP 平均值）。\n")
    md.append(f"- 图表路径：`{png_path}`\n")

    md.append("## Top-10 实验（按 mAP）\n")
    md.append(top_k_table(rows))
    md.append("\n")

    if best:
        md.append("## 最强 Prompt 与分析\n")
        md.append(f"- 最强 Prompt：`{best.get('prompt_name','')}`（改进方向：{best.get('direction','')}）\n")
        md.append(f"- 指标：Rank-1 = {(best.get('rank1',0) or 0):.4f}，mAP = {(best.get('mAP',0) or 0):.6f}，gate = {best.get('gate','')}\n")
        md.append("- 原因分析：\n")
        direction = best.get('direction','')
        if direction == '三行结构':
            md.append("  - 三行结构通过将身份属性（Line1）、三条线索（Line2）与上下文约束（Line3）解耦，减少描述噪声并提升可比性。\n")
            md.append("  - 结构化键值对与名词短语有助于 `embed_text.py` 中的权重分配（颜色、服饰类型、配件、显著特征等），提高文本嵌入的判别性。\n")
            md.append("  - 上下文（视角、可见度、年龄段等）为检索阶段的过滤/重排提供额外信号，降低不相关匹配。\n")
        elif direction == 'Tokens提示':
            md.append("  - 离散属性Tokens（颜色、上衣/下装/鞋/包等）与 CLIP 文本嵌入的词级表示天然对齐，减少长句带来的噪声。\n")
            md.append("  - `embed_text.py` 的加权策略更易于在Tokens粒度生效（显著特征/配饰权重等），提升检索判别性。\n")
            md.append("  - 统一格式有助于API输出稳定，降低描述漂移对嵌入的一致性影响。\n")
        elif direction == '颜色优先':
            md.append("  - 强化颜色线索在Market1501这类数据集上较为有效，快速缩小候选集合。\n")
            md.append("  - 风险：颜色主导可能掩盖服饰类型/配件差异，易受光照/色偏影响。\n")
        elif direction == '两行结构':
            md.append("  - 简化结构降低API输出复杂度，同时保留核心身份线索，减少冗余信息。\n")
        else:
            md.append("  - 该改进方向在当前嵌入与数据分布下更契合（可能更少冗余/结构更稳定），从而获得更高mAP。\n")
        md.append("- 局限与现状：结果仍低于门槛（rank1 0.3 / mAP 0.4）。可能受 API 输出一致性、颜色/词表规范化、嵌入加权策略等影响。\n")

    md.append("## 未来思路与待解决问题\n")
    md.append("- Prompt 约束强化：\n")
    md.append("  - 固定键顺序、限定取值集合（颜色11色、服饰类型词表）、限制冗余与修饰语。\n")
    md.append("  - Line2 线索统一名词短语格式，避免混入形容词或动词。\n")
    md.append("- 嵌入改进：\n")
    md.append("  - 在 `embed_text.py` 中调优权重（如鞋/包/显著特征的比重），并考虑引入加权的颜色相似度。\n")
    md.append("  - 引入 re-ranking（`tools/retrieve_eval.py --re-ranking`）以提升检索质量。\n")
    md.append("- 数据一致性：\n")
    md.append("  - 统一颜色词（11色映射），合并同义词与大小写差异，消除轻微漂移。\n")
    md.append("  - 引入 schema 校验器，对API返回进行格式检查与自动修复。\n")
    md.append("- 评估策略：\n")
    md.append("  - 扩展子集规模、进行跨模型（`clip_l14` vs 其他后端）对比，以验证改进稳健性。\n")

    return '\n'.join(md)


def main(date_str: str):
    summary = load_summary(date_str)
    md = generate_md(date_str, summary)
    outdir = os.path.join('outputs', 'reports', f'EXP_{date_str}')
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, 'report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved {md_path}')
    return 0


if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else 'YYYYMMDD'
    sys.exit(main(date_str))