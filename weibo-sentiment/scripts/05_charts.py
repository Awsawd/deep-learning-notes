"""
根据推理结果生成情感趋势图 + 词云 → outputs/charts/
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line, Page
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = PROJECT_ROOT / "data" / "predicted" / "comments_pred.parquet"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"

STOPWORDS = set(
    "的 了 是 我 都 而 及 與 与 和 或 在 有 就 也 很 到 说 这 那 你 他 她 它 们 "
    "一个 没有 什么 不是 可以 自己 这个 那个 还是 就是 因为 所以 如果 但是".split()
)


def make_trend(df: pd.DataFrame, out_html: Path) -> None:
    if "date" not in df.columns or df["date"].isna().all():
        # 无日期则按序号分桶
        df = df.copy()
        df["date"] = (df.index // max(1, len(df) // 14)).astype(str)
    g = df.groupby(["date", "pred_label_zh"]).size().unstack(fill_value=0)
    for col in ["正面", "中性", "负面"]:
        if col not in g.columns:
            g[col] = 0
    g = g[["正面", "中性", "负面"]]
    x = [str(i) for i in g.index.tolist()]
    line = (
        Line()
        .add_xaxis(x)
        .add_yaxis("正面", g["正面"].tolist(), is_smooth=True)
        .add_yaxis("中性", g["中性"].tolist(), is_smooth=True)
        .add_yaxis("负面", g["负面"].tolist(), is_smooth=True)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="微博评论情感趋势"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
        )
    )
    page = Page(layout=Page.SimplePageLayout)
    page.add(line)
    page.render(str(out_html))
    print("趋势图:", out_html)


def make_wordcloud(df: pd.DataFrame, label_zh: str, out_png: Path) -> None:
    sub = df[df["pred_label_zh"] == label_zh]
    if sub.empty:
        print(f"跳过词云 {label_zh}: 无样本")
        return
    counter: Counter[str] = Counter()
    for text in sub["content"].astype(str):
        for w in jieba.lcut(text):
            w = w.strip()
            if len(w) < 2 or w in STOPWORDS or w.isdigit():
                continue
            counter[w] += 1
    if not counter:
        print(f"跳过词云 {label_zh}: 无有效词")
        return
    wc = WordCloud(width=1000, height=600, background_color="white", font_path=None)
    # Windows 常见中文字体
    for fp in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        if Path(fp).exists():
            wc = WordCloud(width=1000, height=600, background_color="white", font_path=fp)
            break
    img = wc.generate_from_frequencies(counter)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.to_file(str(out_png))
    print("词云:", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=str(PRED_PATH))
    args = parser.parse_args()
    path = Path(args.pred)
    if not path.exists():
        raise SystemExit(f"缺少预测文件: {path}，请先运行 04_infer_weibo.py")

    df = pd.read_parquet(path)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    make_trend(df, CHART_DIR / "sentiment_trend.html")
    make_wordcloud(df, "正面", CHART_DIR / "wordcloud_positive.png")
    make_wordcloud(df, "负面", CHART_DIR / "wordcloud_negative.png")
    print("图表目录:", CHART_DIR)


if __name__ == "__main__":
    main()
