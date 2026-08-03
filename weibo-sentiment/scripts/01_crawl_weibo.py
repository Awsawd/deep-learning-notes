"""
微博评论采集（Playwright + Cookie）

用法:
  # 无 Cookie 时生成演示数据（含时间戳，便于后续趋势图）
  python scripts/01_crawl_weibo.py --demo

  # 真实采集：先在 .env 填 WEIBO_COOKIE，再指定博文页
  python scripts/01_crawl_weibo.py --status-url "https://weibo.com/..." --max-comments 500

说明:
  - 微博前端/接口常变，本脚本优先拦截 ajax 评论接口 JSON；失败则尝试 DOM 文本兜底。
  - Cookie 过期时会提示重新导出；请控制频率，仅用于个人学习。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SEEN_PATH = RAW_DIR / "_seen_ids.json"


def parse_cookie_header(cookie_str: str) -> list[dict]:
    """把浏览器复制的 Cookie 字符串转成 Playwright cookies。"""
    cookies = []
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookies.append(
            {
                "name": name.strip(),
                "value": value.strip(),
                "domain": ".weibo.com",
                "path": "/",
            }
        )
    return cookies


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    return set(json.loads(SEEN_PATH.read_text(encoding="utf-8")))


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_demo(n: int = 1200) -> Path:
    """生成带时间分布的样例评论，便于无 Cookie 时跑通后续流水线。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "comments_demo.jsonl"
    pos = [
        "支持！讲得很有道理",
        "太棒了，期待后续",
        "这个观点我赞同",
        "学到了，谢谢分享",
        "正能量满满",
        "分析很到位",
        "内容质量高",
        "值得一看再看",
    ]
    neu = [
        "围观一下",
        "先马后看",
        "了解了",
        "有人知道后续吗",
        "标记一下",
        "路过留名",
        "收到通知了",
        "看看大家怎么说",
    ]
    neg = [
        "不认同这个说法",
        "太离谱了吧",
        "看完很失望",
        "逻辑不通啊",
        "纯属扯淡",
        "浪费时间",
        "完全没法同意",
        "越看越无语",
    ]
    suffixes = [
        "，说说我的看法",
        "，仅代表个人",
        "，希望官方回应",
        "，相关话题讨论很多",
        "，晚上再仔细看",
        "，转给朋友了",
        "，评论区好热闹",
        "，先记录一下观点",
    ]
    topics = ["科技", "教育", "就业", "消费", "城市", "出行", "健康", "娱乐", "体育", "财经"]
    pools = [pos, neu, neg]
    base = datetime.now() - timedelta(days=14)
    rows = []
    for i in range(n):
        pool = pools[i % 3]
        text = (
            f"{random.choice(pool)}{random.choice(suffixes)}"
            f"。关于{random.choice(topics)}第{i}条补充：样本编号{i}。"
        )
        ts = base + timedelta(hours=i // 3, minutes=i % 60)
        rows.append(
            {
                "id": f"demo_{i}",
                "user": f"user_{i % 200}",
                "content": text,
                "created_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "status_url": "demo://status",
                "source": "demo",
            }
        )
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"已写入演示数据 {len(rows)} 条 -> {out}")
    return out


def extract_comments_from_payload(payload: dict | list, status_url: str) -> list[dict]:
    rows: list[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            # 常见字段：text_raw / text / content
            text = obj.get("text_raw") or obj.get("text") or obj.get("content")
            cid = obj.get("id") or obj.get("idstr") or obj.get("mid")
            user = None
            u = obj.get("user")
            if isinstance(u, dict):
                user = u.get("screen_name") or u.get("name")
            created = obj.get("created_at") or obj.get("createdAt")
            if text and cid:
                # 去掉简单 HTML
                clean = re.sub(r"<[^>]+>", "", str(text))
                rows.append(
                    {
                        "id": str(cid),
                        "user": user or "",
                        "content": clean.strip(),
                        "created_at": created or "",
                        "status_url": status_url,
                        "source": "weibo_ajax",
                    }
                )
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(payload)
    return rows


def crawl_status(status_url: str, cookie: str, max_comments: int, pause: float) -> Path:
    from playwright.sync_api import sync_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"comments_{stamp}.jsonl"
    seen = load_seen()
    collected: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        context.add_cookies(parse_cookie_header(cookie))
        page = context.new_page()

        def on_response(resp):
            try:
                url = resp.url
                if "comment" not in url.lower() and "buildComments" not in url:
                    return
                if "application/json" not in (resp.headers.get("content-type") or ""):
                    return
                data = resp.json()
                for row in extract_comments_from_payload(data, status_url):
                    if row["id"] in seen:
                        continue
                    if not row["content"]:
                        continue
                    seen.add(row["id"])
                    collected.append(row)
            except Exception:
                return

        page.on("response", on_response)
        print("打开:", status_url)
        page.goto(status_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)

        # 滚动加载更多评论
        idle_rounds = 0
        while len(collected) < max_comments and idle_rounds < 8:
            before = len(collected)
            page.mouse.wheel(0, 4000)
            # 尝试点击「查看更多评论」类按钮
            for text in ("查看更多", "更多评论", "加载更多"):
                try:
                    btn = page.get_by_text(text, exact=False).first
                    if btn.is_visible():
                        btn.click(timeout=1000)
                except Exception:
                    pass
            time.sleep(pause)
            if len(collected) == before:
                idle_rounds += 1
            else:
                idle_rounds = 0
            print(f"已收集 {len(collected)} / {max_comments}")

        # DOM 兜底：抓可见评论文本（无稳定 id 时用 hash）
        if len(collected) < min(20, max_comments):
            try:
                texts = page.locator("[class*='comment'] , [class*='Comment']").all_inner_texts()
                for i, t in enumerate(texts):
                    t = re.sub(r"\s+", " ", t).strip()
                    if len(t) < 2:
                        continue
                    cid = f"dom_{hash(t) & 0xFFFFFFFF}"
                    if cid in seen:
                        continue
                    seen.add(cid)
                    collected.append(
                        {
                            "id": cid,
                            "user": "",
                            "content": t[:500],
                            "created_at": "",
                            "status_url": status_url,
                            "source": "weibo_dom",
                        }
                    )
                    if len(collected) >= max_comments:
                        break
            except Exception as e:
                print("DOM 兜底失败:", e)

        browser.close()

    collected = collected[:max_comments]
    if not collected:
        raise RuntimeError(
            "未抓到评论。请检查：1) WEIBO_COOKIE 是否有效 2) 链接是否可打开评论 3) 是否触发验证码。"
            "也可先用 --demo 跑通后续流程。"
        )
    append_jsonl(out, collected)
    save_seen(seen)
    print(f"写入 {len(collected)} 条 -> {out}")
    return out


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="爬取微博评论")
    parser.add_argument("--demo", action="store_true", help="写入本地演示数据，不访问微博")
    parser.add_argument("--demo-n", type=int, default=1200, help="演示条数")
    parser.add_argument("--status-url", type=str, default="", help="微博博文页 URL")
    parser.add_argument("--max-comments", type=int, default=500)
    parser.add_argument("--pause", type=float, default=1.2, help="滚动间隔秒")
    args = parser.parse_args()

    if args.demo:
        write_demo(args.demo_n)
        return

    cookie = os.getenv("WEIBO_COOKIE", "").strip()
    if not cookie:
        raise SystemExit("未设置 WEIBO_COOKIE。请复制 .env.example 为 .env 并填入 Cookie，或使用 --demo。")
    if not args.status_url:
        raise SystemExit("请提供 --status-url，或使用 --demo。")

    crawl_status(args.status_url, cookie, args.max_comments, args.pause)


if __name__ == "__main__":
    main()
