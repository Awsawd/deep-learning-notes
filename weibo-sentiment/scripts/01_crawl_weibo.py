"""
微博评论采集（Playwright + Cookie）

用法:
  # 真实采集：先在 .env 填 WEIBO_COOKIE，再指定博文页
  python scripts/01_crawl_weibo.py --status-url "https://weibo.com/..." --max-comments 500

说明:
  - 主路径：调用 buildComments 接口并用 max_id 翻页。
  - Cookie 过期时需重新导出；请控制频率，仅用于个人学习。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

# import random
# from datetime import timedelta

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


# ---------------------------------------------------------------------------
# 假数据 demo（已停用；需要时取消注释）
# ---------------------------------------------------------------------------
# def write_demo(n: int = 1200) -> Path:
#     """生成带时间分布的样例评论，便于无 Cookie 时跑通后续流水线。"""
#     RAW_DIR.mkdir(parents=True, exist_ok=True)
#     out = RAW_DIR / "comments_demo.jsonl"
#     pos = [
#         "支持！讲得很有道理",
#         "太棒了，期待后续",
#         "这个观点我赞同",
#         "学到了，谢谢分享",
#         "正能量满满",
#         "分析很到位",
#         "内容质量高",
#         "值得一看再看",
#     ]
#     neu = [
#         "围观一下",
#         "先马后看",
#         "了解了",
#         "有人知道后续吗",
#         "标记一下",
#         "路过留名",
#         "收到通知了",
#         "看看大家怎么说",
#     ]
#     neg = [
#         "不认同这个说法",
#         "太离谱了吧",
#         "看完很失望",
#         "逻辑不通啊",
#         "纯属扯淡",
#         "浪费时间",
#         "完全没法同意",
#         "越看越无语",
#     ]
#     suffixes = [
#         "，说说我的看法",
#         "，仅代表个人",
#         "，希望官方回应",
#         "，相关话题讨论很多",
#         "，晚上再仔细看",
#         "，转给朋友了",
#         "，评论区好热闹",
#         "，先记录一下观点",
#     ]
#     topics = ["科技", "教育", "就业", "消费", "城市", "出行", "健康", "娱乐", "体育", "财经"]
#     pools = [pos, neu, neg]
#     base = datetime.now() - timedelta(days=14)
#     rows = []
#     for i in range(n):
#         pool = pools[i % 3]
#         text = (
#             f"{random.choice(pool)}{random.choice(suffixes)}"
#             f"。关于{random.choice(topics)}第{i}条补充：样本编号{i}。"
#         )
#         ts = base + timedelta(hours=i // 3, minutes=i % 60)
#         rows.append(
#             {
#                 "id": f"demo_{i}",
#                 "user": f"user_{i % 200}",
#                 "content": text,
#                 "created_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
#                 "status_url": "demo://status",
#                 "source": "demo",
#             }
#         )
#     out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
#     print(f"已写入演示数据 {len(rows)} 条 -> {out}")
#     return out


def parse_status_ids(status_url: str) -> tuple[str | None, str | None]:
    """从 https://weibo.com/{uid}/{mblogid} 解析 uid 与 mblogid。"""
    m = re.search(r"weibo\.com/(\d+)/([A-Za-z0-9]+)", status_url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def comments_from_build_payload(payload: dict, status_url: str) -> tuple[list[dict], int | None, int | None]:
    """解析 buildComments 返回：只取 data 列表，避免递归把同一条算多次。"""
    rows: list[dict] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        data = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        text = obj.get("text_raw") or obj.get("text") or ""
        cid = obj.get("id") or obj.get("idstr") or obj.get("mid")
        if not text or not cid:
            continue
        u = obj.get("user") if isinstance(obj.get("user"), dict) else {}
        user = u.get("screen_name") or u.get("name") or ""
        rows.append(
            {
                "id": str(cid),
                "user": user,
                "content": re.sub(r"<[^>]+>", "", str(text)).strip(),
                "created_at": str(obj.get("created_at") or ""),
                "status_url": status_url,
                "source": "weibo_api",
            }
        )
    max_id = payload.get("max_id") if isinstance(payload, dict) else None
    total = payload.get("total_number") if isinstance(payload, dict) else None
    try:
        max_id_i = int(max_id) if max_id is not None else None
    except (TypeError, ValueError):
        max_id_i = None
    try:
        total_i = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_i = None
    return rows, max_id_i, total_i


def crawl_status(
    status_url: str,
    cookie: str,
    max_comments: int,
    pause: float,
    *,
    reset_seen: bool = False,
    headed: bool = False,
    flow: int = 1,
) -> Path:
    """用评论接口 max_id 翻页采集。"""
    from playwright.sync_api import sync_playwright
    from urllib.parse import urlencode

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"comments_{stamp}.jsonl"
    if reset_seen and SEEN_PATH.exists():
        SEEN_PATH.unlink()
        print("已清空 _seen_ids.json")
    seen = load_seen()
    print(f"当前已记录 seen={len(seen)} 条（重复 ID 会跳过）")
    collected: list[dict] = []

    uid, mblogid = parse_status_ids(status_url)
    if not mblogid:
        raise SystemExit(f"无法从 URL 解析博文 ID，请使用类似 https://weibo.com/uid/mblogid 的链接: {status_url}")

    with sync_playwright() as p:
        browser = None
        last_err: Exception | None = None
        for launch_kwargs in (
            {"channel": "chrome", "headless": not headed},
            {"channel": "msedge", "headless": not headed},
            {"headless": not headed},
        ):
            try:
                browser = p.chromium.launch(**launch_kwargs)
                print("浏览器启动方式:", launch_kwargs)
                break
            except Exception as e:
                last_err = e
        if browser is None:
            raise RuntimeError(
                "无法启动浏览器。请安装 Chrome/Edge，或清理磁盘后执行: "
                "python -m playwright install chromium"
            ) from last_err

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            viewport={"width": 1400, "height": 900},
        )
        context.add_cookies(parse_cookie_header(cookie))
        page = context.new_page()
        # 先打开一次页面，带上站点上下文（部分接口会校验 referer）
        print("打开:", status_url)
        page.goto(status_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)

        # 1) show 接口拿数字 mid / uid
        show = context.request.get(
            f"https://weibo.com/ajax/statuses/show?id={mblogid}",
            headers={"referer": status_url},
        )
        if not show.ok:
            browser.close()
            raise RuntimeError(f"获取博文信息失败 HTTP {show.status}，Cookie 可能失效或需验证码。")
        show_json = show.json()
        mid = str(show_json.get("id") or show_json.get("mid") or show_json.get("idstr") or "")
        if not uid:
            user = show_json.get("user") if isinstance(show_json.get("user"), dict) else {}
            uid = str(user.get("id") or user.get("idstr") or "")
        if not mid or not uid:
            browser.close()
            raise RuntimeError(f"无法解析 mid/uid。show 返回键: {list(show_json)[:20]}")
        print(f"博文 mid={mid} uid={uid} mblogid={mblogid}")
        total_hint = show_json.get("comments_count") or show_json.get("comment_count")
        if total_hint is not None:
            print(f"页面显示评论数约: {total_hint}（接口可翻页部分，楼中楼另算）")

        # 2) buildComments + max_id 翻页（flow=1 按时间，通常比热门更全）
        max_id = 0
        page_i = 0
        stagnant = 0
        while len(collected) < max_comments and stagnant < 3:
            page_i += 1
            params = {
                "flow": flow,
                "is_reload": 1,
                "id": mid,
                "is_show_bulletin": 2,
                "is_mix": 1 if max_id else 0,
                "count": 20,
                "uid": uid,
                "fetch_level": 0,
                "locale": "zh-CN",
            }
            if max_id:
                params["max_id"] = max_id
            api = "https://weibo.com/ajax/statuses/buildComments?" + urlencode(params)
            resp = context.request.get(api, headers={"referer": status_url})
            if not resp.ok:
                print(f"第 {page_i} 页失败 HTTP {resp.status}")
                break
            payload = resp.json()
            rows, next_max_id, total_number = comments_from_build_payload(payload, status_url)
            if page_i == 1 and total_number is not None:
                print(f"接口 total_number≈{total_number}")

            before = len(collected)
            for row in rows:
                if row["id"] in seen:
                    continue
                if not row["content"]:
                    continue
                seen.add(row["id"])
                collected.append(row)
                if len(collected) >= max_comments:
                    break
            gained = len(collected) - before
            print(
                f"翻页 {page_i}: 本页{len(rows)}条 新增{gained} 累计{len(collected)}/{max_comments} "
                f"next_max_id={next_max_id}"
            )

            if next_max_id is None or next_max_id == 0:
                print("接口返回 max_id=0，分页结束")
                break
            if gained == 0:
                stagnant += 1
            else:
                stagnant = 0
            max_id = next_max_id
            time.sleep(pause)

        browser.close()

    collected = collected[:max_comments]
    if not collected:
        raise RuntimeError(
            "未抓到新评论。可换博文、检查 Cookie，或加 --reset-seen / --headed。"
        )
    append_jsonl(out, collected)
    save_seen(seen)
    print(f"写入 {len(collected)} 条 -> {out}")
    return out


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="爬取微博评论")
    # parser.add_argument("--demo", action="store_true", help="写入本地演示数据，不访问微博")
    # parser.add_argument("--demo-n", type=int, default=1200, help="演示条数")
    parser.add_argument("--status-url", type=str, required=True, help="微博博文页 URL")
    parser.add_argument("--max-comments", type=int, default=500)
    parser.add_argument("--pause", type=float, default=1.2, help="翻页间隔秒")
    parser.add_argument("--reset-seen", action="store_true", help="清空已抓 ID 去重文件后重跑")
    parser.add_argument("--headed", action="store_true", help="有界面模式，便于看是否跳登录/验证码")
    parser.add_argument(
        "--flow",
        type=int,
        default=1,
        choices=[0, 1],
        help="0=按热度 1=按时间（默认，通常更全）",
    )
    args = parser.parse_args()

    # if args.demo:
    #     write_demo(args.demo_n)
    #     return

    cookie = os.getenv("WEIBO_COOKIE", "").strip()
    if not cookie:
        raise SystemExit("未设置 WEIBO_COOKIE。请复制 .env.example 为 .env 并填入 Cookie。")

    crawl_status(
        args.status_url,
        cookie,
        args.max_comments,
        args.pause,
        reset_seen=args.reset_seen,
        headed=args.headed,
        flow=args.flow,
    )


if __name__ == "__main__":
    main()
