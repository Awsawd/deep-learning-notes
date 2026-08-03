"""
下载 CLUENER2020 公开数据到 data/cluener/
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cluener"

# 官方镜像（可能需代理）；失败则尝试 GitHub 附件
URLS = [
    "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip",
    "https://github.com/CLUEbenchmark/CLUENER2020/files/6371700/cluener_public.zip",
]


def download_zip(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    needed = ["train.json", "dev.json", "test.json"]
    if all((DATA_DIR / name).exists() for name in needed):
        print(f"已存在数据，跳过下载: {DATA_DIR}")
        for name in needed:
            print(f"  - {name}: {(DATA_DIR / name).stat().st_size} bytes")
        return

    last_err: Exception | None = None
    raw: bytes | None = None
    for url in URLS:
        try:
            print(f"下载中: {url}")
            raw = download_zip(url)
            print(f"下载完成，大小 {len(raw)} bytes")
            break
        except Exception as e:  # noqa: BLE001 — 尝试下一个镜像
            print(f"失败: {e}")
            last_err = e

    if raw is None:
        raise RuntimeError(
            "所有下载源均失败。请手动下载 cluener_public.zip 并解压到 "
            f"{DATA_DIR}\n最后错误: {last_err}"
        )

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # zip 内可能带一层目录，统一抽到 DATA_DIR
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name.endswith(".json") or name in {"README.md", "vocab.txt"}:
                target = DATA_DIR / name
                target.write_bytes(zf.read(info))
                print(f"写出 {target}")

    missing = [n for n in needed if not (DATA_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"解压后仍缺少: {missing}")
    print("CLUENER 数据准备就绪。")


if __name__ == "__main__":
    main()
