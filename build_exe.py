"""Build script for creating standalone exe using PyInstaller."""
import PyInstaller.__main__
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    os.path.join(HERE, "kakao_emoticon_bot", "main.py"),
    "--name", "KakaoStickerBot",
    "--onedir",
    "--console",
    "--noconfirm",
    "--clean",
    # Hidden imports that PyInstaller may miss
    "--hidden-import", "telegram",
    "--hidden-import", "telegram.ext",
    "--hidden-import", "telegram.constants",
    "--hidden-import", "aiohttp",
    "--hidden-import", "PIL",
    "--hidden-import", "dotenv",
    "--hidden-import", "apscheduler",
    "--hidden-import", "apscheduler.schedulers.asyncio",
    "--hidden-import", "apscheduler.triggers.interval",
    "--hidden-import", "httpx",
    "--hidden-import", "httpcore",
    "--hidden-import", "h11",
    "--hidden-import", "anyio",
    "--hidden-import", "anyio._backends",
    "--hidden-import", "anyio._backends._asyncio",
    "--hidden-import", "sniffio",
    "--hidden-import", "certifi",
    "--hidden-import", "multidict",
    "--hidden-import", "yarl",
    "--hidden-import", "aiosignal",
    "--hidden-import", "frozenlist",
    "--hidden-import", "attr",
    "--hidden-import", "attrs",
    "--hidden-import", "charset_normalizer",
    "--hidden-import", "idna",
    "--hidden-import", "PyMemoryEditor",
    "--hidden-import", "sticker_convert",
    "--hidden-import", "sticker_convert.utils.process",
    "--hidden-import", "psutil",
    "--hidden-import", "socksio",
    "--hidden-import", "tzdata",
    # Collect all submodules of packages that have complex structures
    "--collect-submodules", "telegram",
    "--collect-submodules", "apscheduler",
    "--collect-submodules", "httpx",
    "--collect-submodules", "httpcore",
    "--collect-submodules", "anyio",
    "--collect-submodules", "tzdata",
    "--collect-submodules", "sticker_convert.utils",
    "--collect-submodules", "PyMemoryEditor",
    # Collect data files
    "--collect-data", "tzdata",
    "--collect-data", "certifi",
    # Work directory
    "--workpath", os.path.join(HERE, "build"),
    "--distpath", os.path.join(HERE, "dist"),
    "--specpath", HERE,
])
