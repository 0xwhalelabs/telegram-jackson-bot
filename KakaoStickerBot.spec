# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = []
hiddenimports = ['telegram', 'telegram.ext', 'telegram.constants', 'aiohttp', 'PIL', 'dotenv', 'apscheduler', 'apscheduler.schedulers.asyncio', 'apscheduler.triggers.interval', 'httpx', 'httpcore', 'h11', 'anyio', 'anyio._backends', 'anyio._backends._asyncio', 'sniffio', 'certifi', 'multidict', 'yarl', 'aiosignal', 'frozenlist', 'attr', 'attrs', 'charset_normalizer', 'idna', 'PyMemoryEditor', 'sticker_convert', 'sticker_convert.utils.process', 'psutil', 'socksio', 'tzdata']
datas += collect_data_files('tzdata')
datas += collect_data_files('certifi')
hiddenimports += collect_submodules('telegram')
hiddenimports += collect_submodules('apscheduler')
hiddenimports += collect_submodules('httpx')
hiddenimports += collect_submodules('httpcore')
hiddenimports += collect_submodules('anyio')
hiddenimports += collect_submodules('tzdata')
hiddenimports += collect_submodules('sticker_convert.utils')
hiddenimports += collect_submodules('PyMemoryEditor')


a = Analysis(
    ['C:\\Users\\TAENG\\CascadeProjects\\whalet-telegram-exp-bot\\kakao_emoticon_bot\\main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KakaoStickerBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KakaoStickerBot',
)
