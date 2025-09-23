# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('server/custom_weights', 'custom_weights'), ('server/custom_weights_easyOCR', 'custom_weights_easyocr'), ('server/yolov5', 'yolov5')]
binaries = []
hiddenimports = ['flask_cors', 'easyocr', 'logging.config', 'gitpython', 'matplotlib', 'psutil', 'yaml', 'requests', 'scipy', 'thop', 'tqdm', 'tensorboard', 'wandb', 'seaborn', 'pandas', 'sklearn']
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['server\\app.py'],
    pathex=[],
    binaries=binaries,
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
    a.binaries,
    a.datas,
    [],
    name='ocr_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
