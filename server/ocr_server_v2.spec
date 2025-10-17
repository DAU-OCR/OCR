# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

# ---- Collect all resources for critical libs (avoids "Numpy is not available") ----
pkgs = ['numpy', 'torch', 'torchvision', 'easyocr', 'cv2', 'ultralytics', 'seaborn']

datas = []
binaries = []
hiddenimports = []

for pkg in pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass  # package may not be installed; skip

# If you have local resource folders, include them here (existence optional).
for p in ['custom_weights', 'custom_weights_easyOCR', 'CRNN_model', 'yolov5']:
    if os.path.exists(p):
        datas.append((p, p))

# Some torch dynamo polyfills need explicit inclusion in some environments
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    'torch._dynamo.polyfills.tensor',
    'torch._dynamo.polyfills.math',
    'torch._dynamo.polyfills.random',
    'torch._dynamo.polyfills.builtins',
    'torch._dynamo.polyfills.operator',
]

# Optional: metadata (if needed for torchvision/others)
try:
    datas += copy_metadata('torchvision')
except Exception:
    pass

block_cipher = None

a = Analysis(
    ['app_v2.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assembly=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='ocr_server_v2',
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
