# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [('custom_weights', 'custom_weights'), ('custom_weights_easyOCR', 'custom_weights_easyOCR'), ('yolov5', 'yolov5'), ('CRNN_model', 'CRNN_model')] + copy_metadata('torchvision')
binaries = []
hiddenimports = [
    'flask_cors', 'easyocr', 'logging.config', 'gitpython', 'matplotlib', 'psutil', 'yaml', 'requests', 'scipy', 'thop', 'tqdm', 'tensorboard', 'wandb', 'seaborn', 'pandas', 'sklearn',
    # Manually add the required torch._dynamo.polyfills modules
    'torch._dynamo.polyfills.fx',
    'torch._dynamo.polyfills.tensor',
    'torch._dynamo.polyfills.math',
    'torch._dynamo.polyfills.random',
    'torch._dynamo.polyfills.builtins',
    'torch._dynamo.polyfills.operator',
]
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
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

# Filter out all CUDA-related DLLs to force CPU execution
a.binaries = [x for x in a.binaries if 'cuda' not in os.path.basename(x[0])]

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
