# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: Screen Recorder & GUI Agent - PC
# Build: pyinstaller screen_recorder_pc.spec

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
_spec_dir = os.path.dirname(os.path.abspath(SPEC))
transformers_hidden = collect_submodules("transformers")
accelerate_hidden = collect_submodules("accelerate")
huggingface_hidden = collect_submodules("huggingface_hub")
torch_hidden = collect_submodules("torch")
tokenizers_hidden = collect_submodules("tokenizers")
vllm_hidden = []
extra_datas = (
    collect_data_files("transformers")
    + collect_data_files("tokenizers")
    + collect_data_files("huggingface_hub")
)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=extra_datas,
    hiddenimports=[
        'PIL',
        'PIL._tkinter_finder',
        'cv2',
        'numpy',
        'mss',
        'mss.windows',
        'flask',
        'flask_cors',
        'werkzeug',
        'requests',
        'jinja2',
        'click',
        'memory',
    ] + transformers_hidden + accelerate_hidden + huggingface_hidden + torch_hidden + tokenizers_hidden + vllm_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScreenRecorderPC',
    debug=False,
    icon=os.path.join(_spec_dir, 'icon.ico'),
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ScreenRecorderPC',
)
