# -*- mode: python ; coding: utf-8 -*-

"""
Label Studio Helper - PyInstaller Spec
Embedded Python + 외부 PyTorch 방식
"""

import sys
from pathlib import Path

# Python 표준 라이브러리 경로
python_lib = Path(sys.base_prefix) / 'Lib'

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('core', 'core'),
        # Python 표준 라이브러리 (PyTorch import에 필수)
        (str(python_lib / 'modulefinder.py'), 'Lib'),
        (str(python_lib / 'importlib'), 'Lib/importlib'),
        (str(python_lib / 'pkgutil.py'), 'Lib'),
        (str(python_lib / 'inspect.py'), 'Lib'),
        (str(python_lib / 'dis.py'), 'Lib'),
        (str(python_lib / 'opcode.py'), 'Lib'),
        (str(python_lib / 'ast.py'), 'Lib'),
    ],
    hiddenimports=[
        # gradio 및 의존성
        'gradio', 'gradio.routes', 'gradio.components', 'gradio.blocks',
        'fastapi', 'uvicorn', 'starlette', 'httpx', 'websockets',
        'pydantic', 'jinja2', 'markdown', 'pygments',
        # 비디오/이미지 처리
        'cv2', 'skimage', 'skimage.metrics', 'skimage._shared', 'skimage._shared.utils',
        'numpy', 'PIL', 'matplotlib',
        # PyAV (AV1 코덱)
        'av', 'av.logging', 'av.audio', 'av.video', 'av.container', 'av.codec',
        'av.stream', 'av.format', 'av.filter', 'av.packet', 'av.frame', 'av.option',
        'av.subtitles', 'av.data', 'av.buffer', 'av.error', 'av.utils', 'av._core',
        # Python 표준 라이브러리
        'modulefinder', 'importlib', 'importlib.util', 'importlib.machinery',
        'importlib.metadata', 'importlib.resources', 'importlib.abc',
        'pkgutil', 'inspect', 'typing', 'typing_extensions',
        'collections', 'collections.abc', 'functools', 'operator', 'itertools',
        'contextlib', 'warnings', 'dis', 'opcode', 'token', 'tokenize',
        'linecache', 'traceback', 'ast', 'keyword', 'reprlib',
        'io', 'sys', 'os', 'os.path', 'pathlib', 're', 'json', 'math',
        'platform', 'subprocess', 'threading', 'queue', 'time', 'datetime',
        # Windows
        'psutil', 'win32api', 'win32security', 'win32process', 'win32con',
        'ctypes', 'ctypes.wintypes',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/runtime_hook_stdlib.py'],
    excludes=[
        # PyTorch는 제외 (런타임에 외부 설치)
        'torch', 'torchvision', 'torchaudio',
        'torch.nn', 'torch.cuda', 'torch.distributed', 'torch.optim',
        'torch.utils', 'torch.jit', 'torch.autograd',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='label_studio_helper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 로그 확인용 콘솔 유지
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: 아이콘 추가
    # 일반 사용자 권한 강제 (requestedExecutionLevel)
    uac_admin=False,
    uac_uiaccess=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='label_studio_helper',
)
