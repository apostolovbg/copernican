# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building a macOS application bundle of the
Copernican Suite. All project sources are included so the generated
``Copernican.app`` runs without external dependencies. The bundle may be
signed and notarised once an Apple Developer ID is available. The build keeps
universal2 support on macOS while remaining portable for other platforms.
"""

import sys

TARGET_ARCH = "universal2" if sys.platform == "darwin" else None
# Use host architecture on non-mac platforms to keep CI builds portable.

block_cipher = None

a = Analysis(
    ['copernican.py'],
    pathex=['.'],
    datas=[
        ('copernican_lib', 'copernican_lib'),
        ('engines', 'engines'),
        ('models', 'models'),
        ('data', 'data'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    target_arch=TARGET_ARCH,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Copernican',
    console=False,
    target_arch=TARGET_ARCH,
)

app = BUNDLE(
    exe,
    name='Copernican.app',
    icon=None,
    bundle_identifier='org.copernican.suite',
)
