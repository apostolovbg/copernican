# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building a macOS application bundle of the
Copernican Suite. All project sources are included so the generated
``Copernican.app`` runs without external dependencies. The bundle may be
signed and notarised once an Apple Developer ID is available. The build keeps
universal2 support on macOS while remaining portable for other platforms.
"""

import sys

# ``target_arch`` is only valid on macOS. Passing the argument on other
# platforms causes PyInstaller to abort, so we include it conditionally and
# propagate it through every build phase so the macOS bundle retains
# ``universal2`` support while other platforms build natively.
ARCH_ARGS = {"target_arch": "universal2"} if sys.platform == "darwin" else {}

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
    **ARCH_ARGS,
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
    **ARCH_ARGS,
)

app = BUNDLE(
    exe,
    name='Copernican.app',
    icon=None,
    bundle_identifier='org.copernican.suite',
    **ARCH_ARGS,
)
