# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building a universal2 macOS application bundle of
the Copernican Suite. All project sources are included so the generated
``Copernican.app`` runs without external dependencies. The bundle may be signed
and notarised once an Apple Developer ID is available.
"""

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
    target_arch='universal2',
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
    target_arch='universal2',
)

app = BUNDLE(
    exe,
    name='Copernican.app',
    icon=None,
    bundle_identifier='org.copernican.suite',
)
