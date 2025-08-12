# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building a self-contained Linux binary of the
Copernican Suite. Project sources are bundled so the resulting executable can
run on systems without Python pre-installed.
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
    hiddenimports=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'sympy',
        'jsonschema',
        'camb',
        'yaml',
        'astropy',
        'psutil',
        'setuptools_scm',
    ],
    hookspath=[],
    hooksconfig={},
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='copernican',
    console=True,
)
