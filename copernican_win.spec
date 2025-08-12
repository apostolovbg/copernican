# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building a standalone Windows executable of the
Copernican Suite. The project source directories are bundled so the resulting
``copernican.exe`` can run without requiring a separate checkout of the
repository.
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
