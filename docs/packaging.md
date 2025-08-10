# Packaging Guide

This guide describes how to build standalone executables for the Copernican
Suite using PyInstaller. Each provided spec file bundles the project
source code so the resulting binary can run without an existing
checkout of the repository or a pre-installed copy of Python.

Builds should target Python 3.11 or later and include `camb==1.6.2` to match
the suite's runtime requirements.

## Windows `.exe`
1. Install PyInstaller: `pip install pyinstaller`.
2. Run `pyinstaller copernican_win.spec`.
3. The `dist/copernican.exe` file contains the suite and all bundled sources.

## macOS universal2 `.app`
1. Install PyInstaller on a macOS system.
2. Run `pyinstaller copernican_macos.spec`.
3. The `dist/Copernican.app` bundle supports both Intel and Apple Silicon. The
   spec excludes the optional `yaml._yaml` extension so the pure-Python
   fallback keeps the app universal.

### Signing and notarizing
Once you have an Apple Developer ID, the bundle can be signed and notarized:

```bash
codesign --deep --force --options runtime \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" dist/Copernican.app
hdiutil create -fs HFS+ -volname Copernican \
  -srcfolder dist/Copernican.app dist/copernican.dmg
xcrun notarytool submit dist/copernican.dmg --apple-id YOUR_ID@example.com \
  --team-id TEAMID --password YOUR_APP_SPECIFIC_PASSWORD --wait
xcrun stapler staple dist/Copernican.app
```

These steps sign the application, submit it to Apple for notarization and
staple the approval ticket so the app runs without security prompts.

## Linux self-contained binary
1. Install PyInstaller: `pip install pyinstaller`.
2. Run `pyinstaller copernican_linux.spec`.
3. The `dist/copernican` file is a one-file binary containing the full project source.
