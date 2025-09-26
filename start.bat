@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-09-28
@echo off
set "PKG_NOTICE=Package managers may request your password. The Copernican"
set "PKG_NOTICE=%PKG_NOTICE% Suite never reads or stores it."
REM Start the Copernican Suite on Windows.
REM
REM The script downloads a private Python 3.12+ into '.python', creates a
REM virtual environment from it and re-executes itself inside that
REM environment. System-wide Python installations are ignored.

setlocal
cd %~dp0
set "EXPECTED_VENV=%CD%\.venv"
set "PYDIR=%CD%\.python"
set "PYBIN=%PYDIR%\python.exe"

REM Skip setup when already inside the repository virtual environment.
if defined VIRTUAL_ENV (
    if /I not "%VIRTUAL_ENV%"=="%EXPECTED_VENV%" (
        echo Deactivate the active virtual environment before running.
        echo start.bat.
        exit /b 1
    )
    goto menu
)

REM Bootstrap a dedicated interpreter.
if not exist "%PYBIN%" (
    REM Create the extraction target before unpacking the archive.
    if not exist "%PYDIR%" mkdir "%PYDIR%"
    set "BASE=https://github.com/astral-sh/python-build-standalone/releases"
    set "REL=20250828"
    set "VER=3.12.11"
    set "ARCH=amd64"
    REM Build the download URL without caret continuations to keep it stable.
    set "URL_BASE=%BASE%/download/%REL%/"
    set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-pc-windows-msvc-"
    set "URL_FILE=%URL_FILE%shared-install_only.tar.gz"
    set "URL=%URL_BASE%%URL_FILE%"
    REM Surface the URL and archive path to PowerShell via environment variables
    REM so it can validate the values before downloading and unpacking.
    set "COPERNICAN_PYTHON_URL=%URL%"
    set "COPERNICAN_PYTHON_TAR=python.tar.gz"
    set "COPERNICAN_PYDIR=%PYDIR%"
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
        "$url = $env:COPERNICAN_PYTHON_URL;" ^
        "if ([string]::IsNullOrWhiteSpace($url))" ^
        " { throw 'Copernican Suite download URL is empty.' }" ^
        "Invoke-WebRequest -Uri $url -OutFile $env:COPERNICAN_PYTHON_TAR"
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
        "$tarPath = $env:COPERNICAN_PYTHON_TAR;" ^
        "$targetDir = $env:COPERNICAN_PYDIR;" ^
        "tar -xzf $tarPath -C $targetDir --strip-components=1"
    del python.tar.gz
)
set "PYTHON=%PYBIN%"

REM Create the virtual environment when missing.
if not exist .venv (
    "%PYTHON%" -m venv .venv
)

REM Retry environment creation once to catch rare failures.
if not exist .venv\Scripts\activate.bat (
    rmdir /s /q .venv
    "%PYTHON%" -m venv .venv
    if not exist .venv\Scripts\activate.bat (
        echo Virtual environment creation failed.
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
set PYTHON=python
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install -r requirements.lock
if exist build rmdir /s /q build
%PYTHON% -m pip install --no-deps .
if exist build rmdir /s /q build

call "%~f0" %*
goto :eof

:menu
set STRICT=0
set AUTO=0
:loop
echo Copernican Suite
echo 1^) Launch Copernican Suite
echo 2^) Run the unit test suite
if "%STRICT%"=="1" (
    echo 3^) Disable strict warning mode
) else (
    echo 3^) Enable strict warning mode
)
if "%AUTO%"=="1" (
    echo 4^) Disable automatic dependency installation
) else (
    echo 4^) Enable automatic dependency installation
)
echo 5^) Exit
set /p CHOICE=Select an option: 
if "%CHOICE%"=="1" (
    set COPERNICAN_STRICT_WARNINGS=%STRICT%
    set COPERNICAN_AUTO_INSTALL=%AUTO%
    python copernican.py
    goto :eof
)
if "%CHOICE%"=="2" (
    set COPERNICAN_STRICT_WARNINGS=%STRICT%
    set COPERNICAN_AUTO_INSTALL=%AUTO%
    python -m unittest discover -v
    goto :eof
)
if "%CHOICE%"=="3" (
    if "%STRICT%"=="1" (set STRICT=0) else (set STRICT=1)
    goto loop
)
if "%CHOICE%"=="4" (
    if "%AUTO%"=="1" (set AUTO=0) else (set AUTO=1)
    goto loop
)
if "%CHOICE%"=="5" goto :eof
goto loop

:winget_safe
echo %PKG_NOTICE%
winget %*
exit /b %ERRORLEVEL%

