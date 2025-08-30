@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-08-30

@echo off
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
    goto run
)

REM Bootstrap a dedicated interpreter.
if not exist "%PYBIN%" (
    set "BASE=https://github.com/indygreg/python-build-standalone/releases"
    set "REL=20240710"
    set "VER=3.12.4"
    set "ARCH=amd64"
    set "URL=%BASE%/download/%REL%/cpython-%VER%+%REL%-%ARCH%-pc-windows-"
    set "URL=%URL%msvc-shared-install_only.zip"
    powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile 'python.zip'"
    powershell -Command "Expand-Archive 'python.zip' '%PYDIR%'"
    del python.zip
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
%PYTHON% -m pip install --require-hashes -r requirements.lock
if exist build rmdir /s /q build
%PYTHON% -m pip install --no-deps .
if exist build rmdir /s /q build

call "%~f0" %*
goto :eof

:run
python copernican.py %*

