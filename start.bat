@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-11-24
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
REM Precompute release metadata outside conditionals so cmd.exe expands
REM each token correctly even without delayed expansion.
set "BASE=https://github.com/astral-sh/python-build-standalone/releases"
set "REL=20250828"
set "VER=3.12.11"
set "ARCH=amd64"
set "URL_BASE=%BASE%/download/%REL%/"
set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-pc-windows-msvc-"
set "URL_FILE=%URL_FILE%shared-install_only.tar.gz"
set "URL=%URL_BASE%%URL_FILE%"
set "DOWNLOAD_URL=%URL%"
set "DOWNLOAD_TAR=python.tar.gz"
set "COPERNICAN_PYTHON_URL=%URL%"
set "COPERNICAN_PYTHON_TAR=python.tar.gz"
set "COPERNICAN_PYDIR=%PYDIR%"

REM Skip setup when already inside the repository virtual environment.
if defined VIRTUAL_ENV (
    if /I not "%VIRTUAL_ENV%"=="%EXPECTED_VENV%" (
        echo Deactivate the active virtual environment before running.
        echo start.bat.
        exit /b 1
    )
    goto menu
)

REM Bootstrap a dedicated interpreter. Delete stale downloads that predate
REM the Python 3.12 floor so the managed environment always satisfies the
REM runtime requirement.
set "COPERNICAN_BOOTSTRAP=0"
set "COPERNICAN_PYOK=0"
if exist "%PYBIN%" (
    for /f "delims=" %%I in ('"%PYBIN%" -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)"') do set "COPERNICAN_PYOK=%%I"
    if not defined COPERNICAN_PYOK set "COPERNICAN_PYOK=0"
    if not "%COPERNICAN_PYOK%"=="1" if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
)
if not exist "%PYBIN%" set "COPERNICAN_BOOTSTRAP=1"
if exist "%PYBIN%" if not "%COPERNICAN_PYOK%"=="1" set "COPERNICAN_BOOTSTRAP=1"
if "%COPERNICAN_BOOTSTRAP%"=="1" (
    REM Create the extraction target before unpacking the archive.
    if not exist "%PYDIR%" mkdir "%PYDIR%"
    REM Fail fast when the computed URL is blank so the user sees a clear
    REM diagnostic instead of a confusing PowerShell error.
    if "%DOWNLOAD_URL%"=="" (
        echo Copernican Suite download URL is empty.
        exit /b 1
    )
)
if "%COPERNICAN_BOOTSTRAP%"=="1" call :download_python "%DOWNLOAD_URL%" "%DOWNLOAD_TAR%"
if errorlevel 1 exit /b 1
if "%COPERNICAN_BOOTSTRAP%"=="1" call :extract_python "%DOWNLOAD_TAR%" "%PYDIR%"
if errorlevel 1 exit /b 1
if "%COPERNICAN_BOOTSTRAP%"=="1" del python.tar.gz
set "PYTHON=%PYBIN%"

REM Create the virtual environment when missing.
set "COPERNICAN_VENV_OK=0"
if exist .venv\Scripts\python.exe (
    for /f "delims=" %%I in ('".venv\Scripts\python.exe" -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)"') do set "COPERNICAN_VENV_OK=%%I"
    if not defined COPERNICAN_VENV_OK set "COPERNICAN_VENV_OK=0"
    if not "%COPERNICAN_VENV_OK%"=="1" rmdir /s /q .venv
)
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

:download_python
REM Download the bundled Python interpreter through PowerShell.
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "& { param([string]$urlParam, [string]$outFile) ^
        Set-StrictMode -Version Latest; ^
        $url = $env:COPERNICAN_PYTHON_URL; ^
        if ([string]::IsNullOrWhiteSpace($url)) { ^
            $url = $urlParam; ^
        } ^
        if ([string]::IsNullOrWhiteSpace($url)) { ^
            throw 'Copernican Suite download URL is empty.' ^
        } ^
        Invoke-WebRequest -Uri $url -OutFile $outFile ^
    }" ^
    -Args "%~1", "%~2"
exit /b %ERRORLEVEL%

:extract_python
REM Extract the downloaded interpreter archive with validation.
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "& { param([string]$tarPath, [string]$targetDir) ^
        Set-StrictMode -Version Latest; ^
        if (-not (Test-Path -Path $tarPath -PathType Leaf)) { ^
            throw 'Copernican Suite download archive is missing.' ^
        } ^
        & tar -xzf $tarPath -C $targetDir --strip-components=1 ^
    }" ^
    -Args "%~1", "%~2"
exit /b %ERRORLEVEL%

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

