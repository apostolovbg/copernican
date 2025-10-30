@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-10-30 17:56 UTC
@echo off
set "PKG_NOTICE=Package managers may request your password. The Copernican"
set "PKG_NOTICE=%PKG_NOTICE% Suite never reads or stores it."
REM Start the Copernican Suite on Windows.
REM
REM The script downloads a private Python 3.11 interpreter into '.python',
REM creates a virtual environment from it and re-executes itself inside that
REM environment. System-wide Python installations are ignored so Python
REM 3.12 never leaks into the managed bootstrap sequence.

setlocal
cd %~dp0
set "EXPECTED_VENV=%CD%\.venv"
set "PYDIR=%CD%\.python"
set "PYBIN=%PYDIR%\python.exe"
REM Precompute release metadata outside conditionals so cmd.exe expands
REM each token correctly even without delayed expansion.
set "BASE=https://github.com/astral-sh/python-build-standalone/releases"
set "REL=20251028"
set "VER=3.11.14"
set "ARCH=amd64"
set "URL_BASE=%BASE%/download/%REL%/"
set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-pc-windows-msvc-"
set "URL_FILE=%URL_FILE%install_only.tar.gz"
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

REM Bootstrap a dedicated interpreter. Delete stale downloads that fall
REM outside the Python 3.11 window so the managed environment always
REM satisfies the runtime requirement.
set "COPERNICAN_BOOTSTRAP=0"
set "COPERNICAN_PYOK=0"
if exist "%PYBIN%" (
    for /f "delims=" %%I in ('"%PYBIN%" -c "import sys; print(1 if (3, 11) <= sys.version_info < (3, 12) else 0)"') do set "COPERNICAN_PYOK=%%I"
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
    for /f "delims=" %%I in ('".venv\Scripts\python.exe" -c "import sys; print(1 if (3, 11) <= sys.version_info < (3, 12) else 0)"') do set "COPERNICAN_VENV_OK=%%I"
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
call :ensure_pip
if errorlevel 1 (
    echo Unable to bootstrap pip in the Copernican virtual environment.
    exit /b 1
)
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

:ensure_pip
%PYTHON% -m ensurepip --upgrade
if errorlevel 1 (
    set "COPERNICAN_GETPIP=%TEMP%\copernican-get-pip.py"
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
        "& { param([string]$path) ^
            Set-StrictMode -Version Latest; ^
            $uri = 'https://bootstrap.pypa.io/get-pip.py'; ^
            Invoke-WebRequest -Uri $uri -OutFile $path ^
        }" ^
        -Args "%COPERNICAN_GETPIP%"
    if errorlevel 1 (
        echo Failed to download get-pip.py.
        exit /b 1
    )
    %PYTHON% "%COPERNICAN_GETPIP%"
    set "COPERNICAN_ENSURE_ERR=%ERRORLEVEL%"
    if exist "%COPERNICAN_GETPIP%" del "%COPERNICAN_GETPIP%"
    if not "%COPERNICAN_ENSURE_ERR%"=="0" (
        echo Failed to bootstrap pip via get-pip.py.
        exit /b %COPERNICAN_ENSURE_ERR%
    )
)
exit /b 0

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

