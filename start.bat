@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-10-31
@echo off
set "PKG_NOTICE=Package managers may request your password. The Copernican"
set "PKG_NOTICE=%PKG_NOTICE% Suite never reads or stores it."
REM Start the Copernican Suite on Windows.
REM
REM The launcher now begins with a management menu so contributors can install,
REM reinstall or uninstall the managed interpreter before the runtime starts.
REM When the environment is already present the first option launches the suite
REM without reinstalling dependencies, keeping everyday usage quick while
REM leaving recovery tools one keystroke away.

setlocal
cd %~dp0
set "EXPECTED_VENV=%CD%\.venv"
set "PYDIR=%CD%\.python"
set "PYBIN=%PYDIR%\python.exe"
set "BASE=https://github.com/astral-sh/python-build-standalone/releases"
set "REL=20251028"
set "VER=3.11.14"
set "ARCH=amd64"
set "URL_BASE=%BASE%/download/%REL%/"
set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-pc-windows-msvc-"
set "URL_FILE=%URL_FILE%install_only.tar.gz"
set "DOWNLOAD_URL=%URL_BASE%%URL_FILE%"
set "DOWNLOAD_TAR=python.tar.gz"
set "COPERNICAN_PYTHON_URL=%DOWNLOAD_URL%"
set "COPERNICAN_PYTHON_TAR=python.tar.gz"
set "COPERNICAN_PYDIR=%PYDIR%"
set "COPERNICAN_VERSION_PROBE=import sys; print(1 if (3, 11) ^"
set "COPERNICAN_VERSION_PROBE=%COPERNICAN_VERSION_PROBE%<= ^"
set "COPERNICAN_VERSION_PROBE=%COPERNICAN_VERSION_PROBE%sys.version_info ^"
set "COPERNICAN_VERSION_PROBE=%COPERNICAN_VERSION_PROBE%< (3, 12) else 0)"

if defined VIRTUAL_ENV (
    if /I not "%VIRTUAL_ENV%"=="%EXPECTED_VENV%" (
        echo Deactivate the active virtual environment before running.
        echo start.bat.
        exit /b 1
    )
    goto runtime_menu
)

if defined COPERNICAN_LAUNCHER_TEST (
    if /I "%COPERNICAN_LAUNCHER_TEST%"=="print-menu" (
        call :print_main_menu
        goto :eof
    )
)

:main_menu
call :print_main_menu
if "%COPERNICAN_ENV_READY%"=="1" (
    set /p CHOICE=Select an option:
    if "%CHOICE%"=="1" goto runtime_menu
    if "%CHOICE%"=="2" goto reinstall
    if "%CHOICE%"=="3" goto uninstall
    if "%CHOICE%"=="4" goto :eof
) else (
    set /p CHOICE=Select an option:
    if "%CHOICE%"=="1" goto install
    if "%CHOICE%"=="2" goto :eof
)
goto main_menu

:install
call :install_dependencies
goto runtime_menu

:reinstall
call :reinstall_dependencies
goto runtime_menu

:uninstall
call :uninstall_dependencies
goto main_menu

:runtime_menu
call :ensure_environment_ready
if not "%COPERNICAN_ENV_READY%"=="1" (
    echo Managed dependencies are missing. Install them first.
    exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHON=python
set STRICT=0
set AUTO=0
:runtime_loop
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
    goto runtime_loop
)
if "%CHOICE%"=="4" (
    if "%AUTO%"=="1" (set AUTO=0) else (set AUTO=1)
    goto runtime_loop
)
if "%CHOICE%"=="5" goto :eof
goto runtime_loop

:install_dependencies
echo --- Installing managed dependencies ---
call :bootstrap_python
if errorlevel 1 exit /b 1
call :create_virtualenv
if errorlevel 1 exit /b 1
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
if exist .venv\Scripts\deactivate.bat call .venv\Scripts\deactivate.bat
echo Managed dependencies installed.
exit /b 0

:reinstall_dependencies
echo --- Reinstalling managed dependencies ---
if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
if exist .venv rmdir /s /q .venv
call :install_dependencies
exit /b %ERRORLEVEL%

:uninstall_dependencies
echo --- Removing managed dependencies ---
if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
if exist .venv rmdir /s /q .venv
exit /b 0

:ensure_environment_ready
call :detect_environment
exit /b 0

:print_main_menu
call :detect_environment
echo Copernican Suite
if "%COPERNICAN_ENV_READY%"=="1" (
    echo 1^) Use existing environment
    echo 2^) Reinstall dependencies
    echo 3^) Uninstall dependencies
    echo 4^) Exit
) else (
    echo 1^) Install dependencies
    echo 2^) Exit
)
exit /b 0

:detect_environment
set "COPERNICAN_ENV_READY=0"
set "COPERNICAN_PYOK=0"
if exist "%PYBIN%" (
    for /f "delims=" %%I in ('^
        "%PYBIN%" -c "%COPERNICAN_VERSION_PROBE%"^
    ') do set "COPERNICAN_PYOK=%%I"
)
if not "%COPERNICAN_PYOK%"=="1" (
    exit /b 0
)
set "COPERNICAN_VENV_OK=0"
if exist .venv\Scripts\python.exe (
    for /f "delims=" %%I in ('^
        ".venv\Scripts\python.exe" -c "%COPERNICAN_VERSION_PROBE%"^
    ') do set "COPERNICAN_VENV_OK=%%I"
)
if not "%COPERNICAN_VENV_OK%"=="1" (
    exit /b 0
)
if exist .venv\Scripts\activate.bat set "COPERNICAN_ENV_READY=1"
exit /b 0

:bootstrap_python
set "COPERNICAN_BOOTSTRAP=0"
set "COPERNICAN_PYOK=0"
if exist "%PYBIN%" (
    for /f "delims=" %%I in ('^
        "%PYBIN%" -c "%COPERNICAN_VERSION_PROBE%"^
    ') do set "COPERNICAN_PYOK=%%I"
    if not "%COPERNICAN_PYOK%"=="1" if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
)
if not exist "%PYBIN%" set "COPERNICAN_BOOTSTRAP=1"
if exist "%PYBIN%" if not "%COPERNICAN_PYOK%"=="1" set "COPERNICAN_BOOTSTRAP=1"
if "%COPERNICAN_BOOTSTRAP%"=="0" exit /b 0
if not exist "%PYDIR%" mkdir "%PYDIR%"
if "%DOWNLOAD_URL%"=="" (
    echo Copernican Suite download URL is empty.
    exit /b 1
)
call :download_python "%DOWNLOAD_URL%" "%DOWNLOAD_TAR%"
if errorlevel 1 exit /b 1
call :extract_python "%DOWNLOAD_TAR%" "%PYDIR%"
if errorlevel 1 exit /b 1
if exist "%DOWNLOAD_TAR%" del "%DOWNLOAD_TAR%"
exit /b 0

:create_virtualenv
set "COPERNICAN_VENV_OK=0"
if exist .venv\Scripts\python.exe (
    for /f "delims=" %%I in ('^
        ".venv\Scripts\python.exe" -c "%COPERNICAN_VERSION_PROBE%"^
    ') do set "COPERNICAN_VENV_OK=%%I"
    if not "%COPERNICAN_VENV_OK%"=="1" rmdir /s /q .venv
)
if not exist .venv (
    "%PYBIN%" -m venv .venv
)
if exist .venv\Scripts\activate.bat exit /b 0
rmdir /s /q .venv
"%PYBIN%" -m venv .venv
if not exist .venv\Scripts\activate.bat (
    echo Virtual environment creation failed.
    exit /b 1
)
exit /b 0

:download_python
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

:winget_safe
echo %PKG_NOTICE%
winget %*
exit /b %ERRORLEVEL%
