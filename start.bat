@REM Copyright (c) 2025 Copernican Suite developers.
@REM See LICENSE.md in the repository root for details.
@REM Last Updated: 2025-08-31

@echo off
REM Start the Copernican Suite on Windows.
REM
REM The script locates a Python interpreter, sets up a virtual environment and
REM re-executes itself inside that environment so later runs reuse it.

setlocal
cd %~dp0
set "EXPECTED_VENV=%CD%\.venv"

REM Skip setup when already inside the repository virtual environment.
if defined VIRTUAL_ENV (
    if /I not "%VIRTUAL_ENV%"=="%EXPECTED_VENV%" (
        echo Deactivate the active virtual environment before running start.bat.
        exit /b 1
    )
    goto run
)

REM Locate python.exe or the py launcher.
where python >NUL 2>NUL
if %ERRORLEVEL%==0 (
    set "PYTHON=python"
    set "PYTHON_CMD=python"
) else (
    where py >NUL 2>NUL
    if %ERRORLEVEL%==0 (
        set "PYTHON=py"
        set "PYTHON_CMD=py -3.12"
    ) else (
        echo Python 3.12 is not installed.
        echo Install it with "winget install -e --id Python.Python.3.12" ^
or visit https://www.python.org/downloads/
        exit /b 1
    )
)

REM Verify interpreter version by parsing '--version' output.
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>NUL') do ^
    set "PYVERSION=%%v"
if not defined PYVERSION goto needpython
for /f "tokens=1-2 delims=." %%a in ("%PYVERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if %MAJOR% LSS 3 goto needpython
    if %MAJOR%==3 if %MINOR% LSS 12 goto needpython

if not exist .venv (
    %PYTHON_CMD% -m venv .venv
)

REM Ensure the activation script exists. Recreate once before suggesting that
REM the 'venv' component is missing.
if not exist .venv\Scripts\activate.bat (
    rmdir /s /q .venv
    %PYTHON_CMD% -m venv .venv
    if not exist .venv\Scripts\activate.bat (
        echo Virtual environment support is missing.
        echo Install the Python 'venv' component and try again.
        echo On Debian/Ubuntu: sudo apt install python3.12-venv
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

:needpython
echo Python 3.12 or newer is required.
echo Install it with "winget install -e --id Python.Python.3.12" ^
or visit https://www.python.org/downloads/
exit /b 1

