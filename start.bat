@echo off
REM Start the Copernican Suite on Windows.
REM
REM The script locates a Python interpreter, sets up a virtual environment and
REM re-executes itself inside that environment so later runs reuse it.

setlocal
cd %~dp0

if not "%VIRTUAL_ENV%"=="" goto run

REM Locate python.exe or the py launcher.
where python >NUL 2>NUL
if %ERRORLEVEL%==0 (
    set "PYTHON=python"
    set "PYTHON_CMD=python"
) else (
    where py >NUL 2>NUL
    if %ERRORLEVEL%==0 (
        set "PYTHON=py"
        set "PYTHON_CMD=py -3.11"
    ) else (
        echo Python 3.11 is not installed.
        echo Install it with "winget install -e --id Python.Python.3.11" ^
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
if %MAJOR%==3 if %MINOR% LSS 11 goto needpython

if not exist .venv (
    %PYTHON_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat
set PYTHON=python
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install .

call "%~f0" %*
goto :eof

:run
python copernican.py %*

:needpython
echo Python 3.11 or newer is required.
echo Install it with "winget install -e --id Python.Python.3.11" ^
or visit https://www.python.org/downloads/
exit /b 1

