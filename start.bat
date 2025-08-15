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
) else (
    where py >NUL 2>NUL
    if %ERRORLEVEL%==0 (
        set "PYTHON=py"
    ) else (
        echo Python is not installed.
        echo Install it with "winget install -e --id Python.Python.3" ^
or visit https://www.python.org/downloads/
        exit /b 1
    )
)

if not exist .venv (
    %PYTHON% -m venv .venv
)

call .venv\Scripts\activate.bat
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install .

call "%~f0" %*
goto :eof

:run
python copernican.py %*

