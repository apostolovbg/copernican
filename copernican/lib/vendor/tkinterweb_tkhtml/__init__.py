"""
TkinterWeb-Tkhtml v2.0
This package provides pre-built binaries of a modified version of the Tkhtml3 widget from https://github.com/Andereoo/TkinterWeb-Tkhtml, 
which enables the display of styled HTML and CSS code in Tkinter applications.

If you are not using the compile script's --install option, to add a new Tkhtml version, use the following file naming conventions:
- For Tcl/Tk 8:
  - For a standard release: libTkhtml[major_version.minor_version].[dll/dylib/so] (eg. libTkhtml3.0.dll)
  - For an experimental release: libTkhtml[major_version.minor_version]exp.[dll/dylib/so] (eg. libTkhtml3.1exp.dll)
- For Tcl/Tk 9:
  - For a standard release: libTkhtml[major_version.minor_version]-TclTk9.[dll/dylib/so] (eg. libTkhtml3.0-TclTk9.dll)
  - For an experimental release: libTkhtml[major_version.minor_version]exp-TclTk9.[dll/dylib/so] (eg. libTkhtml3.1exp-TclTk9.dll)

This package can be used to load the Tkhtml widget into Tkinter applications.
but is mainly intended to be used through TkinterWeb, which provides a full Python interface. 
See https://github.com/Andereoo/TkinterWeb.

Copyright (c) 2025 Andrew Clarke
"""

import os

from tkinter import TclVersion

__title__ = 'TkinterWeb-Tkhtml'
__author__ = "Andrew Clarke"
__copyright__ = "Copyright (c) 2025 Andrew Clarke"
__license__ = "MIT"
__version__ = '2.0.0'


TKHTML_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tkhtml")

ALL_TKHTML_BINARIES =  [file for file in os.listdir(TKHTML_ROOT_DIR) if "libTkhtml" in file]

if TclVersion >= 9:
    TKHTML_BINARIES =  [file for file in ALL_TKHTML_BINARIES if "TclTk9" in file]
    HELP_MESSAGE_EXP = f"Download https://github.com/Andereoo/TkinterWeb-Tkhtml/tree/experimental and run 'python compile.py' to compile Tkhtml. \
Copy the binary into {TKHTML_ROOT_DIR}, adding 'exp-TclTk9' after the filename (eg. 'libTkhtml3.1exp-TclTk9.dll')"
else:
    TKHTML_BINARIES =  [file for file in ALL_TKHTML_BINARIES if "TclTk9" not in file]
    HELP_MESSAGE_EXP = f"Download https://github.com/Andereoo/TkinterWeb-Tkhtml/tree/experimental and run 'python compile.py' to compile Tkhtml. \
Copy the binary into {TKHTML_ROOT_DIR}, adding 'exp' after the filename (eg. 'libTkhtml3.1exp.dll')"

HELP_MESSAGE = f"Download https://github.com/Andereoo/TkinterWeb-Tkhtml/tree/3.1-TclTk9 and run 'python compile.py --install' to compile and install Tkhtml. If you think this is a bug, consider filing a bug report."


def get_tkhtml_file(version=None, index=-1, experimental=False):
    "Get the location of the platform's Tkhtml binary"
    if not TKHTML_BINARIES:
        if ALL_TKHTML_BINARIES:
            raise OSError(f"No Tkhtml versions could be found for Tcl/Tk {int(TclVersion)}. {HELP_MESSAGE}")
        else:
            raise OSError(f"No Tkhtml versions could be found for your system. {HELP_MESSAGE}")
    
    if isinstance(version, float):
        version = str(version)
    if version:
        for file in TKHTML_BINARIES:
            if version in file:
                # Note: experimental can be "auto"
                if "exp" in file:
                    if not experimental:
                        raise OSError(f"Tkhtml version {version} is an experimental release but experimental mode is disabled. {HELP_MESSAGE_EXP}")
                    experimental = True
                else:
                    if experimental == True:
                        raise OSError(f"Tkhtml version {version} is not an experimental release but experimental mode is enabled. {HELP_MESSAGE_EXP}")
                    experimental = False
                return os.path.join(TKHTML_ROOT_DIR, file), version, experimental
        raise OSError(f"Tkhtml version {version} either does not exist or is unsupported on your system. {HELP_MESSAGE}")
    else:
        # Get highest numbered avaliable file if a version is not provided
        if experimental == True:
            files = [k for k in TKHTML_BINARIES if 'exp' in k]
            if not files:
                raise OSError(f"No experimental Tkhtml versions could be found on your system. {HELP_MESSAGE_EXP}")
        elif not experimental:
            files = [k for k in TKHTML_BINARIES if 'exp' not in k]
        else:
            files = TKHTML_BINARIES
        file = sorted(files)[index]
        if "exp" in file:
            experimental = True
        else:
            experimental = False
        version = file.replace("libTkhtml", "").replace("exp", "")
        version = version[:version.rfind(".")]
        return os.path.join(TKHTML_ROOT_DIR, file), version, experimental


def get_loaded_tkhtml_version(master):
    """Get the version of the loaded Tkhtml instance.
    This will raise a TclError if Tkhtml is not loaded.
    Only call load_tkhtml_file or load_tkhtml if this fails or if you know this will fail."""
    return master.tk.call("package", "present", "Tkhtml")


def load_tkhtml_file(master, file):
    "Load Tkhtml into the current Tcl/Tk instance"
    if TKHTML_ROOT_DIR not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([
            TKHTML_ROOT_DIR,
            os.environ["PATH"]
        ])
    master.tk.call("load", file)


def load_tkhtml(master):
    "Load Tkhtml into the current Tcl/Tk instance"
    master.tk.call("package", "require", "Tkhtml")