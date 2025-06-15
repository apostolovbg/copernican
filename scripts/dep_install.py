"""Automated dependency installer for the Copernican Suite."""
# DEV NOTE (v1.5d): Installs missing packages and restarts the suite.
import os
import subprocess
import sys


def main():
    packages = sys.argv[1:]
    if not packages:
        sys.exit(0)
    root_dir = os.path.dirname(os.path.dirname(__file__))
    for pkg in packages:
        subprocess.call([sys.executable, '-m', 'pip', 'install', pkg])
    copernican = os.path.join(root_dir, 'copernican.py')
    os.execv(sys.executable, [sys.executable, copernican])


if __name__ == '__main__':
    main()
