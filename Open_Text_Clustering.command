#!/bin/zsh
# Double-click this file in Finder to launch the program.
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"
exec python3 "$APP_DIR/main.py"
