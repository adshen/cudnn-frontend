#!/bin/bash
set -e
clang-format --version
black --version

if [[ "$1" == "--apply" ]]; then
    find include/ samples/ python/ -regex '.*\.\(cpp\|h\)$' | xargs clang-format -i
    find test/ python/ -regex '.*\.py$' | xargs black --line-length 160
    find ./ -maxdepth 1 -regex '.*\.py$' | xargs black --line-length 160
else
    find include/ samples/ python/ -regex '.*\.\(cpp\|h\)$' | xargs clang-format --dry-run -Werror
    find test/ python/ -regex '.*\.py$' | xargs black --check --line-length 160
    find ./ -maxdepth 1 -regex '.*\.py$' | xargs black --check --line-length 160
fi
