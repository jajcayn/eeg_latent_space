#!/bin/bash

# Script for local-remote synchronisation. It synchronizes "code", "tests", and
# "experiments" folder to remote machine

machine=$1
remoteRoot=$2

dest=$machine:$remoteRoot
echo "Syncing into $dest"

rsync -avzhP --delete \
--exclude='__pycache__' \
--exclude='.DS_Store' \
--exclude='.vscode' \
--exclude='.ipynb_checkpoints' \
--exclude='.unotes' \
--exclude='.venv' \
--exclude='.python-version' \
--exclude='.env' \
--exclude=".git" \
code $dest

rsync -avzhP --delete \
--exclude='__pycache__' \
--exclude='.DS_Store' \
--exclude='.vscode' \
--exclude='.ipynb_checkpoints' \
--exclude='.unotes' \
--exclude='.venv' \
--exclude='.python-version' \
--exclude=".git" \
experiments $dest

rsync -avzhP --delete \
--exclude='__pycache__' \
--exclude='.DS_Store' \
--exclude='.vscode' \
--exclude='.ipynb_checkpoints' \
--exclude='.unotes' \
--exclude='.venv' \
--exclude='.python-version' \
--exclude=".git" \
tests $dest

rsync -avzhP meta_runner.sh $dest
