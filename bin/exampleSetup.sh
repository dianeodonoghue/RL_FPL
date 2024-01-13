#! /bin/bash

# Install virtualenv
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m pip install --user virtualenv

# Create Env
python3 -m virtualenv .fpl

# Sign In
source .fpl/bin/activate

# Upgrade pip 
pip install --upgrade pip

# Requirements file
pip install -r requirements.txt

# Set FPL as pythonpath
#export PYTHONPATH="<Path to FPL folder>:$PYTHONPATH"


