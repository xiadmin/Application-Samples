#!/bin/bash

cd ~/Downloads

git clone https://github.com/amaork/libi2c.git
cd ./libi2c
#Checkout to specific commit - newest commit at the time of creation
git checkout dcdf6c6

sudo python setup.py install
