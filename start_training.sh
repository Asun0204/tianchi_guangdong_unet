#!/bin/bash

rm nohup.out
rm logs/*

echo start training

nohup python unet.py &

echo start tensorboard

nohup tensorboard --logdir=logs >/dev/null 2>&1 &

echo training started
