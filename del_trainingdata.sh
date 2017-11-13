#!/bin/bash

echo delete annotations...

rm Data/trainData/annotations/training/*

rm Data/trainData/annotations/validation/*

echo delete images...

rm Data/trainData/images/training/*

rm Data/trainData/images/validation/*

echo delete *.pickle...

rm Data/*.pickle

echo delete completed
