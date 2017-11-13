from __future__ import print_function
import read_SceneParsingData as scene_parsing
import os

data_dir =  os.getcwd() + "/Data/"
print(data_dir)
train_records, valid_records = scene_parsing.read_dataset(data_dir)
print(len(train_records))  # 0
print(len(valid_records))  # 0