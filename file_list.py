import pandas as pd
import os
import numpy
file_dir="/home/austinguish/dataset"
file_list=os.listdir(file_dir)
print(file_list)
filename=[]
filetype=[]
for file in file_list:
    file=os.path.join(file_dir,file)
    if os.path.isfile(file) and ".npy" in file:
        filename.append(file)
filename = numpy.array(filename)
numpy.random.shuffle(filename)
train_file = filename[:int(filename.shape[0]*0.8)]
test_file = filename[int(filename.shape[0]*0.8):]
train_type = []
test_type=[]
for i in train_file:
    if "Pre" in i:
        train_type.append("Preictal")
    elif "Inter" in i:
        train_type.append("Interictal")
for j in test_file:
    if "Pre" in j:
        test_type.append("Preictal")
    elif "Inter" in j:
        test_type.append("Interictal")
dic1={"filename":pd.array(train_file),"filetype":pd.array(train_type)}
dic2={"filename":pd.array(test_file),"filetype":pd.array(test_type)}
pd1=pd.DataFrame(dic1)
pd1.to_csv("TrainSet.csv")
pd2=pd.DataFrame(dic2)
pd2.to_csv("TestSet.csv")