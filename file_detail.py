import os
import pandas as pd
import numpy
import hickle
file_dir="/media/austinguish/USB_DISK/dataset"
file_list=os.listdir(file_dir)
print(file_list)
filename=[]
filesize=[]
filelength=[]
for file in file_list:
    if os.path.isfile(os.path.join(file_dir,file)):
        tmp=hickle.load(os.path.join(file_dir,file))
        filelength.append(tmp.shape[0])
        filename.append(file)
        filesize.append(int(os.path.getsize(os.path.join(file_dir,file))/1000000))
        del tmp
    else:
        continue
dic1={"filename":pd.array(filename),"filesize":pd.array(filesize),"filelength":pd.array(filelength)}

pd1=pd.DataFrame(dic1)
pd1.to_csv("filedetail.csv")
