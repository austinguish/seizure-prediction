import os
import mne.io
import numpy as np
import pandas as pd
from mne.io import RawArray, read_raw_edf
import hickle
edf_dir="/media/austinguish/USB_DISK/Chbmit/"#edf file directory
summary_Text_Dir="/media/austinguish/USB_DISK/Chbmit/"
file_Name_dir="/media/austinguish/USB_DISK/Chbmit/segmentation.csv"
save_dir="/media/austinguish/USB_DISK/dataset/"#npy file saving directory
def strcv(i):
        if i < 10:
            return '0' + str(i)
        elif i < 100:
            return str(i)


def save_signal(target):
    chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
           u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
           u'FT9-FT10', u'FT10-T8']
    file_dir=edf_dir+target[0:5]+'/'+target
    print(file_dir)
    rawEEG = read_raw_edf(file_dir,verbose=0,preload=True)
    rawEEG.pick_channels(chs)
    tmp=rawEEG.to_data_frame().values
    print(tmp.shape)
    hickle.dump(tmp, save_dir+target+".npy", mode='w', compression='gzip')
    print("finished")
    del tmp




a=pd.read_csv(file_Name_dir,header=0)
print(a.values.shape)
tmp=a.values
'''save_signal("chb01_01.edf")'''
for i in range(a.values.shape[0]):
    if (tmp[i][0][3:5] in["01"]):
        save_signal(tmp[i][0])
    else:
        continue
