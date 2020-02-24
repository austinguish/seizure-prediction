import os
import mne.io
import numpy as np
import pandas as pd
from mne.io import RawArray, read_raw_edf
import hickle
edf_dir = "/media/austinguish/USB_DISK/Chbmit/"  # edf file directory
summary_Text_Dir = "/media/austinguish/USB_DISK/Chbmit/"
save_dir = "/media/austinguish/USB_DISK/dataset/"  # npy file saving directory
preictal_dir = "/media/austinguish/USB_DISK/dataset/preictal/"
interictal_dir = "/media/austinguish/USB_DISK/dataset/interictal/"


def search(filename,filelist):
    for i in range(len(filelist)):
        if filelist[i][1]==filename:
            return [i,filelist[i][3]]
    return False


def find_rear(current_file_name,file_list,end,duration):
    total_time=duration-end
    file_pointer=current_file_name
    while(1):
        next_filename=file_pointer[0:6]+strcv(int(file_pointer[6:8])+1)+".edf.npy"
        next_next_filename=file_pointer[0:6]+strcv(int(file_pointer[6:8])+2)+".edf.npy"
        if not (search(next_next_filename,file_list)):
            return False

        total_time+=int(search(next_filename,file_list)[1])
        if total_time>14400:
            return [str(14400-(duration-end)), next_filename]
        rest_time=14400-total_time
        if rest_time<search(next_next_filename,file_list)[1]:
            return [str(rest_time), next_next_filename]
        else:
            file_pointer=next_filename
            continue


def find_front(current_file_name,detail,current_start,duration):
    total_time = current_start
    file_pointer = current_file_name

    while (1):
        front_filename = file_pointer[0:6] + strcv(int(file_pointer[6:8]) -1) + ".edf.npy"
        front_front_filename = file_pointer[0:6] + strcv(int(file_pointer[6:8]) -2) + ".edf.npy"
        if not (search(front_filename, detail)):
            return False
        if not (search(front_front_filename, detail)):
            return False
        total_time += int(search(front_filename, detail)[1])
        if total_time>14400:
            return [str(total_time-14400), front_filename]

        rest_time = 14400 - total_time
        if (rest_time < search(front_front_filename,detail)[1]):
            break
        else:
            file_pointer = front_filename
            continue
    return [str(rest_time), file_pointer]


def strcv(i):
    if i < 10:
        return '0' + str(i)
    elif i < 100:
        return str(i)


def save_signal(target):
    file_dir = edf_dir + target[0:5] + '/' + target
    print(file_dir)
    rawEEG = read_raw_edf(file_dir, verbose=0, preload=True)
    tmp = rawEEG.to_data_frame().values
    hickle.dump(tmp, save_dir + target + ".npy", mode='w', compression='gzip')
    print("finished")
    del tmp
    return (save_dir + target + ".npy")


def preictal_label_wrapper():
    '''
    This part uses the "summary.csv" which was extracted from the official summary.txt
    It includes the patient id, seizure_st_time,seizure_sp_time.As the definition in the paper
    the preictal period is defined as 1h before the seizure onset, and the interictal period is
    defined as periods as being between at least 4 h before seizure onset and 4 h after seizure end.
    '''
    cursor = pd.read_csv("/media/austinguish/USB_DISK/Chbmit/seizure_summary.csv", header=0)
    tmp = cursor.values
    for i in range(tmp.shape[0]):
        if (tmp[i][0][3:5] not in ["01", "03", "07",  "10", "20", "21", "22"]) or tmp[i][0][6:8]=="01":
            continue
        print("当前处理的文件是"+tmp[i][0])
        pres_no = int(tmp[i][0][6:8])
        prev_no = int(tmp[i-1][0][6:8])
        prev_time = int(tmp[i-1][2])
        pres_time = int(tmp[i][1])
        prev_file = save_dir + tmp[i][0][0:6] + strcv(pres_no - 1) + ".edf.npy"
        pres_file=save_dir + tmp[i][0] + ".npy"
        if (i != 0) and ((pres_no - prev_no) == 1):
            print("当前处理的文件相邻的有发作记录")
            buffer_array1 = hickle.load(save_dir + tmp[i-1][0] + ".npy")
            print("载入了"+save_dir + tmp[i-1][0] + ".npy")
            buffer_array2 = hickle.load(save_dir + tmp[i][0] + ".npy")
            print("载入了" + save_dir + tmp[i][0] + ".npy")
            bf1_length = buffer_array1.shape[0]
            bf2_length = buffer_array2.shape[0]

            # when seizure duration=1,check the preictal time length
            if (pres_time + bf1_length / 256 - prev_time) >= 3600:
                print("前一个文件的长度" + str(buffer_array1.shape))
                print("当前文件的长度" + str(buffer_array2.shape))
                if(pres_time<3600):
                    preictal_sample = np.concatenate(
                    (buffer_array1[-(3600 - pres_time) * 256:], buffer_array2[0:pres_time * 256]), axis=0)
                else:
                    preictal_sample=buffer_array2[0:921600]
                print(preictal_sample.shape)
                preictal_sample = preictal_sample.reshape(720, 21, -1)  # 5sec*256hz as a segment
                hickle.dump(preictal_sample, preictal_dir + "preictal" + str(i) + ".npy", mode='w', compression='gzip')
                print("finished")
                del preictal_sample
            else:
                print("发作相邻太短")
                continue
        elif (os.path.exists(prev_file)) :
            if (pres_time + hickle.load(save_dir + tmp[i][0] + ".npy").shape[0] / 256 - prev_time) >= 3600:
                pres_time = int(tmp[i][1])
                buffer_array1 = hickle.load(prev_file)
                print("前一个文件的长度"+str(buffer_array1.shape))
                buffer_array2 = hickle.load(pres_file)
                print("当前文件的长度"+str(buffer_array2.shape))
                if(pres_time<3600):
                    preictal_sample = np.concatenate(
                    (buffer_array1[-(3600 - pres_time) * 256:], buffer_array2[0:pres_time * 256]), axis=0)
                else:
                    preictal_sample=buffer_array2[0:921600]
                print(preictal_sample.shape)
                preictal_sample = preictal_sample.reshape(720, 21, -1)  # 5sec*256hz as a segment
                hickle.dump(preictal_sample, preictal_dir+"preictal" + str(i) + ".npy", mode='w', compression='gzip')
                print("finished")
                del preictal_sample
            else:
                continue
        elif pres_time > 3600:
            buffer_array= hickle.load(pres_file)
            preictal_sample = buffer_array[0:921600]
            print(preictal_sample.shape)
            preictal_sample = preictal_sample.reshape(720, 21, -1)  # 5sec*256hz as a segment
            hickle.dump(preictal_sample, preictal_dir + "preictal" + str(i) + ".npy", mode='w', compression='gzip')
            print("finished")
            del preictal_sample

        else:
            continue


def interictal_label_wrapper():
    cursor = pd.read_csv("/media/austinguish/USB_DISK/Chbmit/seizure_summary1.csv", header=0)
    tmp = cursor.values
    detail=pd.read_csv("./filedetail.csv")
    detail=detail.values
    tmp_info=['3036', 'chb01_07.edf.npy']
    for i in range(0, tmp.shape[0]):
        current_file_name=tmp[i][0]+".npy"
        current_start=int(tmp[i][1])
        current_end=(tmp[i][2])
        if (current_file_name[3:5] not in ["01", "03", "07",  "10", "20", "21", "22"]) or current_file_name[6:8] == "01":
            continue
        print("当前文件为：")
        print(current_file_name)
        duration=search(current_file_name,detail)[1]
        rear_info=find_rear(current_file_name, detail,current_end,duration)
        front_info = find_front(current_file_name, detail, current_start, duration)
        if front_info and tmp_info:
            if front_info[1][3:5]==tmp_info[1][3:5] and front_info[1][6:8]>tmp_info[1][6:8]:
                buf1=hickle.load(save_dir+tmp_info[1])[int(tmp_info[0])*256:]
                buf2=hickle.load(save_dir+front_info[1])[0:int(front_info[0])*256+1]
                total = buf1
                for k in range(int(tmp_info[1][6:8]),int(front_info[1][6:8])+1):
                    print("正在载入"+tmp_info[1][0:6]+strcv(k)+".edf.npy")
                    tmp_array=hickle.load(save_dir+tmp_info[1][0:6]+strcv(k)+".edf.npy")
                    total = np.concatenate((total, tmp_array), axis=0)
                total = np.concatenate((total,buf2),axis=0)
                print(total.shape)
                print("这个可以用，找到首尾了")
                cut_total = total[0:(total.shape[0]-(total.shape[0]%1280))]
                axis_0=int(cut_total.shape[0]/1280)
                del total
                cut_total=cut_total.reshape(axis_0,21,-1)
                hickle.dump(cut_total,interictal_dir+str(i)+".npy", mode='w', compression='gzip')
                del cut_total
                print("处理完啦，删光光咯")
        else:
            tmp_info = rear_info
            continue
        tmp_info=rear_info
        print("处理完成")


preictal_label_wrapper()
interictal_label_wrapper()

