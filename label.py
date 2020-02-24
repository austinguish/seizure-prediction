import hickle
preictal_dir = "/home/austinguish/dataset/Preictal_Dataset.npy"
interictal_dir="/home/austinguish/dataset/Interictal_Dataset.npy"
a = hickle.load(preictal_dir)
b = hickle.load(interictal_dir)
for i in range(a.shape[0]):
    hickle.dump(a[i], "Pre_"+str(i)+".npy", mode='w',compression='gzip')
    for j in range(b.shape[0]):
    hickle.dump(b[i], "Inter_"+str(i)+".npy", mode='w',compression='gzip')

