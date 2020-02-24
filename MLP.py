from torch.utils.data import Dataset, DataLoader
import torch
import hickle
import pandas as pd
import numpy


class EEGTrainDataset(Dataset):
    def __init__(self, root, csv_file):
        self.file_Dir = root
        self.file_info = pd.read_csv(csv_file, header=0).values

    def __len__(self):
        return self.file_info.shape[0]

    def __getitem__(self, index):
        file_dir = self.file_info[index][1]
        label = self.file_info[index][2]
        if label == "Interictal":
            label = 0
        else:
            label = 1
        tmp = hickle.load(file_dir)
        tmp = tmp.reshape((1, 26880))  # convert to linear
        return torch.from_numpy(tmp).float(), torch.tensor([label]).float()


train_set = EEGTrainDataset(root="/home/austinguish/dataset", csv_file="/home/austinguish/dataset/TrainSet.csv")
train_data_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=2)
test_set = EEGTrainDataset(root="/home/austinguish/dataset", csv_file="/home/austinguish/dataset/TrainSet.csv")
test_data_loader = DataLoader(train_set, batch_size=50, shuffle=False, num_workers=2)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=torch.nn.Linear(26880, 300)
        self.fc2=torch.nn.Linear(300, 100)
        self.fc3=torch.nn.Linear(100, 50)
        self.fc4=torch.nn.Linear(50, 20)
        self.fc5=torch.nn.Linear(20, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        out = torch.sigmoid(self.fc5(x))
        return out


model = MLP().cuda()
print(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
lossfunc = torch.nn.BCELoss().cuda()


def accuracy_compute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (numpy.argmax(pred, 1) == label)
    test_np = numpy.float32(test_np)
    return numpy.mean(test_np)


for x in range(1):
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        (inputs, labels) = data
        inputs = torch.autograd.Variable(inputs).cuda()
        labels = torch.autograd.Variable(labels).cuda()
        outputs = model(inputs)
        outputs = outputs.reshape((1,1))
        outputs.cuda()
        loss = lossfunc(outputs, labels)
        if(i%100==0):
            print("label:",labels,"pred:",outputs,"loss:",loss)
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'params.pkl')
test_save_net = MLP().cuda()
test_save_net.load_state_dict(torch.load("params.pkl"))
accuracy_list = []
for i,data in enumerate(test_data_loader):
    (inputs, labels) = data
    inputs = torch.autograd.Variable(inputs).cuda()
    labels = torch.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    print("label:",labels,"pred:",outputs)
    accuracy_list.append(accuracy_compute(outputs,labels))
print(sum(accuracy_list) / len(accuracy_list))




