from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import hickle
import pandas as pd
import numpy


def accuracy_compute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (numpy.argmax(pred, 1) == label)
    test_np = numpy.float32(test_np)
    return numpy.mean(test_np)


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
        tmp = torch.from_numpy(tmp)
        tmp = tmp.reshape((1,1280,21))
        return tmp.float(), torch.tensor([label]).float()


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1=torch.nn.Linear(5056, 300)
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


class DCNN(torch.nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 2))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 2))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,(3,2))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = nn.functional.max_pool2d(nn.functional.relu(x), (2, 2))
        x = self.conv2_bn(self.conv2(x))
        x = nn.functional.max_pool2d(nn.functional.relu(x), (2, 2))
        x = self.conv3_bn(self.conv3(x))
        x = nn.functional.max_pool2d(nn.functional.relu(x), (2, 2))
        x = self.conv4(x)
        return x


'''cnn_model = DCNN().cuda() # test code
mlp_model = MLP().cuda()
test = torch.rand((1,1,1280,21)).cuda()
out = cnn_model(test)
out = mlp_model(out.view(out.size(0),-1))
print(out)'''
train_set = EEGTrainDataset(root="/home/austinguish/dataset", csv_file="/home/austinguish/dataset/TrainSet.csv")
train_data_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=2)
test_set = EEGTrainDataset(root="/home/austinguish/dataset", csv_file="/home/austinguish/dataset/TestSet.csv")
test_data_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
cnn_model = DCNN().cuda()
mlp_model = MLP().cuda()
optimizer = torch.optim.RMSprop(mlp_model.parameters(), lr=0.01, alpha=0.9)
lossfunc = torch.nn.BCELoss().cuda()
for x in range(1):
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        (inputs, labels) = data
        inputs = torch.autograd.Variable(inputs).cuda()
        labels = torch.autograd.Variable(labels).cuda()
        outputs = cnn_model(inputs)
        outputs = mlp_model(outputs.view(outputs.size(0),-1))
        outputs.cuda()
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()
        print(i, ":", accuracy_compute(outputs, labels))


torch.save(cnn_model.state_dict(), 'cnnparams.pkl')
torch.save(mlp_model.state_dict(), 'mlpparams.pkl')
test_cnn_net = DCNN().cuda()
test_mlp_net = MLP().cuda()
test_cnn_net.load_state_dict(torch.load("cnnparams.pkl"))
test_mlp_net.load_state_dict(torch.load("mlpparams.pkl"))
accuracy_list = []
for i,data in enumerate(test_data_loader):
    (inputs, labels) = data
    inputs = torch.autograd.Variable(inputs).cuda()
    labels = torch.autograd.Variable(labels).cuda()
    outputs = test_cnn_net(inputs)
    outputs = test_mlp_net(outputs.view(outputs.size(0),-1))
    print(i,":","label:",labels,"pred:",outputs)
    accuracy_list.append(accuracy_compute(outputs, labels))



print(sum(accuracy_list))
print(len(accuracy_list))

