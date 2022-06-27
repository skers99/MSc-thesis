from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from snn_utils import spiketrains
class XorDataset(Dataset):

    def __init__(self,data,labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        X = self.data[index]
        y = self.labels[index]
        return X,y

def generate_xor_data(in_dim=10,time=100,samples_per_type=200):
    on_rate = time
    off_rate = 1
    on_rates = [on_rate] * int((in_dim/2))
    off_rates = [off_rate] * int((in_dim/2))
    oo = np.array([spiketrains(in_dim,time,off_rates + off_rates) for i in range(samples_per_type)])
    oi = np.array([spiketrains(in_dim,time,off_rates + on_rates) for i in range(samples_per_type)])
    io = np.array([spiketrains(in_dim,time,on_rates + off_rates) for i in range(samples_per_type)])
    ii = np.array([spiketrains(in_dim,time,on_rates + on_rates) for i in range(samples_per_type)])
    labels = np.array([0] * samples_per_type + [1] * samples_per_type * 2 + [0] * samples_per_type).reshape(samples_per_type * 4,)
    data = np.concatenate((oo,oi,io,ii))
    return data,labels

#data,labels = generate_xor_data(samples_per_type=100)
def dataloaders_from_data(data,labels,batch_size=40):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=.2,)
    data_train = XorDataset(X_train,y_train)
    data_test = XorDataset(X_test, y_test)

    dataloader_train = DataLoader(data_train, batch_size=batch_size,shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=batch_size,shuffle=True)
    return dataloader_train, dataloader_test

def xor_dataloaders(in_dim=10,time=100,samples_per_type=200,batch_size=40):
    data, labels = generate_xor_data(in_dim=in_dim,time=time,samples_per_type=samples_per_type)
    dataloader_train, dataloader_test = dataloaders_from_data(data=data,labels=labels,batch_size=batch_size)
    return dataloader_train,dataloader_test