import numpy as np
import torch
import torch.utils.data
import dataLoader as dl

def get_data_loaders(X, Y, config):
    batchSize = config['batchSize']
    device = config['device']
    dtype = config['dtype']

    tensor_x = torch.stack([torch.tensor(i, dtype = dtype) for i in X])
    tensor_y = torch.stack([torch.tensor(i, dtype = dtype) for i in Y])

    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    
    data_set = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = batchSize,
            shuffle = config['shuffle'])
    return data_loader


def train(config, model, train_loader):
    if config['device'] == 'cuda':
        model.cuda()

    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.L1Loss()

    print("---- Starting training -----")
    model.train()
    for epoch in range(config['epochs']):
        for batch, labels in train_loader:
            y_pred = model(batch)
            loss = criterion(y_pred, labels)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("---- Finished training ----")

    return model, criterion

def test(config, model, criterion):
    print("----- Evaluation -----")
    model.eval()
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(test_loader()):
            ypred = model(batch)
            print("batch :{} loss :{}".format(idx, criterion(ypred, labels).item()))

class network(torch.nn.Sequential):
    def __init__(self, config):
        super(network, self).__init__()
        self.dimensions = config['dims']
        self.nlayers = len(self.dimensions)

        for idx, (din, dout) in enumerate(zip(self.dimensions[:-1], self.dimensions[1:])):
            self.add_module(str(idx), torch.nn.Linear(din, dout))
            self.add_module("activ"+str(idx), config['activation']())

    def forward(self, x):
        ypred = super().forward(x)
        return ypred

if __name__ == '__main__':
    X, Y = dl.load()
    config = {
            'batchSize':100,
            'device': 'cpu',
            'dtype': torch.float,
            'shuffle': False,
            'dims': [len(X[0]), 100, 100, 1],
            'activation': torch.nn.ReLU,
            'op': torch.nn.Linear,
            'epochs': 30
            }
    trainLoader = get_data_loaders(X[:100], Y[:100], config)
    for x, y in trainLoader:
        assert len(x) == len(y)
    
    model = network(config)
    print(model.parameters)
    trainedModel = train(config, model, trainLoader)
