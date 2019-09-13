from modelNN import *
import dataLoader as dl

X, Y = dl.load()
config = {
        'batchSize':1,
        'device': 'cuda',
        'dtype' : torch.float,
        'shuffle': True,
        'dims' : [len(X[0]), 100, 50, 1],
        'activation' : torch.nn.ReLU,
        'op':  'none',
        'lr': .000201,
        'epochs': 200,
        'optim':torch.optim.SGD,
        'lossfn':torch.nn.L1Loss
        }

trainLoader = get_data_loaders(X[:100], Y[:100], config)
model = network(config)
trainedModel = train(config, model, trainLoader)
test(config, trainedModel, config['lossfn'](), trainLoader)



