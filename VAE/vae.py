import sys
sys.path.append('../DQN')
sys.path.append('../VAE')
sys.path.append('..')
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt 
from ..globals import indicators, mkt_indicators, oclhva_after, window_size, MAIN_PATH  #
from ..data_work import DataStorage
import pysnooper
from tqdm import tqdm

VAE_FILENAME = r'vae_stock.pth'
VAE_TRAIN_DATA = r"data.csv"


# 还是MAIN_PATH的问题
import os
MAIN_PATH = os.getcwd()

class VAE(nn.Module):
    def __init__(self, input_dim = 100):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc21 = nn.Linear(80, 1)
        self.fc22 = nn.Linear(80, 1)
        self.fc3 = nn.Linear(1, 80)
        self.fc4 = nn.Linear(80, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        # standardization
        std = logvar.mul(1.0e-5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # 生成一个标准正态分布，长度和std一样
        return eps.mul(std).add_(mu) # 直接返回 std.normal_().add_(mu)不就好了吗？为什么要有这个eps??
        # return torch.normal(mu,std)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid()
        # return torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def load_model(self):
        return self.load_state_dict(torch.load(MAIN_PATH+"/VAE/"+VAE_FILENAME))

    def save_model(self):
        return self.save_state_dict(torch.load(MAIN_PATH+"/VAE/"+VAE_FILENAME))

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    # print(recon_x.shape,x.shape,mu.shape, logvar.shape) # why recon_x.shape is (1,100) other than (1,400)?
    BCE = reconstruction_function(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


def embed(df:pd.DataFrame) -> tuple[int,int, int, list[torch.Tensor]]:
    # df = pd.read_csv(VAE_TRAIN_DATA) # v1 uses .csv data, overrides the v2. comment this line to activate v2
    # the following line overrides the definitions in globals.py, comment this line to activate v3
    oclhva_after, indicators = ['open_', 'close_', 'high_', 'low_', 'volume_'], ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24']  
    d = df[[*indicators]] #,*oclhva_after]] # oclhva_after数据存在较多的inf, nan，训练无法收敛
    # d[indicators] = d[indicators]/5-10 # already divided by 100, check preprocess.add_indicators()
    print(d.describe()) # 观察一下数据分布情况再说
    # d = df[[*indicators,*oclhva_after, *mkt_indicators, *mkt_oclhva_after]] # v4, to include market info 
    # 要不要在这里直接reshape一下？ # 注意这里必须是float32，默认是float64,会出错
    return  window_size , len(d.columns), len(d[window_size-1:]), \
            [torch.tensor(x.values,dtype=torch.float32) for x in d.rolling(window_size)][window_size-1:]

if __name__ == "__main__":
    MAIN_PATH = '../'     # keep the default dir as main path when importing other modules
    num_epochs = 10
    batch_size = 12
    learning_rate = 1e-3

    # dataloader
    df = DataStorage(MAIN_PATH).load_processed()    
    window_size, len_columns, len_data, train_data = embed(df)  # len_data not used coz enumerate(dataloader) does the same thing
    dataloader = DataLoader(train_data,batch_size=1, shuffle=True)

    # load model
    vae = VAE(window_size*len_columns)
    # vae.load_model() # load the previous model to continue training
    if torch.cuda.is_available(): vae.cuda()
    reconstruction_function = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # train
    episode = []
    ave_loss = []
    epbar = tqdm(range(num_epochs))
    for epoch in epbar:
        vae.train()
        train_loss = 0
        stepbar = tqdm(enumerate(dataloader))
        for batch_idx, data in stepbar: 
            data = data.view(data.size(0), -1) # torch.Size([1, 20, 5]) -> torch.Size([1, 100])
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data.item()

            optimizer.step()
            # stepbar.set_description("loss:"+str(loss.data.item() / len(data)))
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * len(data),
            #         len(dataloader.dataset),
            #         100. * batch_idx / len(dataloader),
            #         loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))
        # epbar.set_description("Epoch:{} Average loss:{:.4f}".format(epoch, train_loss / len(dataloader.dataset)))
        episode.append(epoch)
        ave_loss.append(train_loss / len(dataloader.dataset))
    torch.save(vae.state_dict(), VAE_FILENAME)
    plt.plot(episode,ave_loss)
    plt.show()