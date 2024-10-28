# Author: jkd
# refer my blog on VAE at https://jitendradhiman.github.io/jkdblog/
#%%
import  torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.datasets import load_digits
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs: {n_gpus}")
#%% Allows accessing dictionary keys as attributes
class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
#%% config
config = {'visualize':False, 'batch_size': 32, 'lr':1e-3, 'latent_dim':2, 'cell_dim':16, "use_rep":True,
          'num_epochs':50, 'max_patience':10, "loss_period":20, "dec_distribution":"cat", "add_noise":False}
config = AttributeDict(config)
#%% Hyperparameters and constants
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1e-12
D = 64 # input_dim for 8x8 mnist digit after flattening
M = config.cell_dim # hidden units
L = config.latent_dim # latent variable dimension
num_classes = 17 # each pixel in the digit-image can take values from 0, ..., 16

# %% mnist small
class Digits(Dataset):
    """Scikit-Learn Digits dataset."""
    def __init__(self, mode='train', train_split=0.5, transforms=None):
        digits = load_digits()
        N = len(digits.images)
#         n_train = int(N * train_split)
#         val_split = (1 - train_split) / 2
#         n_val = n_train + int(N * val_split)
        n_train, n_val = 500, 850
        if mode == 'train':
            self.data = digits.data[:n_train].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[n_train:n_val].astype(np.float32)
        else:
            self.data = digits.data[n_val:].astype(np.float32)
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
#%% visualize a few digits
def VisualizeDigits(**kwargs):
    if kwargs["visualize"]:
        digits = load_digits()
        print(f"Shape of each digit-image: {digits.images[0].shape}")
        print(f"Number of images: {len(digits.images)}")
        print('Visualizing a few digits ->') 
        fig, axes = plt.subplots(2, 10, figsize=(16, 6))
        for i in range(20):
            # axes[i//10, i%10].imshow(digits.images[i], cmap='gray', interpolation='nearest')
            axes[i//10, i%10].imshow(digits.images[i], cmap='gray')
            axes[i//10, i%10].axis('off')
            axes[i//10, i%10].set_title(f"target:{digits.target[i]}")
        plt.tight_layout()
VisualizeDigits(**config)
train_data, val_data, test_data = Digits(mode='train'), Digits(mode='val'), Digits(mode='test')
n_train, n_val, n_test = train_data.__dict__['data'].shape[0], val_data.__dict__['data'].shape[0], test_data.__dict__['data'].shape[0]
print(f"n_train, n_val, n_test, total: {n_train}, {n_val}, {n_test}, {n_train + n_val + n_test}")

#%% data loaders
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
#%% dataset instance from torch loader
x_sample = next(iter(train_loader))
print(f"Input dimension: {x_sample.shape[1]}")
# %% Results dir 
results_dir = 'vae_results/'
results_dir = os.path.expanduser(os.getcwd() + '/' + results_dir)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
name = 'vae'
#%% encoder and decoder nets 
encoder_net = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(), nn.Linear(M, M), nn.LeakyReLU(), nn.Linear(M, 2 * L))
decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(), nn.Linear(M, M), nn.LeakyReLU(), nn.Linear(M, D * num_classes))
# encoder_net = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(), nn.Linear(M, 2 * L))
# decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(), nn.Linear(M, D * num_classes))
#%% log normal diagonal distribution
def log_normal_diagonal(z, mu, log_var):
    L = z.shape[1]
    log_p = -0.5 * L * torch.log(2 * PI) -0.5 * torch.sum(log_var, dim=-1) - 0.5 * torch.sum((z - mu)**2 / torch.exp(log_var), dim=1)
    return log_p 
def log_normal_diag_v1(z, mu, log_var):
    """" Uses in-built function, just for cross verification of the custum implementation
    """
    var = torch.exp(log_var) 
    b = var.shape[0]
    cov_mat = []
    for i in range(b):
        cov_mat.append(torch.diag(var[i]))
    cov_mat = torch.vstack(cov_mat).view(b, var.shape[1], -1)
    dist = torch.distributions.MultivariateNormal(mu, cov_mat)
    log_p = dist.log_prob(z)
    return log_p
# Test log_normal_diagonal 
# z_t = torch.Tensor([[2., 4, 3.], [3., 5, 1.]])
# mu_t = torch.randn_like(z_t)
# logvar_t = torch.randn_like(z_t)
# log_p1 = log_normal_diagonal(z_t, mu_t, logvar_t)
# log_p2 = log_normal_diag_v1(z_t, mu_t, logvar_t)
# print(log_p1, log_p2)
#%% log categorical 
def log_cat(x, p, num_classes):
    """ x : [B, D]
        p: [B, D, num_classes]
    """
    x = F.one_hot(x.long(), num_classes) # [B, D, num_classes]
    temp = x * torch.log(torch.clamp(p, EPS, 1-EPS)) # clamp it to avoid log(0)
    log_p = torch.sum(temp, dim=-1) # [B, D]
    log_p = torch.sum(log_p, dim=-1) # since x is a D-dim vector
    return log_p # [B]
# Test log_cat
# p = F.softmax(torch.rand((x_sample.shape[0], x_sample.shape[1], 17)), dim=-1)
# log_p = log_cat(x_sample, p, 17)
# print(log_p.shape)
# print(log_p)
#%% log normal standard 
def log_normal_standard(z):
    """ z: [B, L]
        mu, log_var: [B, L]
    """
    log_p = -0.5 * z.shape[1] * torch.log(2 * PI) - 0.5 * torch.sum(z**2, dim=-1)
    return log_p  # [B]
# Test log_normal_standard
# z_t = torch.Tensor([[2., 4, 3.], [3., 5, 1.]])
# log_p = log_normal_standard(z_t)
# print(log_p)
#%% Enocder 
class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.encoder = encoder_net 
        self.use_rep = config.use_rep
    def encode(self, x):
        assert not torch.isnan(x).any(), "x contains NaN!"
        h = self.encoder(x) # B x 2L 
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var
    def sample(self, x):
        """ x: B x D
        """
        mu, log_var = self.encode(x) # B x L, each row contains the values of L-dimensional latent variable z
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if self.use_rep:
            z = mu + std * eps
        else:
            z = torch.normal(mu, std)
        return z, mu, log_var # B x L
    def forward(self, x):
        # sample z using reparameterization
        z, mu, log_var = self.sample(x)
        # log-normal distribution P(z|x) ~ Normal(z|mu, var)
        log_p = log_normal_diagonal(z, mu, log_var) # [batch_size]
        return z, log_p # [batch_size]
# Test encoder 
# encoder = Encoder(encoder_net)
# z, out = encoder(x_sample)
# print(out.shape)

# %% Decoder 
class Decoder(nn.Module):
    def __init__(self, decoder_net, num_classes, distribution='cat'):
        super(Decoder, self).__init__()
        self.decoder = decoder_net
        self.num_classes = num_classes
        self.distribution = distribution
    def decode(self, z):
        B = z.shape[0]
        h = self.decoder(z) # [B, D*num_classes]
        if self.distribution == 'cat':
            h =  h.view(B, -1, self.num_classes)
            dec_p = torch.softmax(h, dim=-1) # [B, D, num_classes]
        return dec_p # [B, D, num_classes]
    def sample(self):
        pass
    def forward(self, x, z):
        dec_p = self.decode(z)
        log_p = log_cat(x, dec_p, self.num_classes)
        return log_p # [B]
# Test decoder
# encoder = Encoder(encoder_net) 
# decoder = Decoder(decoder_net, num_classes)
# z_sample, _, _ = encoder.sample(x_sample)
# out = decoder(x_sample, z_sample)
# print(out.shape)

# %% Prior
class Prior(nn.Module):
    def __init__(self):
        super(Prior, self).__init__()
    def sample(self):
        pass
    def forward(self, z):
        log_p = log_normal_standard(z)
        return log_p # [B]
# Test Prior 
# z_sample = Encoder(encoder_net)(x_sample)
# print(Prior()(z).shape)

# %% VAE
class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_classes):
        super(VAE, self).__init__()
        self.encoder =  Encoder(encoder_net)
        self.decoder = Decoder(decoder_net, num_classes)
        self.num_classes = num_classes
        self.prior = Prior()
    def loss(self, enc_log_p, dec_log_p, prior_log_p):
        # ELBO
        # ln P(x) >= Ez~q(z|x)[ln P(x|z)] - Ez~q(z|x)[ln P(z|x)] + Eq~q(z|x)[ln P(z)]
        #          = dec_log_prob + enc_log_prob - prior_log_p
        elbo = dec_log_p + prior_log_p - enc_log_p
        return -torch.mean(elbo) # batch-wise avg negative log likelihood (nll)
    def sample(self):
        pass
    def forward(self, x):
        """x: [B, D]
        """
        z, enc_log_p = self.encoder(x)
        dec_log_p = self.decoder(x, z)
        prior_log_p = self.prior(z)
        return enc_log_p, dec_log_p, prior_log_p # [B], [B], [B]
# Test VAE 
# model_sample = VAE(encoder_net, decoder_net, num_classes)
# out = model_sample(x_sample)
# print(out[0].shape, out[1].shape, out[2].shape)

#%% Training 
model = VAE(encoder_net, decoder_net, num_classes)
if n_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
# optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=config.lr)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=config.lr)
def train(train_loader, model, optimizer):
    model.train()
    avg_loss, total_samples = 0.0, 0
    for b, x, in enumerate(train_loader):
        batch_size = x.shape[0]
        if config.add_noise == True:
            x = x + 0.9 * torch.rand_like(x)
        x = x.to(device)
        enc_log_p, dec_log_p, prior_log_p = model(x)
        assert not torch.isnan(enc_log_p).any()  or not torch.isnan(dec_log_p).any() or not torch.isnan(prior_log_p).any(), f"Nan found at batch {b}"
        if n_gpus > 1:
            b_loss_ = model.module.loss(enc_log_p, dec_log_p, prior_log_p)
        else:
            b_loss_ = model.loss(enc_log_p, dec_log_p, prior_log_p)
        avg_loss += (b_loss_.item() * batch_size)
        total_samples += batch_size
        optimizer.zero_grad()
        b_loss_.backward()
        optimizer.step()
    avg_loss /= total_samples
    return avg_loss 
def validate(val_loader, model):
    model.eval()
    avg_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for x in val_loader:
            batch_size = x.shape[0]
            x = x.to(device)
            enc_log_p, dec_log_p, prior_log_p = model(x)
            if n_gpus > 1:
                b_loss_ = model.module.loss(enc_log_p, dec_log_p, prior_log_p)
            else:
                b_loss_ = model.loss(enc_log_p, dec_log_p, prior_log_p)
            avg_loss += (b_loss_.item() * batch_size)
            total_samples += batch_size
    avg_loss /= total_samples
    return avg_loss
train_loss, val_loss = [], []
start = time.time()
for epoch in range(1, config.num_epochs):
    avg_batch_train_loss = train(train_loader, model, optimizer)
    avg_batch_val_loss = validate(val_loader, model)
    train_loss.append(avg_batch_train_loss)
    val_loss.append(avg_batch_val_loss)
    if epoch % config.loss_period == 0:
        print(f"Epoch: {epoch}, Train Loss: {avg_batch_train_loss:0.4f}, Val Loss: {avg_batch_val_loss:0.4f}")
print(f"Time take: {time.time() - start} Second")
# %% plots
plt.figure(1)
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
# %%
