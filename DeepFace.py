
import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy

path = 'datasets/img_align_celeba'

tran = transforms.Compose([transforms.Resize((64, 64)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = dsets.ImageFolder(root = path, transform = tran)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = 128,
                                             shuffle = True, num_workers = 2)


num_gpu = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

image_batch = next(iter(dataset_loader))
%matplotlib auto
plt.figure(figsize = (8, 8))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(image_batch[0].to(device)[:64],
                                                    padding = 2, normalize = True).cpu(),
                        (1, 2, 0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ) 

    def forward(self, input):
        return self.main(input)
    

NetG = Generator(num_gpu).to(device)
NetG.apply(weights_init)
print(NetG)

class Discriminator(nn.Module):
    def __init__(self, num_gpu):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

NetD = Discriminator(num_gpu).to(device)
NetD.apply(weights_init)

criterion = nn.BCELoss()
noice = torch.randn(64, 100, 1, 1, device = device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(NetD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(NetG.parameters(), lr = 0.0002, betas = (0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(25):
    for i, data in enumerate(dataset_loader, 0):
        NetD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = NetD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = NetG(noise)
        label.fill_(fake_label)
        output = NetD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        NetG.zero_grad()
        label.fill_(real_label) 
        output = NetD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch + 1, 25, i, len(dataset_loader), errD.item(), errG.item()))
        if i % 100 == 0:
            torchvision.utils.save_image(real_cpu, '%s/real_samples.png' % "datasets/img_align_celeba/Real/", normalize = True)
            fake = NetG(noise)
            torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("datasets/img_align_celeba/Generated/", epoch), normalize = True)
            
        
            
# Saving the model parameters.
save_path_D = 'datasets/img_align_celeba/model_D.pth'
save_path_G = 'datasets/img_align_celeba/model_G.pth'            
            
torch.save(NetD.state_dict(), save_path_D)
torch.save(NetG.state_dict(), save_path_G)            
            
            
            
            
