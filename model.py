import os
from parameters import BATCH_SIZE, N_WORKERS, LR, BETA1, BETA2, X_DATASET, Y_DATASET, EPOCHS, IMG_SIZE
from dataset import Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import ToPILImage
from torch.nn import L1Loss, SmoothL1Loss   
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

import numpy as np
import pickle

from PIL import Image
import matplotlib.pyplot as plt
import imageio

class Discriminator(nn.Module):
    def __init__(self,conv_dim=32):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, conv_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim, conv_dim*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*2, conv_dim*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*4, conv_dim*8, 4, padding=1),
            nn.InstanceNorm2d(conv_dim*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim*8, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.main(x)
    
class Generator(nn.Module):
    def __init__(self, conv_dim=64, n_res_block=9):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(conv_dim, conv_dim*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim*2, conv_dim*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*4),
            nn.ReLU(inplace=True),

            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),

            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_dim, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
class CycleGAN:

    def __init__(self, g_conv_dim=128, d_conv_dim=128, n_res_block=12, G_X2Y=False, G_Y2X=False, D_X=False, D_Y=False, norm='L2', dataset_path='cezanne2photo'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.norm = norm
        self.dataset_path = dataset_path
        self.lr = LR
        self.beta1 = BETA1
        self.beta2 = BETA2

        self.G_XtoY = Generator(conv_dim=g_conv_dim, n_res_block=n_res_block).to(self.device)
        if G_X2Y:
            try:
                self.G_XtoY.load_state_dict(torch.load((f'results/{self.dataset_path}/loss_{self.norm}/weights/G_X2Y.pt')))
            except Exception as E:
                pass
        self.G_XtoY = DataParallel(self.G_XtoY, device_ids=[0, 1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G_XtoY = self.G_XtoY.to(device)

        self.G_YtoX = Generator(conv_dim=g_conv_dim, n_res_block=n_res_block).to(self.device)
        if G_Y2X:
            try:
                self.G_YtoX.load_state_dict(torch.load((f'results/{self.dataset_path}/loss_{self.norm}/weights/G_Y2X.pt')))
            except Exception as E:
                pass
        self.G_YtoX = DataParallel(self.G_YtoX, device_ids=[0, 1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G_YtoX = self.G_YtoX.to(device)

        self.D_X = Discriminator(conv_dim=d_conv_dim).to(self.device)
        if D_X:
            try:
                self.D_X.load_state_dict(torch.load((f'results/{self.dataset_path}/loss_{self.norm}/weights/D_X.pt')))  
            except Exception as E:
                pass
        self.D_X = DataParallel(self.D_X, device_ids=[0, 1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D_X = self.D_X.to(device)

        self.D_Y = Discriminator(conv_dim=d_conv_dim).to(self.device)
        if D_Y:
            try:
                self.D_Y.load_state_dict(torch.load((f'results/{self.dataset_path}/loss_{self.norm}/weights/D_Y.pt')))
            except Exception as E:
                pass
        self.D_Y = DataParallel(self.D_Y, device_ids=[0, 1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D_Y = self.D_Y.to(device)

        print("Models running of {}".format(self.device))

    def load_model(self, filename):
        save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
        return torch.load(save_filename)

    def real_mse_loss(self, D_out):
        if self.norm == 'L2':
            return torch.mean((D_out-1)**2)
        elif self.norm == 'L1':
            return L1Loss()(torch.sigmoid(D_out), torch.ones_like(D_out))
        else:
            return SmoothL1Loss()(torch.sigmoid(D_out), torch.ones_like(D_out))

    def fake_mse_loss(self, D_out):
        if self.norm == 'L2':
            return torch.mean(D_out**2)
        elif self.norm == 'L1':
            return L1Loss()(D_out, torch.zeros_like(D_out))
        else:
            return SmoothL1Loss()(D_out, torch.zeros_like(D_out))

    def cycle_consistency_loss(self, real_img, reconstructed_img, lambda_weight):
        if self.norm == 'L2':
            reconstr_loss = torch.mean(torch.abs(real_img - reconstructed_img))
        elif self.norm == 'L1':
            reconstr_loss = L1Loss()(real_img, reconstructed_img)
        else:
            reconstr_loss = SmoothL1Loss()(real_img, reconstructed_img)
        return lambda_weight*reconstr_loss    

    def train_generator(self, optimizers, images_x, images_y):
        # Generator YtoX
        optimizers["g_optim"].zero_grad()

        fake_images_x = self.G_YtoX(images_y)

        d_real_x = self.D_X(fake_images_x)
        g_YtoX_loss = self.real_mse_loss(d_real_x)

        recon_y = self.G_XtoY(fake_images_x)
        recon_y_loss = self.cycle_consistency_loss(images_y, recon_y, lambda_weight=10)


        # Generator XtoY
        fake_images_y = self.G_XtoY(images_x)

        d_real_y = self.D_Y(fake_images_y)
        g_XtoY_loss = self.real_mse_loss(d_real_y)

        recon_x = self.G_YtoX(fake_images_y)
        recon_x_loss = self.cycle_consistency_loss(images_x, recon_x, lambda_weight=10)

        g_total_loss = g_YtoX_loss + g_XtoY_loss + recon_y_loss + recon_x_loss
        g_total_loss.backward()
        optimizers["g_optim"].step()

        return g_total_loss.item()
 
    def train_discriminator(self, optimizers, images_x, images_y):
        # Discriminator x
        optimizers["d_x_optim"].zero_grad()

        d_real_x = self.D_X(images_x)
        d_real_loss_x = self.real_mse_loss(d_real_x)
        
        fake_images_x = self.G_YtoX(images_y)

        d_fake_x = self.D_X(fake_images_x)
        d_fake_loss_x = self.fake_mse_loss(d_fake_x)
        
        d_x_loss = d_real_loss_x + d_fake_loss_x
        d_x_loss.backward()
        optimizers["d_x_optim"].step()


        # Discriminator y
        optimizers["d_y_optim"].zero_grad()
            
        d_real_y = self.D_Y(images_y)
        d_real_loss_x = self.real_mse_loss(d_real_y)
    
        fake_images_y = self.G_XtoY(images_x)

        d_fake_y = self.D_Y(fake_images_y)
        d_fake_loss_y = self.fake_mse_loss(d_fake_y)

        d_y_loss = d_real_loss_x + d_fake_loss_y
        d_y_loss.backward()
        optimizers["d_y_optim"].step()

        return d_x_loss.item(), d_y_loss.item()

    def train(self, print_every=10, nb_epoch=EPOCHS, verbose=True):

        # Dataset
        x_dataset = Dataset("trainA", self.dataset_path)
        y_dataset = Dataset("trainB", self.dataset_path)

        # Dataloader
        data_loader_x = DataLoader(x_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
        data_loader_y = DataLoader(y_dataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

        # Optimizer
        g_params = list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())
        optimizers = {
            "g_optim": Adam(g_params, self.lr, [self.beta1, self.beta2]),
            "d_x_optim": Adam(self.D_X.parameters(), self.lr, [self.beta1, self.beta2 ]),
            "d_y_optim": Adam(self.D_Y.parameters(), self.lr, [self.beta1, self.beta2 ])
        }

        losses = []
        g_total_loss_min = np.Inf

        if verbose:
            print('Running on {}'.format(self.device))
        for epoch in range(nb_epoch):
            if verbose:
                print("Epoch: {} en traitement".format(epoch))
            for (images_x, images_y) in zip(data_loader_x, data_loader_y):
                images_x, images_y = images_x.to(self.device), images_y.to(self.device)
                
                g_total_loss = self.train_generator(optimizers, images_x, images_y)
                d_x_loss, d_y_loss = self.train_discriminator(optimizers, images_x, images_y)
                
            
            if epoch % print_every == 0:
                losses.append((d_x_loss, d_y_loss, g_total_loss))
                if verbose:
                    print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'
                    .format(
                        epoch, 
                        nb_epoch, 
                        d_x_loss, 
                        d_y_loss, 
                        g_total_loss
                    ))
                
            if g_total_loss < g_total_loss_min:
                g_total_loss_min = g_total_loss
                
                torch.save(self.G_XtoY.state_dict(), f"results/{self.dataset_path}/loss_{self.norm}/weights/G_X2Y.pt")
                torch.save(self.G_YtoX.state_dict(), f"results/{self.dataset_path}/loss_{self.norm}/weights/G_Y2X.pt")
                
                torch.save(self.D_X.state_dict(), f"results/{self.dataset_path}/loss_{self.norm}/weights/D_X.pt")
                torch.save(self.D_Y.state_dict(), f"results/{self.dataset_path}/loss_{self.norm}/weights/D_Y.pt")
                
                if verbose:
                    print("Models Saved")
                
        # Save losses
        with open(f'results/{self.dataset_path}/loss_{self.norm}/losses_{self.norm}.pkl', 'wb') as fichier:
            pickle.dump(losses, fichier)
            fichier.close()

        self.lr = optimizers["g_optim"].param_groups[0]['lr']
        self.beta1 = optimizers["g_optim"].param_groups[0]['betas'][0]
        self.beta2 = optimizers["g_optim"].param_groups[0]['betas'][1]

        return losses
    
    def val(self, epoch=None):
        # Test Dataset
        x_dataset = Dataset("testA", self.dataset_path)
        y_dataset = Dataset("testB", self.dataset_path)

        samples = []

        for i in range(10):
            # idx = np.random.randint(len(y_dataset))
            idx = 30 + i
            fixed_y = y_dataset.get(idx)
            fixed_y_normalized = y_dataset[idx]
            fake_x = self.G_YtoX(torch.unsqueeze(fixed_y_normalized, dim=0))
            samples.append([fixed_y, torch.squeeze(fake_x, 0)])

        for i in range(10):
            # idx = np.random.randint(len(x_dataset))
            idx = 30 + i
            fixed_x = x_dataset.get(idx)
            fixed_x_normalized = x_dataset[idx]
            fake_y = self.G_XtoY(torch.unsqueeze(fixed_x_normalized, dim=0))
            samples.append([fixed_x, torch.squeeze(fake_y, 0)])


        for i, img in enumerate(samples):
            # Convertir les images en objets PIL
            original, fake = img
            original = ToPILImage()(original).convert("RGB")
            fake = ToPILImage()(fake).convert("RGB")

            # Créer une nouvelle image qui contient les deux images côte à côte
            new_image = Image.new('RGB', (IMG_SIZE * 2, IMG_SIZE))
            new_image.paste(original, (0, 0))
            new_image.paste(fake, (IMG_SIZE, 0))

            # Afficher l'image
            plt.figure()
            plt.imshow(new_image, interpolation='nearest')
            plt.axis('off')

            # Sauvegarder l'image
            if epoch is None:
                filename = f'results/{self.dataset_path}/loss_{self.norm}/exp/result_{i}.png'
            else:
                filename = f'results/{self.dataset_path}/loss_{self.norm}/exp/result_{i}_epoch_{epoch}.png'
            plt.savefig(filename)
            plt.close()

    def make_gif(self):
        # Train-Val-GIF
        self.val(epoch=0)

        for epoch in range(1, EPOCHS):
            # Train-Val
            print("Epoch: {} en traitement pour la norme {}".format(epoch, self.norm))
            self.train(print_every=100, nb_epoch=1, verbose=False)
            self.val(epoch=epoch)

        # Create GIF
        for i in range(20):
            with imageio.get_writer(f'results/{self.dataset_path}/loss_{self.norm}/exp/result_{i}.gif', mode='I', fps=2) as writer:
                for epoch in range(EPOCHS):
                    writer.append_data(imageio.imread(f'results/{self.dataset_path}/loss_{self.norm}/exp/result_{i}_epoch_{epoch}.png'))
                    os.remove(f'results/{self.dataset_path}/loss_{self.norm}/exp/result_{i}_epoch_{epoch}.png')
        return