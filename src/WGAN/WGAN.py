# %matplotlib inline
import datetime
import os
import random
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torcheval.metrics import BinaryAccuracy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Root directory for dataset
data_root = '../datasets/DSPS23 Pavement/Task 1 Crack Type/training_data/ts1/images/'

# Result directory
result_root = './src/WGAN/results/'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3 (RGB) For grayscale images this is 1
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.00005

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


def batch_visualizer(device, dataloader, number_of_images=64):
    """
    Usage: Visualizes a batch of images from the dataset
    :param device: cpu or gpu
    :param dataloader: pytorch dataloader object which has the dataset
    """
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:number_of_images], padding=2, normalize=True).cpu(),
                     (1, 2, 0)))
    plt.show()


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # the generator model architecture is defined here
        self.main = nn.Sequential(
            # input is Z, going into a convolution. nz is the vector size of the latent space for each image
            # ngf is the number of feature maps in the generator.
            # Feature maps are like channels in the image. So this number goes down as the network gets deeper
            # since we are reducing the number of channels to get to the number of channels in the image
            # nc is the number of channels in the final image
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 128x128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 256x256
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # the discriminator model architecture is defined here
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256 because our images are 256 * 256 * 1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 64x64
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 32x32
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 16x16
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),  # 64x64
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),  # 64x64
            nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class WGAN:
    def __init__(self, device, ngpu, nz=100):
        # Create the generator and initialize the weights
        self.netG = Generator(ngpu).to(device)
        self.netG.apply(self.weight_initializer)
        # Create the Discriminator and initialize the weights
        self.critic = Discriminator(ngpu).to(device)
        self.critic.apply(self.weight_initializer)
        # Initialize BCEWithLogitsLoss function for the GAN since the last layer of the discriminator is a linear layer
        self.criterion = nn.BCEWithLogitsLoss()

        # Latent vector to visualize the progression of the generator
        self.nz = nz
        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        # Mean loss calculation for each epoch
        self.G_losses_mean = None
        self.D_losses_mean = None

    def get_noise(self, n_samples, noise_dim, device='cpu'):
        '''
        Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
        where
            n_samples: the number of samples to generate based on  batch_size
            noise_dim: the dimension of the noise vector
            device: device type can be cuda or cpu
        '''

        return torch.randn(n_samples, noise_dim, 1, 1, device=device)

    def train(self, dataloader,
              device,
              nz,
              lr=0.0002,
              beta1=0.5,
              num_epochs=5,
              verbose=1,
              save_checkpoint=True,
              result_root=None,
              checkpoint_interval=1,
              generate_images=False, show_comparison_flag=False):
        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.0
        fake_label = 0.0

        # Optimizers for the generator and the discriminator
        gen_opt = torch.optim.RMSprop(self.netG.parameters(), lr=lr)
        critic_opt = torch.optim.RMSprop(self.critic.parameters(), lr=lr)

        # Critic training iterations are different from the generator training iterations. We train it more

        cur_step = 0
        display_step = 500
        CRITIC_ITERATIONS = 5
        WEIGHT_CLIP = 0.01
        # Training Loop

        # First a batch of real images are created and then a batch of fake images are created Loss is calculated for
        # both the discriminator and the generator log(1-D(G(z))) is the loss for the generator. We want to minimize
        # this loss so that the generator can fool the discriminator log(D(x)) + log(1-D(G(z))) is the loss for the
        # discriminator. We want to minimize this loss so that the discriminator can distinguish between real and
        # fake images. The discriminator and the generator are trained separately. The discriminator is trained first
        # and then the generator is trained. This is done because the discriminator is a more complex network and it
        # takes longer to train.

        # Lists to keep track of progress
        iters = 0

        print("Starting Training Loop...")
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = result_root + f'run_{now}'
        os.mkdir(run_path)
        f = open(run_path + '/log.txt', 'w')
        f.close()
        # For each epoch
        for epoch in range(num_epochs):
            # for each batch in the dataloader
            # Dataloader returns the batches
            for real_image, _ in tqdm(dataloader):
                cur_batch_size = real_image.shape[0]

                real_image = real_image.to(device)
                # Train Critic more than the generator for reliable gradients
                for _ in range(CRITIC_ITERATIONS):
                    fake_noise = self.get_noise(cur_batch_size, nz, device=device)
                    fake = self.netG(fake_noise)
                    critic_fake_pred = self.critic(fake).reshape(-1)
                    critic_real_pred = self.critic(real_image).reshape(-1)
                    # claculate the critic loss
                    critic_loss = -(torch.mean(critic_real_pred) - torch.mean(critic_fake_pred))
                    self.critic.zero_grad()
                    # To make a backward pass and retain the intermediary results
                    critic_loss.backward(retain_graph=True)
                    # Update optimizer
                    critic_opt.step()

                    # clip critic weights between -0.01, 0.01
                    for p in self.critic.parameters():
                        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                # Train Generator: min E[critic(fake)]
                gen_fake = self.critic(fake).reshape(-1)
                gen_loss = torch.mean(gen_fake)
                self.netG.zero_grad()
                gen_loss.backward()
                # Update optimizer
                gen_opt.step()
                self.D_losses.append(critic_loss.item())
                self.G_losses.append(gen_loss.item())
                ## Visualization code ##
                if cur_step % display_step == 0 and cur_step > 0:
                    print(f"Step {cur_step}: Generator loss: {gen_loss}, critic loss: {critic_loss}")
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    gen_loss = 0
                    critic_loss = 0
                cur_step += 1
            self.G_losses_mean = np.append(self.G_losses_mean, np.mean(self.G_losses[-len(dataloader):]))
            self.D_losses_mean = np.append(self.D_losses_mean, np.mean(self.D_losses[-len(dataloader):]))
            # Save model checkpoints by epoch interval
            if save_checkpoint:
                if not os.path.exists(run_path + '/current'):
                    os.mkdir(run_path + '/current')
                self.save_model(run_path + '/current/')
                if epoch % checkpoint_interval == 0:
                    os.mkdir(run_path + '/epoch_' + str(epoch))
                    self.save_model(run_path + '/epoch_' + str(epoch) + '/')  # The model is saved
                    if generate_images:
                        if show_comparison_flag:
                            self.side_by_side_comparison(dataloader, save_flag=True, show_flag=False,
                                                         save_path=run_path + '/generated_images/')

                        self.generate_images(100, run_path + '/epoch_' + str(epoch) + '/generated_images/',
                                             denormalize=True)  # The images are generated

            # Log every epoch
            with open(run_path + '/log.txt', 'a') as f:
                f.write(f'Epoch: {epoch} G_loss: {self.G_losses_mean[-1]} D_loss: {self.D_losses_mean[-1]} \n')
        # self.log_to_csv(run_path + '/log.txt')

    def generate_images(self, num_images, path, denormalize=False):
        """
        Usage: Generates images with the generator
        :param num_images: number of images to generate
        :param path: path to save the images
        """
        # Generate images
        if not os.path.exists(path):
            os.mkdir(path)
        with torch.no_grad():
            noise = torch.randn(num_images, nz, 1, 1, device=device)
            fake = self.netG(noise).detach().cpu()
        # Save images
        for i in range(num_images):
            if denormalize:
                vutils.save_image(fake[i] * 0.5 + 0.5, path + 'image_' + str(i) + '.png')
            else:
                vutils.save_image(fake[i], path + 'image_' + str(i) + '.png')

    # def log_to_csv(self, run_address):
    #     data = pd.read_csv(run_address, delimiter=' ')
    #     data = data.applymap(lambda x: re.sub('[^0-9]', '', str(x)))
    #     data = data.apply(pd.to_numeric, errors='coerce')
    #     data.columns = ['epoch', 'G_loss', 'D_loss']
    #     data.to_csv(run_address[:-4] + '.csv', index=False)

    def weight_initializer(self, m):
        """
        Usage: Initializes the weights of the model from a normal distribution with mean 0 and std 0.02
        :param m: model
        """
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, val=0)

    def save_model(self, path):
        # Saving the model
        torch.save(self.netG.state_dict(), path + 'generator.pth')
        torch.save(self.critic.state_dict(), path + 'discriminator.pth')

    def load_model(self, path, ngpu, device):
        # Loading the model
        self.netG = Generator(ngpu).to(device)
        self.netG.load_state_dict(torch.load(path + 'generator.pth'))
        self.critic = Discriminator(ngpu).to(device)
        self.critic.load_state_dict(torch.load(path + 'discriminator.pth'))

    def result_visualization(self):
        # Visualization of the results
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def side_by_side_comparison(self, data_loader, save_flag=False, save_path=None, show_flag=True):
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        if show_flag:
            plt.show()
        if save_flag:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path + 'comparison.png')
            plt.close()


if __name__ == '__main__':
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=data_root, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]))

    # Creating the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # batch_visualizer(device, dataloader, number_of_images=64)

    model = WGAN(device=device, ngpu=ngpu)
    # run_name = 'run_2023-07-11_10-25-45'
    # model.load_model(result_root + run_name + '/current/', ngpu=ngpu, device=device)
    model.train(dataloader=dataloader,
                device=device,
                num_epochs=200,
                verbose=1,
                nz=nz,
                lr=lr,
                beta1=beta1,
                save_checkpoint=True,
                checkpoint_interval=20,
                result_root=result_root,
                generate_images=True
                )

    # model.generate_images(num_images=100, path=result_root + run_name + '/generated_images/')
