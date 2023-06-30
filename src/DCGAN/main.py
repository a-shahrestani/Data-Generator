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
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Root directory for dataset
data_root = '../../../datasets/Celeba/'

# Result directory
result_root = './results/'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3 (RGB) For grayscale images this is 1
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.001

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
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 64x64
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
            # input is (nc) x 64 x 64 because our images are 64 * 64 * 3
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN:
    def __init__(self, device, ngpu, nz=100):
        # Create the generator and initialize the weights
        self.netG = Generator(ngpu).to(device)
        self.netG.apply(self.weight_initializer)
        # Create the Discriminator and initialize the weights
        self.netD = Discriminator(ngpu).to(device)
        self.netD.apply(self.weight_initializer)
        # Initialize BCELoss function for the GAN
        self.criterion = nn.BCELoss()
        # Latent vector to visualize the progression of the generator
        self.nz = nz
        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        # Mean loss calculation for each epoch
        self.G_losses_mean = None
        self.D_losses_mean = None

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
              generate_images=False):
        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.0
        fake_label = 0.0

        # Adam Optimizer for both the generator and the discriminator as per the DCGAN paper
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

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
            for i, data in enumerate(dataloader, 0):
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #########################################

                # First the discriminator is trained on real images
                self.netD.zero_grad()  # Zeroing the gradients of the discriminator
                # Format batch
                real_cpu = data[0].to(device)  # The real images are sent to the GPU
                b_size = real_cpu.size(0)  # The batch size is calculated
                label = torch.full((b_size,), real_label, device=device)  # The labels for the real images are created

                # Forward pass
                output = self.netD(real_cpu).view(
                    -1)  # The real images are sent to the discriminator and the output is calculated
                errD_real = self.criterion(output, label)  # The loss for the real images is calculated

                # Backward pass
                errD_real.backward()  # The gradients are calculated
                D_x = output.mean().item()  # The mean of the output is calculated

                # train on the fake images
                # noise in generated
                noise = torch.randn(b_size, nz, 1, 1, device=device)  # The noise is created
                # generate fake images
                fake = self.netG(noise)
                label.fill_(fake_label)  # The labels for the fake images are created
                # Classify the fake images with the discriminator

                # Forward pass
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)  # The loss for the fake images is calculated

                # Backward pass
                errD_fake.backward()  # The gradients are calculated
                D_G_z1 = output.mean().item()  # The mean of the output is calculated

                # Computing the total loss for the discriminator
                errD = errD_real + errD_fake
                optimizerD.step()  # The optimizer is updated

                # (2) Update G network: maximize log(D(G(z))) #########################################################

                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since only discriminator is updated, the output of the generator is calculated again
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)  # The loss for the generator is calculated
                # Backward pass
                errG.backward()  # The gradients are calculated
                D_G_z2 = output.mean().item()  # The mean of the output is calculated
                # Update G
                optimizerG.step()  # The optimizer is updated

                if verbose == 1:
                    # Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                              (epoch + 1, num_epochs, i, len(dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                        with torch.no_grad():  # No gradients are calculated saving memory and time
                            fake = self.netG(fixed_noise).detach().cpu()
                        self.img_list.append(
                            vutils.make_grid(fake, padding=2, normalize=True))  # The fake images are saved
                if verbose == 2:
                    print(f'Epoch {epoch + 1}/{num_epochs} Batch {i + 1}/{len(dataloader)}')
                # Saving losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                iters += 1
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

            # Log every epoch
            with open(run_path + '/log.txt', 'a') as f:
                f.write(f'Epoch: {epoch} G_loss: {self.G_losses_mean[-1]} D_loss: {self.D_losses_mean[-1]} \n')
        # self.log_to_csv(run_path + '/log.txt')
        if generate_images:
            self.generate_images(100, run_path + '/generated_images/')

    def generate_images(self, num_images, path):
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
            vutils.save_image(fake[i], path + 'image_' + str(i) + '.png')

    def log_to_csv(self, run_address):
        data = pd.read_csv(run_address, delimiter=' ')
        data = data.applymap(lambda x: re.sub('[^0-9]', '', str(x)))
        data = data.apply(pd.to_numeric, errors='coerce')
        data.columns = ['epoch', 'G_loss', 'D_loss']
        data.to_csv(run_address[:-4] + '.csv', index=False)

    def weight_initializer(self, m):
        """
        Usage: Initializes the weights of the model from a normal distribution with mean 0 and std 0.02
        :param m: model
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def save_model(self, path):
        # Saving the model
        torch.save(self.netG, path + 'generator.rar')
        torch.save(self.netD, path + 'discriminator.rar')

    def load_model(self, path):
        # Loading the model
        self.netG = torch.load(path + 'generator.rar')
        self.netD = torch.load(path + 'discriminator.rar')

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

    def side_by_side_comparison(self, data_loader):
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
        plt.show()


if __name__ == '__main__':
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=data_root, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # Creating the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # batch_visualizer(device, dataloader, number_of_images=64)

    model = GAN(device=device, ngpu=ngpu)
    run_name = 'run_2023-06-27_11-29-00'
    model.load_model(result_root + run_name + '/current/')
    model.train(dataloader=dataloader,
                device=device,
                num_epochs=200,
                verbose=1,
                nz=nz,
                lr=lr,
                beta1=beta1,
                save_checkpoint=True,
                checkpoint_interval=10,
                result_root=result_root
                )

    # model.generate_images(num_images=100, path=result_root + run_name + '/generated_images/')
