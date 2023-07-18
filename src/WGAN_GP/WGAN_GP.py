import datetime
import math
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
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Root directory for dataset
data_root = '../datasets/Total_Crack/'

# Result directory
result_root = './src/WGAN_GP/results/'

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
        # # input is Z, going into a convolution. nz is the vector size of the latent space for each image
        # # ngf is the number of feature maps in the generator.
        # # Feature maps are like channels in the image. So this number goes down as the network gets deeper
        # # since we are reducing the number of channels to get to the number of channels in the image
        # # nc is the number of channels in the final image
        modules = []
        modules.append(nn.ConvTranspose2d(nz, ngf * ngf // 4, 4, 1, 0, bias=False))
        modules.append(nn.BatchNorm2d(ngf * ngf // 4))
        modules.append(nn.ReLU(True))
        for i in reversed(range(int(math.log2(ngf // 4)))):
            modules.append(nn.ConvTranspose2d(ngf * 2 ** (i + 1), ngf * 2 ** i, 4, 2, 1, bias=False))
            modules.append(nn.BatchNorm2d(ngf * 2 ** i))
            modules.append(nn.ReLU(True))
        modules.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        modules.append(nn.Tanh())
        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # the discriminator model architecture is defined here
        modules = []
        modules.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(int(math.log2(ndf // 4))):
            modules.append(nn.Conv2d(ndf * 2 ** i, ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            # Can be commented out to remove instance norm
            modules.append(nn.InstanceNorm2d(ndf * 2 ** (i + 1), affine=True))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv2d((ndf * ndf // 4), 1, 4, 1, 0, bias=False))
        modules.append(nn.Sigmoid())
        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)


class WGAN_GP:
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
              critic_iterations=5,
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
                # First the discriminator is trained for critic_iterations times
                for _ in range(critic_iterations):

                    z = torch.normal(0, 0.1, size=(data.size(0), nz, 1, 1))
                    if torch.cuda.is_available():
                        img, z = data.cuda(), z.cuda()

                    # train D
                    self.netD.zero_grad()
                    loss_real = -self.netD(img).mean()
                    fake_img = self.netG(z).detach()
                    loss_fake = self.netD(fake_img).mean()
                    gp = self.gradient_penalty(data.detach(), fake_img, LAMDA=10, cuda=torch.cuda.is_available())
                    loss_D = loss_real + loss_fake + gp
                    loss_D.backward()
                    optimizerD.step()

                # train G
                z = torch.normal(0, 1, size=(img.size(0), nz, 1, 1))
                if torch.cuda.is_available():
                    z = z.cuda()
                self.netG.zero_grad()
                loss_G = -self.netD(self.netG(z)).mean()
                loss_G.backward()
                optimizerG.step()

                self.D_losses.append(loss_D.item())
                self.G_losses.append(loss_G.item())

                if verbose == 1:
                    # Output training stats
                    print(
                        f'[Epoch {epoch + 1}/{num_epochs}] [G loss: {loss_G.item()}] [D loss: {loss_D.item()} | '
                        f'loss_real: {loss_real.item()} loss_fake: {loss_fake.item()}]')
                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                        with torch.no_grad():  # No gradients are calculated saving memory and time
                            fake = self.netG(fixed_noise).detach().cpu()
                        self.img_list.append(
                            vutils.make_grid(fake, padding=2, normalize=True))  # The fake images are saved
                if verbose == 2:
                    print(f'Epoch {epoch + 1}/{num_epochs} Batch {i + 1}/{len(dataloader)}')
                # Saving losses for plotting later
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
                    if generate_images:
                        self.generate_images(100, run_path + '/epoch_' + str(epoch) + '/generated_images/',
                                             denormalize=True)  # The images are generated

            # Log every epoch
            with open(run_path + '/log.txt', 'a') as f:
                f.write(f'Epoch: {epoch} G_loss: {self.G_losses_mean[-1]} D_loss: {self.D_losses_mean[-1]} \n')

        # self.log_to_csv(run_path + '/log.txt')

    def gradient_penalty(self, real_img, fake_img, LAMDA=10, cuda=True):
        b_size = real_img.size(0)
        alpha = torch.rand(b_size, 1)
        alpha = alpha.expand(b_size, real_img.nelement() // b_size).reshape(real_img.shape)
        if cuda:
            alpha = alpha.cuda()
        x = (alpha * real_img + (1 - alpha) * fake_img).requires_grad_(True)
        if cuda:
            x = x.cuda()
        out = self.netD(x)

        grad_outputs = torch.ones(out.shape)
        if cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = \
            torch.autograd.grad(outputs=out, inputs=x, grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[
                0]
        gradients = gradients.reshape(b_size, -1)

        return LAMDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

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
        torch.save(self.netG.state_dict(), path + 'generator.pth')
        torch.save(self.netD.state_dict(), path + 'discriminator.pth')

    def load_model(self, path, ngpu, device):
        # Loading the model
        self.netG = Generator(ngpu).to(device)
        self.netG.load_state_dict(path + 'generator.pth')
        self.netD = Discriminator(ngpu).to(device)
        self.netD.load_state_dict(path + 'discriminator.pth')
        # self.netG= torch.load(path + 'generator.rar', map_location=device)
        # self.netD = torch.load(path + 'discriminator.rar', map_location=device)

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

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)


if __name__ == '__main__':
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=data_root, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))]  # changed to a single channel
    ))

    # Creating the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # batch_visualizer(device, dataloader, number_of_images=64)

    model = WGAN_GP(device=device, ngpu=ngpu)
    # run_name = 'run_2023-07-06_15-13-00'
    # model.load_model(result_root + run_name + '/current/', ngpu, device)
    model.train(dataloader=dataloader,
                device=device,
                num_epochs=1000,
                verbose=2,
                nz=nz,
                lr=lr,
                beta1=beta1,
                save_checkpoint=True,
                checkpoint_interval=100,
                result_root=result_root,
                generate_images=True
                )

    # model.generate_images(num_images=100, path=result_root + run_name + '/generated_images/', denormalize=True)
#
