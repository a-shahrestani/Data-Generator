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
import torch.nn.functional as F
from tqdm import tqdm

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Root directory for dataset
data_root = '../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Images/class_seperated/'

# Result directory
result_root = './src/Conditional GAN/results/C_WGAN/'

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

# Number of classes
classes = 4


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
    def __init__(self, ngpu, classes):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # the generator model architecture is defined here
        # # input is Z, going into a convolution. nz is the vector size of the latent space for each image
        # # ngf is the number of feature maps in the generator.
        # # Feature maps are like channels in the image. So this number goes down as the network gets deeper
        # # since we are reducing the number of channels to get to the number of channels in the image
        # # nc is the number of channels in the final image

        self.ngpu = ngpu
        self.deconv1_1 = nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False)
        self.deconv1_1_bn = nn.BatchNorm2d(ngf * 16)
        self.deconv1_2 = nn.ConvTranspose2d(1, ngf * 16, 4, 1, 0, bias=False)
        self.deconv1_2_bn = nn.BatchNorm2d(ngf * 16)
        self.deconv2 = nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(ngf * 16)
        self.deconv3 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(ngf * 8)
        self.deconv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(ngf * 4)
        self.deconv5 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.deconv5_bn = nn.BatchNorm2d(ngf * 2)
        self.deconv6 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.deconv6_bn = nn.BatchNorm2d(ngf)
        self.deconv7 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)), 0.2)
        x = F.leaky_relu(self.deconv6_bn(self.deconv6(x)), 0.2)
        x = torch.tanh(self.deconv7(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, ngpu, classes):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # the discriminator model architecture is defined here
        self.conv1_1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        # number of classes is set to 1
        self.conv1_2 = nn.Conv2d(1, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ndf * 4)
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ndf * 8)
        self.conv4 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ndf * 16)
        self.conv5 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(ndf * 32)
        self.conv6 = nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(ndf * 64)
        self.conv7 = nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False)

    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2)
        x = self.conv7(x)
        return x


class C_WGAN:
    def __init__(self, device, ngpu, nz=100):
        # Create the generator and initialize the weights
        self.netG = Generator(ngpu, classes).to(device)
        self.netG.apply(self.weight_initializer)
        # Create the Discriminator and initialize the weights
        self.netD = Discriminator(ngpu, classes).to(device)
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

    def train(self, image_size, label_size, dataloader,
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
              g_iters=1,
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
            batch = tqdm(enumerate(dataloader), desc="Epoch " + str(epoch), total=len(dataloader.dataset) // batch_size)
            for i, (data, labels) in enumerate(dataloader):
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #########################################
                # First the discriminator is trained for critic_iterations times

                self.netD.zero_grad()  # Zeroing the gradients of the discriminator
                # Format batch
                images = data.to(device)  # The real images are sent to the GPU
                b_size = images.size(0)  # The batch size is calculated

                y_real = torch.full((b_size,), 1, dtype=torch.float, device=device)
                y_fake = torch.full((b_size,), 0, dtype=torch.float, device=device)

                broadcasted_labels = torch.zeros(b_size, 1, image_size, image_size, device=device)
                g_labels = labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=device, dtype=torch.float)
                d_labels = broadcasted_labels + g_labels
                img = data.to(device)

                # discriminator training
                disc_losses = []
                disc_accuracies = []

                for _ in range(critic_iterations):
                    noise = torch.randn(b_size, nz, 1, 1, device=device)

                    # train D
                    self.netD.zero_grad()
                    output = self.netD(images, d_labels)
                    loss_real = -output.mean()

                    fake_img = self.netG(noise, g_labels).detach()
                    loss_fake = self.netD(fake_img, d_labels).mean()
                    gp = self.gradient_penalty(img.detach(), fake_img, d_labels, LAMDA=10,
                                               cuda=torch.cuda.is_available())
                    loss_D = loss_fake + loss_real + gp
                    loss_D.backward()
                    optimizerD.step()

                # train G
                for i in range(g_iters):
                    self.netG.zero_grad()  # fake labels are real for generator cost
                    # Since only discriminator is updated, the output of the generator is calculated again
                    fake = self.netG(noise, g_labels)
                    output = self.netD(fake, d_labels).view(-1)
                    # Calculate G's loss based on this output
                    loss_G = -output.mean()
                    loss_G.backward()  # The gradients are calculated
                    # Update G
                    optimizerG.step()  # The optimizer is updated
                    # gen_losses.append(errG.item())

                # TODO: Write the logging section for accuracy and loss
                self.D_losses.append(loss_D.item())
                self.G_losses.append(loss_G.item())
                if verbose == 1:
                    # Output training stats
                    print(
                        f'[Epoch {epoch + 1}/{num_epochs}] [G loss: {loss_G.item()}] [D loss: {loss_D.item()} | '
                        f'loss_real: {loss_real.item()} loss_fake: {loss_fake.item()}] | Batch {i + 1}/{len(dataloader)}')
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
                    self.save_model(run_path + '/epoch_' + str(epoch) + '/')
                    self.generate_images(100, run_path + '/current/generated_images/',
                                         denormalize=True)
                    # The model is saved
                    if generate_images:
                        self.generate_images(100, run_path + '/epoch_' + str(epoch) + '/generated_images/',
                                             denormalize=True)  # The images are generated

            # Log every epoch
            with open(run_path + '/log.txt', 'a') as f:
                f.write(f'Epoch: {epoch} G_loss: {self.G_losses_mean[-1]} D_loss: {self.D_losses_mean[-1]} \n')

        # self.log_to_csv(run_path + '/log.txt')

    def gradient_penalty(self, real_img, fake_img, d_labels, LAMDA=10, cuda=True):
        b_size = real_img.size(0)
        alpha = torch.rand(b_size, 1)
        alpha = alpha.expand(b_size, real_img.nelement() // b_size).reshape(real_img.shape)
        if cuda:
            alpha = alpha.cuda()
        x = (alpha * real_img + (1 - alpha) * fake_img).requires_grad_(True)
        if cuda:
            x = x.cuda()
        out = self.netD(x, d_labels)

        grad_outputs = torch.ones(out.shape)
        if cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = \
            torch.autograd.grad(outputs=out,
                                inputs=x,
                                grad_outputs=grad_outputs,
                                create_graph=True,
                                only_inputs=True,
                                retain_graph=True)[0]
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
            label = torch.randint(0, 4, (num_images, 1, 1, 1), device=device, dtype=torch.float)
            noise = torch.randn(num_images, nz, 1, 1, device=device)
            fake = self.netG(noise, label).detach().cpu()
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
        self.netG.load_state_dict(torch.load(path + 'generator.pth'))
        self.netD = Discriminator(ngpu).to(device)
        self.netD.load_state_dict(torch.load(path + 'discriminator.pth'))
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

    model = C_WGAN(device=device, ngpu=ngpu)
    # run_name = 'run_2023-07-18_14-44-15'
    # model.load_model(result_root + run_name + '/current/', ngpu, device)

    model.train(dataloader=dataloader,
                image_size=image_size,
                label_size=classes,
                device=device,
                num_epochs=200,
                verbose=2,
                nz=nz,
                lr=lr,
                beta1=beta1,
                save_checkpoint=True,
                checkpoint_interval=20,
                result_root=result_root,
                generate_images=True
                )

    # model.generate_images(num_images=100, path=result_root + run_name + '/generated_images/', denormalize=True)
#
