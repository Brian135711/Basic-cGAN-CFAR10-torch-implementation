import cv2
import mediapipe as mp
import numpy as np
import time
from shapely import Point, Polygon
import pandas as pd
from PIL import Image, ImageChops
import matplotlib.pyplot as plt  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


#set gpu for execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size, n_classes, txt_label):
        super().__init__()
        self.img_size = img_size
        #print(f'IMG SIZE INTO DISCRIMINATOR:', img_size)
        #print(f'N_CLASSES INTO DISCRIMINATOR:', n_classes)
        self.embed = nn.Embedding(n_classes, img_size[0]*img_size[1]) #n_classes x 1/2(latent_dim 100) latent dim may change
       
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 128, (3, 3), stride=(2, 2), padding=0), # 4 channels (3 for image and 1 for txt_label) to 128 channels
            nn.LeakyReLU(0.2), # Leaky ReLU activation function with a slope of 0.2 to introduce non-linearity to the model. output dimensions: (16, 16, 128)
            nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=1), # 128 channels to 128 channels with stride of 2 and padding of 1 to reduce the dimensions, output dimensions: (8, 8, 128)
            nn.LeakyReLU(0.2) # Leaky ReLU activation function with a slope of 0.2 to introduce non-linearity to the model. output dimensions: (8, 8, 128)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), #input dimensions: [2,128,8,8] output dimensions: [2, 128*8*8] -> [2, 8192]  #Flatten() skips the first (batch_size) dimension and combines the rest into a single dimension. flatten() doesn't skip first dimension.
            nn.Dropout(0.4), #input dimensions: [2, 8192] output dimensions: [2, 8192]
            nn.Linear(4 * 4 * 4 * 128, 1), #input dimensions: [2, 8192] output dimensions: [2, 1] #Linear layer to get the final output of the discriminator model with input dimensions of 8192 and output dimensions of 1
            nn.Sigmoid()
        )

    def forward(self, image, txt_label, n_batch):
        #print(f'IMAGE SIZE INTO DISCRIMINATOR:', image.shape)
        #print (f'txt_label SIZE INTO DISCRIMINATOR:', txt_label.shape)
        #print(f'BATCH SIZE INTO DISCRIMINATOR:', n_batch)
        embedding = self.embed(txt_label).view(txt_label.shape[0], 1, self.img_size[0], self.img_size[1])
        image = torch.cat([image, embedding], dim=1) # Concatenate the image and the text label embedding
        #print(f'IMAGE SIZE AFTER CONCAT INTO CONV_FEATURES:', image.shape)
        conv_features = self.conv_layers(image) #run the image through the convolutional layers to get the features
        #print('conv_features:', conv_features.shape)
        output = self.fc_layers(conv_features) #run through the fully connected layers to get the output
        #print('OUTPUT SIZE FROM DISCRIMINATOR:', output.shape)
        return output

# Define the generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size):
        super().__init__()
        self.img_size = img_size

        self.embed = nn.Embedding(n_classes, latent_dim) #n_classes x 1/2(latent_dim 100) latent dim may change
        
        self.latent_to_img = nn.Sequential(
            nn.Linear(latent_dim + latent_dim  , 128 * int(self.img_size[0] * self.img_size[1] / 4)), #input size: latent_dim + n_classes, output size: 128 * (img_size[0] * img_size[1] / 4), output dimensions: [2, 128 * (32 * 32 / 4)] -> [2, 128 * 256] -> [2, 32768]
            nn.LeakyReLU(0.2), #input dimensions: [2, 32768] output dimensions: [2, 32768]
            #unflatten to shape [2,128,8,8] #input dimensions: [2, 32768] output dimensions: [2, 512, 8, 8] #Unflatten() is the opposite of flatten(). It takes a tensor and reshapes it into a specified shape.
            nn.Unflatten(1, (512, int(self.img_size[0] // 4), int(self.img_size[1] // 4)))
            

        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 128, (2, 2), stride=(3, 3), padding=3), #input dimensions: [2, 128, 8, 8] output dimensions: [2, 128, 16, 16] #Deconvolutional layer to increase the dimensions of the input tensor
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 3, kernel_size= (self.img_size[0] // 2, self.img_size[1] // 2), stride =(1,1), padding=0),
            nn.Tanh()
        )

    def forward(self, noise, txt_label):
        #latent vector: Batch_size x noise_dim 
        embedding = self.embed(txt_label) 
        #print(f'EMBEDDING SIZE BEFORE CONCAT:', embedding.shape)
        #print(f'NOISE SIZE BEFORE CONCAT:', noise.shape)
        noise = torch.cat([noise, embedding], dim=1) # Concatenate the noise and the text label embedding
        #print(f'NOISE SIZE AFTER CONCAT:', noise.shape)
        img = self.latent_to_img(noise) #run the noise through the linear layers to get the features
        #print(f'IMAGE SIZE GENERATOR AFTER LATENT TO IMG:', img.shape)
        img = self.deconv_layers(img) #run the noise through the deconvolutional layers to get the image
        #print(f'IMG SIZE RETURNED BY GENERATOR:', img.shape)
        return img
    

class GANLoss(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, g_output, g_txt_label, real_txt_label):
        d_output = self.discriminator(g_output, g_txt_label)
        return F.binary_cross_entropy(d_output, real_txt_label)

# Define the combined GAN model for training the generator
def define_gan(generator, discriminator):
    discriminator.eval()  # Make discriminator non-trainable
    gan_loss = GANLoss(discriminator)  # Create GANLoss module
    gan_model = nn.Sequential(generator, gan_loss)  # Now it's valid
    return gan_model


#Use the CFAR10 dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision import transforms

# Define the dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = CIFAR10(root='data/', download=True, transform=transform)
txt_label = dataset.classes
print(txt_label)

# Define the models
latent_dim = 100
n_classes = 10
img_size = (32, 32, 3)
generator = Generator(latent_dim, n_classes, img_size).to(device)
discriminator = Discriminator(img_size, n_classes, txt_label).to(device)

# Define the GAN model
gan_model = define_gan(generator, discriminator)



# Training function
def train(gan_model, generator, discriminator, dataset, latent_dim, n_classes, img_size, n_epochs=100, n_batch=100, lr=0.0002):
    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
    # Define the optimizer
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    # Define the loss function
    criterion = nn.BCELoss()
    # Loop through the epochs
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}')
        # Loop through the dataloader
        for i, (real_img, real_txt_label) in enumerate(dataloader):
            real_img = real_img.to(device)
            real_txt_label = real_txt_label.to(device)
            # Generate the fake image
            noise = torch.randn(real_img.shape[0], latent_dim).to(device)
            fake_txt_label = torch.randint(0, n_classes, (real_img.shape[0],)).to(device)
            fake_img = generator(noise, fake_txt_label)
            # Train the discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(real_img, real_txt_label, n_batch)
            fake_output = discriminator(fake_img.detach(), fake_txt_label, n_batch)
            d_real_loss = criterion(real_output, torch.ones_like(real_output))
            d_fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            # Train the generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_img, fake_txt_label, n_batch)
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            g_optimizer.step()
            # Print the losses
            if i % 100 == 0:
                print(f'Iteration {i}, d_loss={d_loss.item()}, g_loss={g_loss.item()}')


#Train the cGAN model
train(gan_model, generator, discriminator, dataset, latent_dim, n_classes, img_size)

#save the model to disk
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# Generate a batch of fake images
noise = torch.randn(10, latent_dim).to(device)  
fake_txt_label = torch.randint(0, n_classes, (10,)).to(device)
fake_img = generator(noise, fake_txt_label)
fake_img = fake_img.cpu().detach()
# Display the fake images
plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(fake_img[i].permute(1, 2, 0))
    plt.axis('off')
plt.show()
