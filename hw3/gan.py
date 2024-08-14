import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        self.cnn = nn.Sequential(
            nn.Conv2d(in_size[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Calculate the number of features after the CNN layers
        num_features = self._calc_num_cnn_features(in_size)
        
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        

        # ========================

    def _calc_num_cnn_features(self, in_shape):
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            out_shape = self.cnn(x).shape
        return np.prod(out_shape)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        
        # ========================
        return output


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim


        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # Starting with a linear layer to transform the latent vector into
        # a feature map of size (featuremap_size x featuremap_size).
        self.fc = nn.Linear(z_dim, 512 * featuremap_size * featuremap_size)
        
        # Define the transpose convolutional layers to upscale the feature map.
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Tanh()  # Output range will be [-1, 1]
        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n, self.z_dim, device=device)
    
        if with_grad:
            # Compute the samples using the generator
            samples = self(z)
        else:
            # Compute the samples without tracking gradients
            with torch.no_grad():
                samples = self(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape into initial 4x4 feature map
        x = self.net(x)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    criterion = nn.BCEWithLogitsLoss()
    
    # Create labels with noise
    criterion = nn.BCEWithLogitsLoss()
    labels_real = torch.full_like(y_data, data_label) + torch.rand_like(y_data) * label_noise - label_noise/2
    labels_generated = torch.full_like(y_generated, 1 - data_label) + torch.rand_like(y_generated) * label_noise - label_noise/2
    loss_data = criterion(y_data, labels_real)
    loss_generated = criterion(y_generated, labels_generated)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    criterion = nn.BCEWithLogitsLoss()
    labels_generated = torch.full_like(y_generated, data_label)
    loss = criterion(y_generated, labels_generated)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_model.train()
    gen_model.train()

    # Forward pass real data through discriminator
    real_scores = dsc_model(x_data)
    
    # Generate fake data using the generator
    fake_data = gen_model.sample(x_data.size(0), with_grad=False)
    fake_scores = dsc_model(fake_data)

    # 2. Calculate discriminator loss
    dsc_loss = dsc_loss_fn(real_scores, fake_scores)

    # 3. Update discriminator parameters
    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()

    # Generator update
    # 1. Generate fake data again but this time allow gradient tracking
    fake_data = gen_model.sample(x_data.size(0), with_grad=True)
    fake_scores = dsc_model(fake_data)

    # 2. Calculate generator loss
    gen_loss = gen_loss_fn(fake_scores)

    # 3. Update generator parameters
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) > 1 and gen_losses[-1] < min(gen_losses[:-1]):
        torch.save(gen_model.state_dict(), checkpoint_file)
        print(f"*** Saved checkpoint {checkpoint_file} ")
        saved = True
    # ========================
    #torch.save(gen_model, checkpoint_file)
    #print(f"*** Saved checkpoint {checkpoint_file} ")
    #saved = True
    return saved
