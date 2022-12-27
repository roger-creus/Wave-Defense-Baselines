import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import wandb

class VAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.permute(0,3,1,2)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.000025
        recons_loss = F.mse_loss(recons, input.permute(0,3,1,2))

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, logger = None):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)

        if logger is not None:
            fig, axs = plt.subplots(1, num_samples, sharey=True, figsize=(18, 2))

            for s in range(num_samples):
                axs[s].imshow(samples[s,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
                axs[s].axis('off')
            
            logger.experiment.log({'VAE samples': fig})
            plt.close(fig)
            
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        recons = self.forward(x)[0]

        num_examples = x.size(0)

        fig, axs = plt.subplots(2, num_examples, figsize=(18, 4))

        for s in range(num_examples):
            axs[0,s].imshow(x[s,:,:,:].cpu().detach().numpy(), interpolation='nearest')
            axs[1,s].imshow(recons[s,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
            
            axs[0,s].axis('off')
            axs[1,s].axis('off')
        
        wandb.log({'VAE reconstructions': fig})
        plt.close(fig)

        return recons

class GaussianBuffer():
    def __init__(self, num_envs, obs_dim):
        super().__init__()

        self.num_envs = num_envs
        self.obs_dim = obs_dim

        self.buffer = np.zeros((num_envs, 1,obs_dim))
        
        self.add(np.ones((num_envs, 1, obs_dim)))
        self.add(-np.ones((num_envs, 1, obs_dim)))
        
    def add(self, obs):
        self.buffer = np.concatenate((self.buffer,obs), axis=1)

    def get_params(self):
        means = np.mean(self.buffer, axis=1)
        stds = np.std(self.buffer, axis=1)
        return means, stds

    def logprob(self, obs):
        obs = obs.copy()
        means, stds = self.get_params()
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-5
        stds = np.clip(stds, thresh, None)

        logprob = -0.5*np.sum(np.log(2*np.pi*stds), axis = 1) - np.sum(np.square(obs-means)/(2*np.square(stds)), axis = 1)
        return logprob

    def reset(self, idx = None):
        if idx is None:
            self.buffer = np.zeros((self.num_envs, 1, self.obs_dim))
        else:
            self.buffer[idx, :, :] = np.zeros((1, 1, self.obs_dim))