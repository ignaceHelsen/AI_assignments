import copy
import io
import json
import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

import wandb
from dataset import Dataset
from losses import calc_vae_loss
from models import VanillaAutoEncoder, VariationalAutoEncoder


def load_config(config: dict) -> dict:
    path = os.path.join(config["save_path"], config["name"], "config.json")
    with open(path) as file:
        config = json.load(file)
    return config


def save_config(config: dict):
    to_save_config = copy.deepcopy(config)
    path = os.path.join(to_save_config["save_path"], to_save_config["name"], "config.json")
    if to_save_config.get("device") is not None:
        del to_save_config["device"]
    with open(path, 'w') as file:
        json.dump(to_save_config, file)


def train_autoencoder(model: VanillaAutoEncoder, options: dict, dataset: Dataset,
                      optimizer: torch.optim.Optimizer):
    """"
        This function should train your AE.
        TODO: Implement the code below.
    """
    # TODO: define the loss function.
    distance = nn.MSELoss()

    for epoch in range(options["num_epochs"]):
        for data in dataset.train_loader:
            img, _ = data
            img = torch.Tensor(img).to(options["device"])

            optimizer.zero_grad()

            # TODO: forward the image through the model.
            output = model.forward(img)

            # TODO: calculate the loss
            loss = distance(output, img)

            # TODO: Backpropagate the loss through the model;
            loss.backward()

            # TODO: use the optimizer in a correct way to update the weights.
            optimizer.step()

        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, options["num_epochs"], loss.item()))
        recon = test_autoencoder(model, dataset, options)
        gen = generate_using_encoder(model, options)
        wandb.log(
            {
                "loss": loss.item(),
                "Image reconstruction": wandb.Image(recon),
                "Image generation": wandb.Image(dataset.denormalize(gen))
            }
        )


def train_vae(model: VariationalAutoEncoder, options: dict, dataset: Dataset,
              optimizer: torch.optim.Optimizer):
    """"
    This function should train your VAE.
    TODO: Implement the code below.
    """
    # TODO: define the loss function.
    distance = nn.MSELoss()

    for epoch in range(options["num_epochs"]):
        for data in dataset.train_loader:
            img, _ = data
            img = torch.Tensor(img).to(options["device"])

            optimizer.zero_grad()

            # TODO: forward the image through the model.
            output = model.forward(img)

            # TODO: calculate the loss
            loss = calc_vae_loss(distance, output, img, options)

            # TODO: Backpropagate the loss through the model;
            loss.backward()

            # TODO: use the optimizer in a correct way to update the weights.
            optimizer.step()

        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, options["num_epochs"], loss.item()))
        recon = test_vae(model, dataset, options)
        gen = generate_using_encoder(model, options)
        wandb.log(
            {
                "loss": loss.item(),
                "Image reconstruction": wandb.Image(recon),
                "Image generation": wandb.Image(dataset.denormalize(gen))
            }
        )


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """"
    You need to perform reparametrization for your VAE
    The goal of reparametrization is to have a probability involved to encode a value
    onto a certain place in the latent space.
    TODO: Implement this below.
    """
    return mu + torch.exp(log_var / 2) * torch.distributions.Normal(0, 1).sample(mu.shape)

def save(model: nn.Module, config: dict):
    if not os.path.exists(os.path.join(config["save_path"], config["name"])):
        os.makedirs(os.path.join(config["save_path"], config["name"]))
    torch.save(model.state_dict(), os.path.join(config["save_path"], config["name"], config["name"]) + ".pt")
    save_config(config)


def load(model: nn.Module, config: dict):
    try:
        model.load_state_dict(torch.load(os.path.join(config["save_path"], config["name"], config["name"]) + ".pt"))
        model.eval()
    except IOError:
        print("Could not load module!!")


def test_autoencoder(model: [VanillaAutoEncoder, VariationalAutoEncoder], dataset: Dataset,
                     options: dict) -> torch.Tensor:
    """"
    This function tests the autoencoder by plotting the original image and its reconstruction.
    """
    examples = enumerate(dataset.test_loader)
    _, (example_data, example_targets) = next(examples)

    reconstruction = model.forward(example_data[:8].to(options["device"])).detach()
    comparison_images = torch.cat((dataset.denormalize(example_data[:8]), reconstruction), dim=0)

    return comparison_images


def test_vae(model: VariationalAutoEncoder, dataset: Dataset, options: dict):
    """"
    This function tests the VAE by plotting the original image and its reconstruction.
    """
    examples = enumerate(dataset.test_loader)
    _, (example_data, example_targets) = next(examples)

    reconstruction, _, _, _ = model.forward(example_data[:8].to(options["device"]))
    comparison_images = torch.cat((dataset.denormalize(example_data[:8]), reconstruction.detach()), dim=0)

    return comparison_images


def generate_using_encoder(model: [VanillaAutoEncoder, VariationalAutoEncoder], options: dict) -> torch.Tensor:
    """"
    This function generates images using your module.
    """
    gen_image = model.generate(torch.randn(64, options["latent_dim"]).to(options["device"])).detach()

    return gen_image


def plot_latent(autoencoder: nn.Module, dataset: Dataset, options: dict, num_batches: int = 100) -> Image:
    """
    Plot the latent space to see how it differs between models.
    """
    plt.figure()
    if options["latent_dim"] <= 1:
        print("Cannot visualise the latent space, as there are less then 2 dimensions...")
    else:
        for i, (x, y) in enumerate(dataset.test_loader):
            z = autoencoder.encode(x.to(options["device"]))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = copy.deepcopy(Image.open(img_buf))
        img_buf.close()

        return im


def plot_latent_pca(autoencoder: nn.Module, dataset: Dataset, options: dict, num_batches: int = 100) -> Image:
    """
    Plot the latent space to see how it differs between models.
    """
    if options["latent_dim"] < 2:
        print("Cannot perform 2D Principal Component Analysis, as there are less then 2 dimensions...")
    else:
        plt.figure()
        pca = PCA(n_components=2)
        for i, (x, y) in enumerate(dataset.test_loader):
            z = autoencoder.encode(x.to(options["device"]))
            z = z.to('cpu').detach().numpy()
            pca.fit(z)
            reduced_z = pca.transform(z)
            plt.scatter(reduced_z[:, 0], reduced_z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = copy.deepcopy(Image.open(img_buf))
        img_buf.close()

        return im


def plot_latent_pca_3d(autoencoder: nn.Module, dataset: Dataset, options: dict, num_batches: int = 100) -> Image:
    """
    Plot the latent space to see how it differs between models.
    """
    if options["latent_dim"] < 3:
        print("Cannot perform 3D Principal Component Analysis, as there are less then 3 dimensions...")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        pca = PCA(n_components=3)
        for i, (x, y) in enumerate(dataset.test_loader):
            z = autoencoder.encode(x.to(options["device"]))
            z = z.to('cpu').detach().numpy()
            pca.fit(z)
            reduced_z = pca.transform(z)
            ax.scatter(reduced_z[:, 0], reduced_z[:, 1], reduced_z[:, 2], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = copy.deepcopy(Image.open(img_buf))
        img_buf.close()

        return im
