import random

from options import CustomParser
from utils import *

parsed_data = CustomParser().parse({})
config = load_config(parsed_data)
config["device"] = parsed_data["device"]
config["latent_dim"] = 10

# set all random seeds for reproducibility
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed(config["random_seed"])
random.seed(config["random_seed"])

# set device
if config["device"] == "cuda" and torch.cuda.is_available():
    config["device"] = torch.device("cuda:0")
else:
    config["device"] = torch.device("cpu")

wandb.init(project="AE_project", entity="the-elite-schakels", name=config["name"] + "_test")

# Let's first prepare the MNIST dataset,
# run the test_dataset.py file to view some examples and see the dimensions of your tensor.
dataset = Dataset(config)

# define the model
model = VanillaAutoEncoder(config)
model.train(False)

# load the model parameters
load(model, config)

# log to wandb
log = {}

recon = test_autoencoder(model, dataset, config)
log.update(
    {"Image reconstruction": wandb.Image(recon)}
)

gen = generate_using_encoder(model, config)
log.update(
    {"Image generation": wandb.Image(dataset.denormalize(gen))}
)

latent_im = plot_latent(model, dataset, config)
if latent_im is not None:
    log.update({"Latent visualisation": wandb.Image(latent_im)})

pca_im = plot_latent_pca(model, dataset, config)
if pca_im is not None:
    log.update({"PCA visualisation (2D)": wandb.Image(pca_im)})

pca_3d_im = plot_latent_pca_3d(model, dataset, config)
if pca_3d_im is not None:
    log.update({"PCA visualisation (3D)": wandb.Image(pca_3d_im)})

wandb.log(log)
