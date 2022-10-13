import random

import torch.cuda

import models
import utils
from options import CustomParser
from utils import *

print(torch.cuda.is_available())

config = CustomParser().parse({})

# set all random seeds for reproducibility
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed(config["random_seed"])
random.seed(config["random_seed"])

# set device
if config["device"] == "cuda" and torch.cuda.is_available():
    config["device"] = torch.device("cuda:0")
else:
    config["device"] = torch.device("cpu")

wandb.init(project="VAE_project", entity="the-elite-schakels", name=config["name"] + "_train")

# Let's first prepare the MNIST dataset,
# run the test_dataset.py file to view some examples and see the dimensions of your tensor.
dataset = Dataset(config)

# TODO: define and train the model. Use the function from utils.py
model = models.VariationalAutoEncoder(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
utils.train_vae(model, config, dataset, optimizer)

# save the model
save(model, config)

# display some images with its reconstruction
log = {}

recon = test_vae(model, dataset, config)
log.update(
    {"Image reconstruction": wandb.Image(recon)}
)

gen = generate_using_encoder(model, config)
log.update(
    {"Image generation": wandb.Image(dataset.denormalize(gen))}
)

wandb.log(log)
