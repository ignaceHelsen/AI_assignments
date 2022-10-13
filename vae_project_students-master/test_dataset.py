from dataset import Dataset
from options import CustomParser

config = CustomParser().parse({})
dataset = Dataset(config)
dataset.show_examples()
