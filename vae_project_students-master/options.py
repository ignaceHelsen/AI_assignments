import argparse


class CustomParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ### TRAINING OPTIONS ###
        self.parser.add_argument('--num_epochs', type=int, default=3)
        self.parser.add_argument('--learning_rate', type=float, default=0.001)
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--latent_dim', type=int, default=10)
        self.parser.add_argument('--dataset', type=str, default="MNIST",
                                 help="The dataset used. Choose between MNIST or CIFAR10.")
        self.parser.add_argument('--lambda', type=float, help="The lambda value of B-VAE.", default=1)

        ### MISC ###
        self.parser.add_argument('--random_seed', type=int, default=1)
        self.parser.add_argument('--batch_size_test', type=int, default=1000)
        self.parser.add_argument('--device', type=str, default="cuda")
        self.parser.add_argument('--name', type=str, help="The name of the model.", default="VAE")
        self.parser.add_argument('--save_path', type=str, default="experiments/")
        self.parser.add_argument('--load_path', type=str, default="experiments/")

    def parse(self, config: dict) -> dict:
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        for k, v in sorted(args.items()):
            if v is not None:
                config[k] = v

        return config
