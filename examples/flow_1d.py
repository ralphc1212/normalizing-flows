import numpy as np
import scipy as sp
import scipy.stats
import itertools
import logging
import matplotlib.pyplot as plt

import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal, Uniform

sys.path.append('../')
from nf.flows import *
from nf.models import NormalizingFlowModel

def gen_data(n=512):
    return np.r_[np.random.randn(n // 2, 1) + np.array([2]),
                 np.random.randn(n // 2, 1) + np.array([-2])]

def plot_data(x, bandwidth = 0.2, **kwargs):
    kde = sp.stats.gaussian_kde(x[:,0])
    x_axis = np.linspace(-5, 5, 200)
    plt.plot(x_axis, kde(x_axis), **kwargs)

class data_lognormal:

    def __init__(self, location):
        with open(location+'/lognormal_100.out', 'r') as f:
            lines = f.readlines()

        self.all = torch.from_numpy(np.array([float(x) for x in lines])).unsqueeze(1).float()

        del lines
        f.close()


# try modification
if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=int(1e4), type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--flow", default="ActNorm", type=str)
    argparser.add_argument("--iterations", default=5000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flow = eval(args.flow)
    flows = [flow(dim=1) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(1), torch.eye(1)*10000)
    # prior = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    model = NormalizingFlowModel(prior, flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # x = torch.Tensor(gen_data(args.n))


    data = data_lognormal('/home/nandcui/data')
    x = data.all
    indices = torch.randperm(x.shape[0])[:10000]

    x = x[indices]

    test_x = data.all.copy_()

    plot_data(x, color = "black")
    plt.show()

    mean = torch.mean(x[:,0])
    std = torch.std(x[:,0])


    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - mean) / std

    for i in range(test_x.shape[1]):
        test_x[:,i] = (test_x[:,i] - mean) / std

    for i in range(args.iterations):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Logprob: {logprob.mean().data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plot_data(x, color="black", alpha=0.5)
    plt.title("Training data")
    plt.subplot(1, 3, 2)
    plot_data(z.data, color="darkblue", alpha=0.5)
    plt.title("Latent space")
    plt.subplot(1, 3, 3)
    samples = model.sample(500).data

    plot_data(samples, color="black", alpha=0.5)
    plt.title("Generated samples")
    plt.savefig("./ex_1d.png")

    z, prior_logprob, log_det = model(test_x)

    torch.set_printoptions(precision=20)

    print(data.all[:20])
    print(test_x[:20])
    print(z[:20])
