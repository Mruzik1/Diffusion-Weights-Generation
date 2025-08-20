import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, split="train", weights=None):
        # compute Negative Log-Likelihood
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # compute Kullback-Leibler divergence
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # final VAE loss - ELBO
        loss = weighted_nll_loss + self.kl_weight * kl_loss
        log = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_loss": kl_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach().mean(),
        }
        return loss, log