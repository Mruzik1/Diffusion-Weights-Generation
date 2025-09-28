import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from weights_encoding.modules import Encoder, Decoder
from weights_encoding.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from zoodatasets.basedatasets import reconstruct_weights


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        learning_rate,
        ckpt_path=None,
        ignore_keys=[],
        input_key="weight",
        cond_key="dataset",
        device="cuda",
        monitor=None
    ):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        if isinstance(input, dict):
            input_tensor = input[self.input_key].to(self.devices)
            model_properties = input.get('model_properties', None)
        else:
            input_tensor = input.to(self.devices)
            model_properties = None
            
        posterior = self.encode(input_tensor)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        
        # Return additional info for reconstruction if available
        if model_properties is not None:
            return input_tensor, dec, posterior, model_properties
        return input_tensor, dec, posterior

    def get_input(self, batch, k):
        x = batch[k].to(self.devices)
        
        # Handle ZooDataset weight format - already flattened and padded
        if len(x.shape) == 2:  # [batch_size, max_len]
            # Reshape for conv layers: [batch_size, 1, height, width]
            # Assuming square-like dimensions for conv operations
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            
            # Find appropriate height/width for reshaping
            # Use a reasonable aspect ratio for the weight data
            import math
            sqrt_len = int(math.sqrt(seq_len))
            height = sqrt_len
            width = seq_len // height
            
            # Pad if necessary to make it rectangular
            if height * width < seq_len:
                width += 1
                pad_size = height * width - seq_len
                x = F.pad(x, (0, pad_size), "constant", 0)
            
            x = x.view(batch_size, 1, height, width)
            
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        
        # Forward pass with batch to get model_properties if available
        forward_result = self(batch)
        reconstructions, posterior = forward_result[1], forward_result[2]

        if optimizer_idx == 0:
            # train encoder + decoder + logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train"
            )
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train"
            )

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        
        # Forward pass with batch to get model_properties if available
        forward_result = self(batch)
        reconstructions, posterior = forward_result[1], forward_result[2]
            
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val"
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val"
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def reconstruct_to_model_weights(self, reconstructed_weights, model_properties):
        if "layer_info" not in model_properties:
            return None
            
        batch_size = reconstructed_weights.shape[0]
        results = []
        
        for i in range(batch_size):
            # Flatten the reconstructed weights back to 1D
            weights_1d = reconstructed_weights[i].flatten()
            
            # Get layer info for this sample
            if isinstance(model_properties, list):
                layer_info = model_properties[i]["layer_info"]
            else:
                layer_info = model_properties["layer_info"]
            
            # Reconstruct using the imported function
            reconstructed_state_dict = reconstruct_weights(weights_1d, layer_info)
            results.append(reconstructed_state_dict)
        
        return results


class VAENoDiscModel(AutoencoderKL):
    def __init__(
        self, ddconfig,
        lossconfig,
        embed_dim,
        learning_rate,
        ckpt_path=None,
        ignore_keys=[],
        input_key="weight",
        cond_key="dataset",
        device="cuda"
    ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            input_key=input_key,
            cond_key=cond_key,
            learning_rate=learning_rate
        )
        self.devices = device

    def training_step(self, batch, batch_idx):
        forward_result = self(batch)
        inputs, reconstructions, posterior = forward_result[0], forward_result[1], forward_result[2]

        mse = F.mse_loss(inputs, reconstructions.unsqueeze(1))
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return aeloss + 1000.0 * mse

    def validation_step(self, batch, batch_idx):
        forward_result = self(batch)
        inputs, reconstructions, posterior = forward_result[0], forward_result[1], forward_result[2]
            
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )
        return optimizer


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        # TODO: Should be true by default but check to not break older stuff
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
