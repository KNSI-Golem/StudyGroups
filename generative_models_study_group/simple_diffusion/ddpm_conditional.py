import os
import copy
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA, Discriminator
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, noise=None, cfg_scale=3, batch=8):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if noise is None:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            else:
                x = noise

            data = torch.utils.data.TensorDataset(x, labels)
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False)
            generated_xs = []
            for x, labels in dataloader:
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                generated_xs.append(x)
        
        generated_xs = torch.cat(generated_xs)
        model.train()  
        return generated_xs
    

class DiffusionAdv(Diffusion):
    def sample(self, model, n, labels, noise=None, cfg_scale=3, batch=8):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if noise is None:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            else:
                x = noise

            data = torch.utils.data.TensorDataset(x, labels)
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False)
            generated_xs = []
            for x, labels in dataloader:
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                    predicted_img = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_img = model(x, t, None)
                        predicted_img = torch.lerp(uncond_predicted_img, predicted_img, cfg_scale)
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = predicted_img
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                generated_xs.append(x)
        
        generated_xs = torch.cat(generated_xs)
        model.train()  
        return generated_xs


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    n_to_plot = args.n_to_plot
    viz_noise = torch.randn((n_to_plot, 3, args.image_size[0], args.image_size[1])).to(args.device)
    epochs_per_plot = args.epochs_per_plot
    viz_labels = torch.arange(n_to_plot // 4).long().repeat(4).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % epochs_per_plot == 0:
            sampled_images = diffusion.sample(model, n=n_to_plot, labels=viz_labels, noise=viz_noise)
            ema_sampled_images = diffusion.sample(ema_model, n=n_to_plot, labels=viz_labels, noise=viz_noise)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"), nrow=4)
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"), nrow=4)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

def train_with_adverserial_objective(args):
    '''The same as train function but uses adverserial objective (and Discriminator class) instead of MSE'''
    setup_logging_adv(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    discriminator = Discriminator(num_classes=args.num_classes).to(device)
    discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=args.lr)
    adverserial_loss = nn.BCELoss()

    n_to_plot = args.n_to_plot
    viz_noise = torch.randn((n_to_plot, 3, args.image_size[0], args.image_size[1])).to(args.device)
    epochs_per_plot = args.epochs_per_plot
    viz_labels = torch.arange(n_to_plot // 4).long().repeat(4).to(device)

    adv_weight = 0.1

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None

            # discriminator loss
            predicted_noise = model(x_t, t, labels)

            errD_fake = adverserial_loss(discriminator(predicted_noise, t, labels), torch.zeros(predicted_noise.shape[0]).to(device))

            errD_real = adverserial_loss(discriminator(noise, t, labels), torch.ones(predicted_noise.shape[0]).to(device))

            D_loss = errD_fake + errD_real
            D_loss = D_loss * adv_weight
            discriminator_optimizer.zero_grad()
            D_loss.backward()
            discriminator_optimizer.step()

            # generator loss
            predicted_noise = model(x_t, t, labels)
            mse_loss = mse(noise, predicted_noise)

            G_loss = adverserial_loss(discriminator(predicted_noise, t, labels),
                                        torch.ones(predicted_noise.shape[0]).to(device))

            G_total_loss = G_loss * adv_weight + mse_loss * (1.0 - adv_weight)
            optimizer.zero_grad()
            G_total_loss.backward()
            optimizer.step()

            # logging
            D_loss = D_loss.mean().item()
            G_loss = G_loss.mean().item()
            mse_loss = mse_loss.mean().item()
            
            ema.step_ema(ema_model, model)

            pbar.set_postfix(loss_D=D_loss, loss_G=G_loss, mse_loss=mse_loss)
            logger.add_scalar("loss_D", D_loss, global_step=epoch * l + i)
            logger.add_scalar("loss_G", G_loss, global_step=epoch * l + i)
            logger.add_scalar("loss_mse", mse_loss, global_step=epoch * l + i)

        if epoch % epochs_per_plot == 0:
            sampled_images = diffusion.sample(model, n=n_to_plot, labels=viz_labels, noise=viz_noise)
            ema_sampled_images = diffusion.sample(ema_model, n=n_to_plot, labels=viz_labels, noise=viz_noise)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results_adv", args.run_name, f"{epoch}.jpg"), nrow=4)
            save_images(ema_sampled_images, os.path.join("results_adv", args.run_name, f"{epoch}_ema.jpg"), nrow=4)
            torch.save(model.state_dict(), os.path.join("models_adv", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models_adv", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models_adv", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 1000
    args.batch_size = 64
    args.image_size = (48, 32)
    args.num_classes = 67
    args.dataset_path = './data/margonem'
    args.device = "cuda"
    args.lr = 3e-4
    args.n_to_plot = 16
    args.epochs_per_plot = 20
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

