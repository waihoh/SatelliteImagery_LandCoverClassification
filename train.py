import os
import sys

import torch
import torch.nn as nn
from torch import optim
import logging

from tqdm import tqdm
from eval import eval_net
from unet.unet import UNet

from utils.dataset import BasicDataset
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from classcount import classcount

dir_img = 'data/training_data/images/'
dir_mask = 'data/training_data/masks/'
dir_checkpoint = 'checkpoints/'
if not os.path.exists(dir_checkpoint):
    os.mkdir(dir_checkpoint)


def train_net(net, device, epochs=5, batch_size=1, lr=1e-3, val_percent=0.1, save_cp=True, img_scale=0.5):
    dataset = BasicDataset(imgs_dir=dir_img, masks_dir=dir_mask, scale=img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset=dataset, lengths=[n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training: 
    Epochs: {epochs}
    Batch size: {batch_size}
    Learning rate: {lr}
    Training size: {n_train}
    Validation size: {n_val}
    Checkpoints: {save_cp}
    Device: {device.type}
    Image scaling: {img_scale}
    ''')

    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-8)

    weights_classes = torch.from_numpy(classcount(train_loader))
    weights_classes = weights_classes.to(device, dtype=torch.float32)

    print("Class Distribution", weights_classes)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=weights_classes)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # use half precision model for training
                if torch.cuda.is_available():
                    net.half()

                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has {net.n_channels} input channels but images have {imgs.shape[1]} channels.'

                imgs_type = torch.float16 if torch.cuda.is_available() else torch.float32
                imgs = imgs.to(device=device, dtype=imgs_type)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long  # for cross entropy loss
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                # convert the prediction to float32 to avoid nan in loss calculation
                masks_pred = masks_pred.type(torch.float32)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix({'Epoch loss': epoch_loss/n_train})

                # convert model to full precision for optimization of weights
                net.float()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)  # TODO: what's the purpose of clipping here?
                optimizer.step()

                pbar.update(n=imgs.shape[0])
                global_step += 1

            writer.add_scalar(tag='Loss/train', scalar_value=epoch_loss/n_train, global_step=epoch+1)

            for tag, value in net.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag='weights/' + tag, values=value.data.cpu().numpy(), global_step=epoch+1)
                writer.add_histogram(tag='grads/' + tag, values=value.data.cpu().numpy(), global_step=epoch+1)

            val_score = eval_net(net=net, loader=val_loader, device=device)

            if net.n_classes > 1:
                logging.info(f'Validation CE Loss: {val_score}')
                writer.add_scalar(tag='Loss/test', scalar_value=val_score, global_step=epoch+1)
            else:
                logging.info(f'Validation Dice Coeff: {val_score}')
                writer.add_scalar(tag='Dice/test', scalar_value=val_score, global_step=epoch+1)

            writer.add_images(tag='images', img_tensor=imgs, global_step=epoch+1)
            if net.n_classes == 1:
                writer.add_images(tag='masks/true', img_tensor=true_masks, global_step=epoch + 1)
                writer.add_images(tag='masks/pred', img_tensor=torch.sigmoid(masks_pred) > 0.5, global_step=epoch + 1)

            if (epoch+1) % 5 == 0:
                if save_cp:
                    torch.save(net.state_dict(), dir_checkpoint + f"CP_epoch{epoch+1}.pth")
                    logging.info(f'Checkpoint {epoch+1} saved.')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=7, bilinear=True)

    logging.info(f'Network:\n' 
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=device)

    try:
        train_net(net=net, epochs=5, batch_size=4, device=device, img_scale=0.5, val_percent=0.1)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_checkpoint + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
