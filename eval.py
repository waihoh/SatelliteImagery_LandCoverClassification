import torch
import torch.nn.functional as F
from tqdm import tqdm
from diceloss import dice_coef_9cat_loss


def eval_net(net, loader, device):
    """
    Evaluation without the densecrf with the dice coefficient
    :param net: UNet
    :param loader: val_loader
    :param device: device
    :return: total / number of validation
    """

    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', disable=True, leave=True) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(input=mask_pred, target=true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coef_9cat_loss(y_true=true_masks, y_pred=pred).item()

            pbar.update()

    net.train()
    return tot/n_val
