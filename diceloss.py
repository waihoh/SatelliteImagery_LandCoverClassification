import torch


def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    y_pred = torch.softmax(input=y_pred, dim=1)
    y_true_flatten = torch.flatten(input=torch.nn.functional.one_hot(y_true.to(torch.int64), num_classes=7))  # TODO: check num_classes.
    y_pred_flatten = torch.flatten(y_pred.permute(0, 2, 3, 1)) # TODO - check the dimensions of y_pred, i.e. channel first/last.

    intersect = torch.sum(input=y_true_flatten * y_pred_flatten, axis=-1)
    denominator = torch.sum(input=y_true_flatten + y_pred_flatten, axis=-1)
    return torch.mean(input=(2. * intersect / (denominator + smooth)))


def dice_coef_9cat_loss(y_true, y_pred):
    return 1 - dice_coef_9cat(y_true=y_true, y_pred=y_pred)
