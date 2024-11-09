import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        self.exp_l1 = nn.L1Loss(reduction=reduction)

    def __call__(self, motions, predicted_motions, split_name):
        loss = 0

        batch_size = predicted_motions["exp"].shape[0]

        lip_keypoints = [6, 12, 14, 17, 19, 20]
        lip_exp_l1 = self.exp_l1(predicted_motions["exp"].reshape(batch_size, -1, 3)[:, :6, :],
                                 motions["exp"].reshape(batch_size, -1, 3)[:, lip_keypoints, :])
        loss += 1 * lip_exp_l1


        log = {f"{split_name}/lip_exp_l1" : lip_exp_l1.clone().detach().mean(),
               f"{split_name}/basic_loss" : loss.clone().detach().mean()}
        
        return loss, log
