import torch
from torch import nn
from torchvision import ops


class SetCriterion(nn.Module):
    def __init__(self, losses, weight_dict):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def loss_edge(self, outputs, targets):
        """
            Compute the losses related to the edges: the hed loss.
            targets dicts must contain the key "edge_maps" containing a tensor of dim [2, h, w]
        """
        assert "edge_pred" in outputs

        src_edges = outputs['edge_pred'] # [batch_size, time, 2, out_h, out_w]

        target_edges = targets[0]['edge_maps'].unsqueeze(0)

        losses = {
            'loss_edge': self.weight_dict['loss_hed'] * hed_loss(src_edges, target_edges) +
                         self.weight_dict['loss_focal'] * ops.sigmoid_focal_loss(src_edges.float(), target_edges.float(),
                                                                                 gamma=0.5, reduction='sum')
        }
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'edge': self.loss_edge
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


def hed_loss(output, label):
    """ Calculate sum of weighted cross entropy loss. """
    total_loss = 0
    b, t, c, w, h = label.shape
    output = torch.sigmoid(output)
    for i in range(t):
        for j in range(c):
            p = output[:, i, j, :, :].unsqueeze(1)
            t = label[:, i, j, :, :].unsqueeze(1)
            w = 1
            loss = w * weighted_cross_entropy_loss(p, t)
            total_loss = total_loss + loss

    total_loss = total_loss / b * 1.0
    return total_loss


def weighted_cross_entropy_loss(preds, edges):
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    if imh<320 or imw<320:
        print('error')
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(1)
        t = edges[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float() + 1e-6  # Shape: [b,].
        num_neg = c * h * w - num_pos + 1e-6  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss


if __name__ == "__main__":
    src = torch.zeros((1, 5, 2, 4, 4))
    edge = torch.ones((1, 5, 2, 4, 4))
    hed_loss(src, edge)