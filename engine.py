from typing import Iterable
import torch
import torchvision.transforms as T
import util.misc as utils


transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, flow_maps, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        flow_maps = flow_maps.to(device)
        targets = utils.targets_to(targets, device)

        outputs = model(samples, flow_maps)

        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value = losses.item()
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
