
import torch
import datetime as dt

from pathlib import Path
from typing import Union
from torch.utils.tensorboard import SummaryWriter

class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def _is_aws_or_gcloud_path(tb_log_dir: str) -> bool:
    return tb_log_dir.startswith("gs://") or tb_log_dir.startswith("s3://")

def _make_path_if_local(tb_log_dir: Union[str, Path]) -> Union[str, Path]:
    if isinstance(tb_log_dir, str) and _is_aws_or_gcloud_path(tb_log_dir):
        return tb_log_dir

    tb_log_dir = Path(tb_log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return tb_log_dir

class Logger():
    def __init__(self, path, method):
        self.path = path
        date = dt.datetime.now()
        date = date.strftime("%Y_%m_%d_%H_%M_%S")

        tb_log_dir = _make_path_if_local(self.path)
        tb_log_dir = self.path + '/' + method
        tb_log_dir = _make_path_if_local(tb_log_dir)
        tb_log_dir = self.path + '/' + method + '/' + date
        tb_log_dir = _make_path_if_local(tb_log_dir)
        self.logger = SummaryWriter(tb_log_dir)

    def result(self, title, log_data, n_iter):
        self.logger.add_scalar(title, log_data, n_iter)

    def config(self, config, metric_dict):
        config = vars(config)
        self.logger.add_hparams(config, metric_dict, run_name=None)

__all__ = ['AverageMeter', 'Logger']