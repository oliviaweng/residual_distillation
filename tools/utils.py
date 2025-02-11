import os, shutil, torch, copy, sys, PIL, math, random, time, pdb
import torchvision.transforms as transforms
import numpy                  as np
from pathlib import Path
import re

def _data_transforms_cifar10(args):
    CIFAR_MEAN      = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD       = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    if args.cutout_length > 0:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform

def save_checkpoint(model, optimizer, loss, save_dir, epoch=None, is_best=False, pre = None):

    ckpt = os.path.join(save_dir, "{}_checkpoint_loss={:.4f}.pth.tar".format(pre, loss))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckpt)
    # Rename previous last checkpoint(s) for removal later
    last_checkpoints = list(Path(save_dir).rglob("*last.pth.tar"))
    for l in last_checkpoints:
        l = str(l)
        os.rename(l, l + '.old')
    # Save last checkpoint
    last_ckpt = os.path.join(save_dir, "{}_last.pth.tar".format(pre))
    shutil.copyfile(ckpt, last_ckpt)
    # Remove old last checkpoints
    last_checkpoints = list(Path(save_dir).rglob("*last.pth.tar.old"))
    for l in last_checkpoints:
        print(f'Removing old last checkpoint: {l}')
        os.remove(l)
    # Save best model (according to highest validation acc)
    if is_best:
        best_ckpt = os.path.join(save_dir, "{}_best.pth.tar".format(pre))
        shutil.copyfile(ckpt, best_ckpt)
    # Only save the top 3 best models
    save_top_k = 3 
    checkpoints = list(Path(save_dir).rglob("*.pth.tar"))
    best_k_models = {}
    for checkpoint in checkpoints:
        checkpoint = str(checkpoint)
        index = checkpoint.find('=')
        if index != -1:
            index = index + len('=')
            match = re.search('[A-z]', checkpoint[index:])
            if match:
                curr_ckpt_loss = checkpoint[index : index + match.start() - 1]
                # -1 due to . in filename --------------------------------^^^
                best_k_models[checkpoint] = float(curr_ckpt_loss)
    if len(best_k_models) < 1:
        return # No saved checkpoints yet
    
    best_k_models_sorted = sorted(best_k_models, key=best_k_models.get)
    num_models_to_delete = len(best_k_models) - save_top_k
    for _ in range(num_models_to_delete):
        model_to_del = best_k_models_sorted.pop(-1)
        best_k_models.pop(model_to_del)
        os.remove(model_to_del)

        
class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

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
        
def param_size(model):
    """Count parameter size in MB."""
    num_params = np.sum(
        np.prod(v.size()) for k, v in model.named_parameters() 
        if not k.startswith("_aux_head"))
    return num_params / 1024. / 1024.

def accuracy(output, target, topk=(1,)):
    """Compute precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def prepare_logger(xargs):
    args = copy.deepcopy( xargs )
    from tools.logger import Logger
    logger = Logger(args.save_dir, args.seed)
    logger.log('Main Function with logger : {:}'.format(logger))
    logger.log('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    # Set CUDA_VISIBLE DEVICSE to number of available GPUs
    cuda_visible_devices = []
    for i in range(torch.cuda.device_count()):
        cuda_visible_devices.append(i)
    cuda_visible_devices_str = ','.join(map(str, cuda_visible_devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices_str
    logger.log("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))
    return logger

def adjust_learning_rateB(optimizer, epoch, lr_max, lr_min, stone, epoch_max, batch_pro, paramdict, paramname):
    def CosineAnnealingLR(lr_min, lr_max, epoch, epoch_max):
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / epoch_max)) / 2.
        return lr
    if epoch >= stone or paramname is None:
        lr = CosineAnnealingLR(lr_min / 10., lr_min, epoch - stone, epoch_max - stone)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = CosineAnnealingLR(lr_min, lr_max, epoch % batch_pro, batch_pro)
        indexs = []
        for name in paramname:
            indexs.append(paramdict[name])
        for ind, param_group in enumerate(optimizer.param_groups):
            if ind in indexs:
                param_group['lr'] = lr
    return lr

def adjust_learning_rateC(optimizer, epoch, lr_max, lr_min, stone, epoch_max, batch_pro):
    
    def CosineAnnealingLR(lr_max, lr_min, epoch, epoch_max):
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / epoch_max)) / 2.
        return lr
    if epoch >= stone:
        lr = CosineAnnealingLR(1e-4, lr_min, epoch - stone, epoch_max - stone)
    else:
        lr = CosineAnnealingLR(lr_max, lr_min, epoch, batch_pro)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr      
    return lr

        
def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def adjust_learning_rateA(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rateS(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    for step in args.stone:
        if (epoch >= step):
              lr = lr * 0.1
        else:
              break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rateD(optimizer, epoch, epoch_max, lr_max, lr_min):
    
    def CosineAnnealingLR(lr_max, lr_min, epoch, epoch_max):
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / epoch_max)) / 2.
        return lr
    
    lr = CosineAnnealingLR(lr_max, lr_min, epoch, epoch_max)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string