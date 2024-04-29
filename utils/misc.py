import torch
import os

__all__ = ['AverageMeter', 'get_optimizer', 'save_checkpoint']


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               args.lr)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                   args.lr)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(),
                                args.lr)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点信息
    filepath = os.path.join(checkpoint, filename)
    print('checkpoint filepath = ', filepath)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        model_name = 'garbage_resnext101_model_' + str(state['epoch']) + '_' + str(
            int(round(state['train_acc'] * 100, 0))) + '_' + str(
            int(round(state['test_acc'] * 100, 0))) + '.pth'
        print('Validation loss decreased  Saving model ..,model_name = ', model_name)
        model_path = os.path.join(checkpoint, model_name)
        print('model_path = ', model_path)
        torch.save(state['state_dict'], model_path)
