import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import numpy as np
import os
import random

from tqdm import tqdm
import math
import time
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'imagenet')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, stochastic_depth_rate=0.1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stochastic_depth_rate = stochastic_depth_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # !forgot
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)  # !forgot
        x = self.conv3(x)
        x = self.bn3(x)
        # !x = self.relu(x)  redundant relu before shortcut mistakenly

        if self.training and self.stochastic_depth_rate > 0:
            if torch.rand(1).item() >= self.stochastic_depth_rate:
                x = x / (1 - self.stochastic_depth_rate)
            else:
                x = x * 0.0

        x += self.shortcut(identity)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, drop_rate=0.1, stochastic_depth_rate=0.1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        total_blocks = sum(layers)
        drop_rates = [i / total_blocks * stochastic_depth_rate for i in range(total_blocks)]

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1,
                                       drop_rates=drop_rates[0:layers[0]])
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2,
                                       drop_rates=drop_rates[layers[0]:layers[0] + layers[1]])
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2,
                                       drop_rates=drop_rates[layers[0] + layers[1]:layers[0] + layers[1] + layers[2]])
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2,
                                       drop_rates=drop_rates[layers[0] + layers[1] + layers[2]:
                                                             layers[0] + layers[1] + layers[2] + layers[3]])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()  # !forgot to implement

    def _make_layer(self, block, num_blocks, out_channels, stride, drop_rates):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, stride in enumerate(strides):
            layers.append(block(self.in_channels, out_channels, stride, drop_rates[i]))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_resnet50(num_classes=1000, drop_rate=0.1, stochastic_depth_rate=0.1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, drop_rate=drop_rate,
                  stochastic_depth_rate=stochastic_depth_rate)


class ImageNetAugmentation:
    def __init__(self, is_training=True, image_size=224):
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __call__(self, x):
        return self.transform(x)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)  # !size[0] mistakenly
    index = torch.randperm(batch_size).to(x.device)

    mixed = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smooth=0.1):
        super().__init__()
        self.smooth = smooth
        self.confidence = 1.0 - smooth
        self.classes = classes

    def forward(self, logits, target):
        pred = F.log_softmax(logits, dim=-1)
        # return a [N,] tensor with indexed pred as elements([N,C]->[N,])
        nll_loss = -pred.gather(dim=1, index=target.unsqueeze(1)).squeeze()
        # return a [N,] tensor with mean of each pred as elements([N,C]->[N,])
        smooth_loss = -pred.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smooth * smooth_loss  # mixed loss
        return loss.mean()  # vector -> number


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def _calculate_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs * self.base_lr
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self._calculate_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1

    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'current_epoch': self.current_epoch,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr
        }

    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.current_epoch = state_dict['current_epoch']
        self.min_lr = state_dict['min_lr']
        self.base_lr = state_dict['base_lr']

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Config:
    drop_rate = 0.1
    stochastic_depth_rate = 0.1

    mixup_alpha = 0.2
    mixup_prob = 0.5

    print_freq = 100

    train_dir = os.environ.get('IMAGENET_TRAIN_DIR', os.path.join(DEFAULT_DATA_ROOT, 'train'))
    val_dir = os.environ.get('IMAGENET_VAL_DIR', os.path.join(DEFAULT_DATA_ROOT, 'val'))

    batch_size = 256

    num_workers = 0 if os.name == 'nt' else 8
    pin_memory = True

    label_smoothing_rate = 0.1

    optimizer_type = 'sgd'

    adamw_lr = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-8

    sgd_lr = 0.1
    momentum = 0.9
    nesterov = True

    weight_decay = 1e-4

    warmup_epochs = 5
    total_epochs = 120
    min_lr = 1e-6

    checkpoint_path = os.environ.get(
        'RESNET50_IMAGENET_CHECKPOINT',
        os.path.join(OUTPUT_ROOT, 'checkpoints', 'resnet50_imagenet_best.pth')
    )
    log_dir = os.environ.get(
        'RESNET50_IMAGENET_LOG_DIR',
        os.path.join(OUTPUT_ROOT, 'runs', 'resnet50_imagenet')
    )

    min_delta = 0.01

    patience = 15



def get_dataloaders(config, distributed=False):
    train_transform = ImageNetAugmentation(is_training=True)
    val_transform = ImageNetAugmentation(is_training=False)

    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transform)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,  # !sampler wrote batch_sampler mistakenly
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,  # !sampler wrote batch_sampler mistakenly
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # !

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # !
        res.append(correct_k.mul_(100.0 / batch_size))  # !
    return res


def train_epoch(model, config, train_loader, epoch, device, criterion, scaler, optimizer, distributed=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    if distributed and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)  # !sampler belongs to train_loader

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}') if not distributed or dist.get_rank() == 0 else train_loader

    for batch_idx, (inputs, targets) in enumerate(pbar):
        data_time.update(time.time() - end)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mixup = config.mixup_prob > 0 and torch.rand(1) < config.mixup_prob  # !both conditions got wrong
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.mixup_alpha)  # !第一项就是inputs

        with autocast():  # !() forgot
            outputs = model(inputs)

            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        optimizer.zero_grad()  # !(spell)zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if not use_mixup:
            res1, res2 = accuracy(outputs, targets, (1, 5))
            top1.update(res1[0], targets.size(0))  # ![0] forgot
            top5.update(res2[0], targets.size(0))

        losses.update(loss.item(), targets.size(0))  # !.item() & targets.size(0) forgot
        batch_time.update(time.time() - end)
        end = time.time()

        if not distributed or dist.get_rank() == 0:
            if batch_idx % config.print_freq == 0:
                pbar.set_postfix({  # !set_ forgot
                    'Losses': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.2f}%',  # !unit forgot
                    'Top5': f'{top5.avg:.2f}%',
                    'Time': f'{batch_time.avg:.3f}s'  # !unit forgot
                })

    return losses.avg, top1.avg, top5.avg  # ! avg forgot


def validate(model, val_loader, device, criterion, distributed=False):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():  # !torch. forgot
        end = time.time()
        pbar = tqdm(val_loader, desc='Validate') if not distributed or dist.get_rank() == 0 else val_loader

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)  # !forgot to transfer into GPU
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)  # !out of autocast mistakenly
                loss = criterion(outputs, targets)

            res1, res2 = accuracy(outputs, targets, (1, 5))
            top1.update(res1[0], targets.size(0))
            top5.update(res2[0], targets.size(0))
            losses.update(loss.item(), targets.size(0))  # !.item() forgot

            batch_time.update(time.time() - end)
            end = time.time()

            if not distributed or dist.get_rank() == 0:
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.2f}%',
                    'Top5': f'{top5.avg:.2f}%',
                    'Time': f'{batch_time.avg:.3f}s'
                })

    return losses.avg, top1.avg, top5.avg


def resume_training(checkpoint_path, device, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        print(f'Resuming training from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:  # !scheduler needs a check for the existence of its state_dict,lest forgot saving
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler.current_epoch = checkpoint['epoch'] + 1
            print(f'Warning: No scheduler state found, setting epoch manually!')  # !

        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint['best_acc1']

        print(f'Resuming from epoch {start_epoch + 1} with Top1 best accuracy={best_acc1:.2f}%')
        print(f'Scheduler state: current epoch={scheduler.current_epoch}')  # !
        return start_epoch, best_acc1
    else:
        init_lr=scheduler._calculate_lr(0)
        for param_group in optimizer.param_groups:
            param_group['lr']=init_lr
        print('No checkpoint found, starting from scratch!')
        return 0, 0.0


def main():
    config = Config()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')  # !total_ forgot,

    train_loader, val_loader = get_dataloaders(config)
    print(f'Training samples: {len(train_loader.dataset):,}')
    print(f'Validate samples: {len(val_loader.dataset):,}')
    print(f'Total classes: {len(train_loader.dataset.classes)}')

    model = create_resnet50(1000, config.drop_rate, config.stochastic_depth_rate)
    model = model.to(device)  # !forgot

    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')  # !() after numel forgot

    if torch.cuda.device_count() > 1:  # !>0 mistakenly
        model = nn.DataParallel(model)  # !forgot the syntax
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')

    if config.label_smoothing_rate > 0:
        criterion = LabelSmoothingLoss(1000, config.label_smoothing_rate)
    else:
        criterion = nn.CrossEntropyLoss()

    if config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.adamw_lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.sgd_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )

    scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.total_epochs, config.min_lr)

    start_epoch, best_acc1 = resume_training(config.checkpoint_path, device, model, optimizer, scheduler)

    if start_epoch == 0:
        scheduler.step()

    if start_epoch > 0:
        expected_lr = scheduler.get_current_lr()
        actual_lr = optimizer.param_groups[0]['lr']
        print(f'Scheduler validation - Expected LR: {expected_lr:.6f}, Actual LR: {actual_lr:.6f}')

        if abs(expected_lr - actual_lr) > 1e-8:
            for param_group in optimizer.param_groups:
                param_group['lr'] = expected_lr
            print(f'Warning: non-sync discovered, learning rate is manually set to {expected_lr:.6f}!')

    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    writer = SummaryWriter(config.log_dir)

    scaler = GradScaler()

    patience_counter = 0

    print('\nStarting training...')
    print('=' * 80)

    for epoch in range(start_epoch, config.total_epochs):  # !range() forgot
        print(f'\nEpoch {epoch + 1}/{config.total_epochs}')
        print('-' * 60)

        train_loss, train_acc1, train_acc5 = train_epoch(model, config, train_loader, epoch, device, criterion, scaler,
                                                         optimizer)
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, device, criterion)  # !criterion, device: wrong order

        current_lr = optimizer.param_groups[0]['lr']  # !may not best practice: scheduler.get_current_lr
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validate', val_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top1', train_acc1, epoch)
        writer.add_scalar('Accuracy/Train_Top5', train_acc5, epoch)
        writer.add_scalar('Accuracy/Val_Top1', val_acc1, epoch)
        writer.add_scalar('Accuracy/Val_Top5', val_acc5, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(
            f'Training Loss: {train_loss:.4f} | Training Top1 Accuracy: {train_acc1:.2f}% | Training Top5 Accuracy: {train_acc5:.2f}%')
        print(
            f'Validate Loss: {val_loss:.4f} | Validate Top1 Accuracy: {val_acc1:.2f}% | Validate Top5 Accuracy: {val_acc5:.2f}%')
        print(f'Learning Rate: {current_lr:.6f} -> {new_lr:.6f}')

        if val_acc1 > best_acc1 + config.min_delta:
            best_acc1 = val_acc1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'config': config  # !forgot
            }

            torch.save(checkpoint, config.checkpoint_path)

            print(f'NEW BEST MODEL saved with Best Top1 Accuracy: {best_acc1:.2f}%')

        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{config.patience}')

        if patience_counter >= config.patience:
            print(f'\nEarly stopped after {epoch + 1} epochs of training')
            print(f'Best Top1 Accuracy: {best_acc1:.2f}%')
            break

        print('=' * 80)

    print('\nTraining completed!')
    print(f'Best Top1 Accuracy: {best_acc1:.2f}%')

    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc1, final_acc5 = validate(model, val_loader, device,
                                                  criterion)  # !criterion, device: wrong order

    print(
        f'\nFinal Loss: {final_loss} | Final Top1 Accuracy: {final_acc1:.2f}% | Final Top5 Accuracy: {final_acc5:.2f}%')
    print(f'Best Epoch: {checkpoint["epoch"] + 1}')  # !forgot to implement

    writer.close()

    print('\nTraining completed successfully!')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nException interrupt: {e}')
        raise

