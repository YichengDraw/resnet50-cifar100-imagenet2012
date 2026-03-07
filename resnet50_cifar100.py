import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import numpy as np
import os
import random
from tqdm import tqdm
import math

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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


set_seed(42)

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):
        super(Bottleneck, self).__init__()
        # 1*1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3*3
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1*1
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # activition
        self.relu = nn.ReLU(inplace=True)

        # shortcut
        self.shortcut = nn.Sequential()

        # projection shortcut for dimensions or strides don't match
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # stochastic depth
        self.drop_rate = drop_rate

    def forward(self, x):
        # main path
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # PS: not relu yet

        # stochastic depth
        if self.training and self.drop_rate > 0:
            survival_rate = 1 - self.drop_rate
            if torch.rand(1).item() < survival_rate:  # zero out the block output
                out = out / survival_rate  # scale to maintain mean
            else:
                out = out * 0.0

        # residual connection
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes=100, drop_rate=0.2, stochastic_depth_rate=0.2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        depths = [3, 4, 6, 3]
        total_blocks = sum(depths)
        drop_rates = [x * stochastic_depth_rate / total_blocks for x in range(total_blocks)]

        block_idx = 0

        self.layer1 = self._make_layer(block, 64, depths[0], stride=1,
                                       drop_rates=drop_rates[block_idx:block_idx + depths[0]])
        block_idx += depths[0]

        self.layer2 = self._make_layer(block, 128, depths[1], stride=2,
                                       drop_rates=drop_rates[block_idx:block_idx + depths[1]])
        block_idx += depths[1]

        self.layer3 = self._make_layer(block, 256, depths[2], stride=2,
                                       drop_rates=drop_rates[block_idx:block_idx + depths[2]])
        block_idx += depths[2]

        self.layer4 = self._make_layer(block, 512, depths[3], stride=2,
                                       drop_rates=drop_rates[block_idx:block_idx + depths[3]])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, drop_rates):
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
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_resnet50(num_classes=100, drop_rate=0.2, stochastic_depth_rate=0.2):
    return ResNet(Bottleneck, num_classes=num_classes, drop_rate=drop_rate, stochastic_depth_rate=stochastic_depth_rate)


class CIFAR100Augmentation:
    def __init__(self, is_training=True):
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),

                transforms.RandomErasing(p=0.2, scale=(0.02, 0.25), ratio=(0.3, 3.3))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            ])

    def __call__(self, x):
        return self.transform(x)


def cutmix_data(x, y, alpha=1.0):
    batch_size, channels, height, width = x.size()

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    lam = max(0.1, min(0.9, lam))
    index = torch.randperm(batch_size).to(x.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    actual_cut_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = width * height
    lam = 1 - (actual_cut_area / total_area)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_cutmix_accuracy(outputs, targets_a, targets_b, lam):
    _, predicted = outputs.max(1)
    correct_a = predicted.eq(targets_a).float()
    correct_b = predicted.eq(targets_b).float()
    accuracy = lam * correct_a + (1 - lam) * correct_b
    return accuracy.sum().item()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)

        nll_loss = -pred.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth_loss = -pred.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = (self.current_epoch + 1) / self.warmup_epochs * self.base_lr
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1


class Config:
    drop_rate = 0.2
    drop_path_rate = 0.2

    batch_size = 128
    num_epochs = 300
    learning_rate = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    nesterov = True

    warmup_epochs = 5

    cutmix_alpha = 1.0
    cutmix_prob = 0.8
    label_smoothing = 0.1

    data_dir = os.environ.get('CIFAR100_DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
    checkpoint_path = os.environ.get(
        'RESNET50_CIFAR100_CHECKPOINT',
        os.path.join(OUTPUT_ROOT, 'checkpoints', 'resnet50_cifar100_best.pth')
    )
    log_dir = os.environ.get(
        'RESNET50_CIFAR100_LOG_DIR',
        os.path.join(OUTPUT_ROOT, 'runs', 'resnet50_cifar100')
    )

    num_workers = 0 if os.name == 'nt' else 4
    pin_memory = True
    print_freq = 50

    patience = 40
    min_delta = 0.01



def get_data_loaders(config):
    train_transform = CIFAR100Augmentation(is_training=True)
    test_transform = CIFAR100Augmentation(is_training=False)

    train_dataset = datasets.CIFAR100(
        root=config.data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=config.data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        drop_last=True, persistent_workers=config.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, scaler, config, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    cutmix_applied = 0

    pbar = tqdm(train_loader, desc=f'Epoch{epoch + 1}/{config.num_epochs}')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_cutmix = config.cutmix_prob > 0 and torch.rand(1) < config.cutmix_prob

        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, config.cutmix_alpha)
            cutmix_applied += 1

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)

            if use_cutmix:
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total += targets.size(0)

        if use_cutmix:
            correct += calculate_cutmix_accuracy(outputs, targets_a, targets_b, lam)
        else:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        if batch_idx % config.print_freq == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'CutMix': f'{cutmix_applied}/{batch_idx + 1}'
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    cutmix_rate = cutmix_applied / len(train_loader)

    return epoch_loss, epoch_acc, cutmix_rate


def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def main():
    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')

    print(f'Loading CIFAR-100 dataset...')
    train_loader, test_loader = get_data_loaders(config)

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    print('Creating ResNet-50 model...')
    model = create_resnet50(
        num_classes=100,
        drop_rate=config.drop_rate,
        stochastic_depth_rate=config.drop_path_rate
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = nn.DataParallel(model)

    if config.label_smoothing > 0:
        criterion = LabelSmoothingLoss(classes=100, smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.num_epochs
    )

    scaler = GradScaler()

    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    writer = SummaryWriter(config.log_dir)

    print('\nStarting training...')
    print('=' * 80)

    best_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'learning_rate': [], 'cutmix_rate': []
    }

    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
        print('-' * 40)

        train_loss, train_acc, cutmix_rate = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch, device
        )

        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('CutMix/Application_rate', cutmix_rate, epoch)

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(current_lr)
        training_history['cutmix_rate'].append(cutmix_rate)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f} → {new_lr:.6f}')
        print(f'CutMix Applied: {cutmix_rate:.1%} of batches')

        if val_acc > best_acc + config.min_delta:
            best_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
                'training_history': training_history
            }

            torch.save(checkpoint, config.checkpoint_path)
            print(f'NEW BEST MODEL saved! Validation accuracy:{best_acc:.2f}%')

        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{config.patience}')

        if patience_counter >= config.patience:
            print(f'\n Early stopping triggered after {epoch + 1} epochs')
            print(f'Best validation accuracy: {best_acc:.2f}%')
            break

        print('=' * 80)

    print('\n Training completed!')
    print(f'Best validation accuracy achieved: {best_acc:.2f}%')

    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc = validate_epoch(model, test_loader, criterion, device)

    print(f'Final Results:')
    print(f'Test Loss: {final_loss:.4f}')
    print(f'Test Accuracy: {final_acc:.2f}%')
    print(f'Best Epoch: {checkpoint["epoch"] + 1}')

    writer.close()
    print('\n Training completed successfully!')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n Training interrupted by user')
    except Exception as e:
        print(f'\n Training failed with error: {e}')
        raise

