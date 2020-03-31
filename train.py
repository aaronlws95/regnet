import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

from src.model import RegNet, get_num_parameters
from src.dataset import Kitti_Dataset

# Setup
os.environ['TORCH_HOME'] = os.path.join('D:\\', 'machine_learning')
start_epoch = 0

# Config
RUN_ID = 0
SAVE_PATH = str(Path('data')/'checkpoints'/'run_{:05d}'.format(RUN_ID))
LOG_PATH = str(Path('data')/'tensorboard'/'run_{:05d}'.format(RUN_ID))
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 5000
BATCH_SIZE = 4
SAVE_RATE = 1000
LOG_RATE = 10

# Dataset
dataset_params = {
    'base_path': Path('data')/'KITTI_SMALL',
    'date': '2011_09_26',
    'drives': [5],
    'h_fov': (-90, 90),
    'v_fov': (-24.9, 2.0),
    'd_rot': 1,
    'd_trans': 0.1,
}

dataset = Kitti_Dataset(dataset_params)
train_loader = DataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

# Model
model = RegNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.cuda()

# Load
# MODEL_ID = 1000
# model_load = torch.load(str(Path(SAVE_PATH)/'model_{:05d}.pth'.format(MODEL_ID)))
# model.load_state_dict(model_load['model_state_dict'])
# start_epoch = model_load['epoch']
# model.cuda()

# Tensorboard
writer = SummaryWriter(log_dir=LOG_PATH)

# Train
training_params = {
    'dataset_params': dataset_params,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
}

running_loss = 0.0
model.train()
for epoch in range(start_epoch, EPOCHS):
    for i, data in enumerate(train_loader):
        # Load data
        rgb_img = data['rgb'].cuda()
        depth_img = data['depth'].cuda()
        decalib_quat_real = data['decalib_real_gt'].cuda()
        decalib_quat_dual = data['decalib_dual_gt'].cuda()

        # Forward pass
        out = model(rgb_img, depth_img)

        # Zero optimizer
        optimizer.zero_grad()

        # Calculate loss
        real_loss = criterion(out[:, :4], decalib_quat_real)
        dual_loss = criterion(out[:, 4:], decalib_quat_dual)
        loss = real_loss + dual_loss

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Logging
        running_loss += loss.item()
        n_iter = epoch * len(train_loader) + i
        if n_iter % LOG_RATE == 0:
            print('Epoch: {:5d} | Batch: {:5d} | Loss: {:03f}'.format(epoch + 1, i + 1, running_loss / LOG_RATE))
            writer.add_scalar('Loss/train', running_loss / LOG_RATE, n_iter)
            running_loss = 0.0

        # Save model
        if n_iter % SAVE_RATE == 0:
            model_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_params': training_params,
            }
            torch.save(model_save, SAVE_PATH + '/model_{:05d}.pth'.format(n_iter))
# Save final model
model_save = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_params': training_params,
}
torch.save(model_save, SAVE_PATH + '/model_{:05d}.pth'.format(n_iter))