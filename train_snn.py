from pathlib import Path
import logging
import argparse
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn
from utils.utils import letters
from utils.train_utils import get_datasets
import numpy as np
import random
from models.snn import SlayerMLP as model

# to reproduce the results
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


parser = argparse.ArgumentParser("Train BrailleLetter models.")

parser.add_argument("--epochs", type=int, help="Number of epochs.", default=10)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default="checkpoints/",
)
parser.add_argument(
    "--network_config",
    type=str,
    help="Path SNN network configuration.",
    default='configs/network.yml',
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument("--fold", type=int, help="Fold number to train from.", default=1)
parser.add_argument("--trial_number", type=int, help="Trial number.", default=1)
parser.add_argument("--hidden_size", type=int, help="Size of hidden layer (only true for two layer network).", default=128)
parser.add_argument("--batch_size", type=int, help="Batch Size.", default=16)
parser.add_argument("--val_freq", type=int, help="Runs validation at each val_freq.", default=10)
parser.add_argument("--gpu_id", type=int, help="Cuda device id.", default=0)



args = parser.parse_args()


# generic name for the model
model_name = f'model_{args.trial_number}_{args.fold}'
writer = SummaryWriter(Path(args.checkpoint_dir)/ model_name)
output_size = len(letters)

# check dataset

train_dataset, train_loader, val_dataset, val_loader = get_datasets(args.data_dir, args.fold, output_size, args.trial_number, batch_size=args.batch_size, test=False)

params = snn.params(args.network_config)

device = torch.device(f"cuda:{args.gpu_id}")
net = model(input_size=160, params=params, output_size=output_size).to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.5)
criteria = snn.loss(params).to(device)

def _train():
    correct = 0
    loss = 0
    net.train()
    for data, target, label in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = net.forward(data)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        spike_loss = criteria.numSpikes(output, target)
        loss += spike_loss.item()

        optimizer.zero_grad()
        spike_loss.backward()
        optimizer.step()

    loss_value = loss / len(train_dataset)
    acc_value = correct / len(train_dataset)
    
    writer.add_scalar("loss/train", loss_value, epoch)
    writer.add_scalar("acc/train", acc_value, epoch)
    
    return loss_value, acc_value
    
def _val():
    correct = 0
    loss = 0
    net.eval()
    with torch.no_grad():
        for data, target, label in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = net.forward(data)
            correct += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            spike_loss = criteria.numSpikes(output, target)  # numSpikes
            loss += spike_loss.item()

        loss_value = loss / len(val_dataset)
        acc_value = correct / len(val_dataset)
    
        writer.add_scalar("loss/val", loss_value, epoch)
        writer.add_scalar("acc/val", acc_value, epoch)
        
        return loss_value, acc_value

def _save_model(state):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(args.checkpoint_dir) / f"{model_name}_{state}.pt"
    torch.save(net.state_dict(), checkpoint_path)
    
    
pre_best_val_loss = 1e15
pre_best_val_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = _train()
    
    if epoch % args.val_freq == 0:
        val_loss, val_acc = _val()

        if val_loss < pre_best_val_loss:
            print('saving best val loss model')
            _save_model('bestLoss')
            pre_best_val_loss = val_loss
        if val_acc > pre_best_val_acc:
            print('saving best val accuracy model')
            _save_model('bestAcc')
            pre_best_val_acc = val_acc

_save_model('lastIter')
        
with open(Path(args.checkpoint_dir)/f'{model_name}_args.pkl', "wb") as f:
    pickle.dump(args, f)