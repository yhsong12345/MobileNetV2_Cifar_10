import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from engine import *
from tqdm.auto import tqdm
from model import *
from datasets import create_datasets, create_data_loaders
from utils import *
import os
import time




# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=200,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, 
                    default=0.1, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='Batch Size')
parser.add_argument('-s', '--image_size', type=int, default=32,
                    help='image size')
parser.add_argument('-m', '--model', type=str, default= 'MobilenetV2',
                    help='Model Selection')
parser.add_argument('-d', '-sav_dir', type=str, dest='save_dir', help='directory', default='outputs')
args = vars(parser.parse_args())

# learning_parameters 
lr = args['learning_rate']
epochs = args['epochs']
BATCH_SIZE = args['batch_size']
s = args['image_size']
m = args['model']
d = args['save_dir']
h = True


# get the training, validation and test_datasets
train_dataset, valid_dataset, test_dataset = create_datasets(s)
# get the training and validaion data loaders
train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, BATCH_SIZE
)

save_dir = d + f"/{m}"

# Check the save_dir exists or not
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

Model = SelectModel(m)
# print(Model)

# build the model
model = Model.to(device)
print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

# loss function
criterion = nn.CrossEntropyLoss()

# if h:
#     model.half()
#     criterion.half()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=0.0001)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[100, 150])

if m in ['Resnet110']:
    for param in optimizer.param_groups:
        param['lr'] = lr*.1



# initialize SaveBestModel class
save_best_model = SaveBestModel()


# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                            optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                criterion)
    lr_scheduler.step()
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    save_model(m, epoch, model, optimizer, criterion)
    # save the best model till now if we have the least loss in the current epoch
    save_best_model(
        m, valid_epoch_acc, epoch, model, optimizer, criterion
    )
    print('-'*50)
    
# save the trained model weights for a final time
# save_model(m, epochs, model, optimizer, criterion)
save_data(m, train_acc, valid_acc, train_loss, valid_loss)
# save the loss and accuracy plots
save_plots(m, train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')