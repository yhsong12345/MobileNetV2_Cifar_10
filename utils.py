import torch
import matplotlib.pyplot as plt
import os
import pandas as pd



            

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_acc=float(0)
    ):
        self.best_valid_acc = best_valid_acc
        
    def __call__(
        self, m, current_valid_acc,
        epoch, model, optimizer, criterion
    ):
        path = f'./outputs/{m}'
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation acc: {self.best_valid_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{path}/best_model.pt')


def save_model(m, epoch, model, optimizer, criterion):
    print(f'Saving final model...')
    path = f'./outputs/{m}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, f'{path}/model{epoch}.pt')



def save_data(m, train_acc, valid_acc, train_loss, valid_loss):
    print("Saving losses and accuracies")
    path = f'./outputs/{m}'
    data = {'train_accuracy': train_acc, 'valid_accuracy':valid_acc,
            'train_loss': train_loss, 'valid_loss': valid_loss}
    df = pd.DataFrame(data=data)
    df.to_excel(f'{path}/{m}result.xlsx')

    


def save_plots(m,train_acc, valid_acc, train_loss, valid_loss):
    path = f'./outputs/{m}'
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png')


    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='green', linestyle='--',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='--',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')


