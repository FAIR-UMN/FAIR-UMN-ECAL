# This script includes additional helper functions we need
import torch
import matplotlib.pyplot as plt

def save_model(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)
    print('The trained model has been saved!')

def plot_loss(loss, fig_name):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()

def show_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training loss vs. Epoch')
    plt.tight_layout()
    # plt.show()
    plt.close()

if __name__ == '__main__':
    pass