import matplotlib.pyplot as plt
import os

def plot_loss_function(train_loss, test_loss, task='regression'):
    plt.plot(train_loss, label='Train_Loss')         
    plt.plot(test_loss, label='Test Loss')
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training metrics')
    plt.legend()
    plot_path = os.path.join("plots", f"{task}_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()