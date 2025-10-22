import matplotlib.pyplot as plt
import os
import json

def plot_loss_function(train_loss, test_loss, experiment_name='regression') -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train_Loss')         
    plt.plot(test_loss, label='Test Loss')
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training metrics')
    plt.legend()
    
    # saving
    os.makedirs(f'logs/{experiment_name}', exist_ok=True)
    plot_path = os.path.join(f"logs/{experiment_name}/", "loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def model_comparison() -> None:
    names = []
    values = []

    for root, dirs, files in os.walk('logs/'):
        if 'results.json' in files:
            if os.path.basename(root) == 'best_trial.json':
                continue
                
            path = os.path.join(root, 'results.json')
            with open(path, 'r', encoding='utf-8') as file:
                history = json.load(file)

            rel_path = os.path.relpath(root, 'logs')
            if rel_path == ".":
                name = os.path.basename(root)
            else:
                name = rel_path.replace(os.sep, '_')

            names.append(name)
            values.append(history['r2'][-1])


    sorted_data = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    sorted_names = [item[0] for item in sorted_data]
    sorted_values = [item[1] for item in sorted_data]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_names[::-1], sorted_values[::-1])
    plt.xlabel('R^2')
    plt.ylabel('Experiment name')
    plt.title('Model Comparison')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5, axis='x')
    plt.savefig('plots/ModelComparison.png')
    plt.show()
    plt.close()