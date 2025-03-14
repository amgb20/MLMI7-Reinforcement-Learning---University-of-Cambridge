import os
import matplotlib.pyplot as plt

def save_fig(fig, question: str, world_name: str, task: str):
    if os.path.exists(f'results/images/{question}/{world_name}_{task}.png'):
        os.remove(f'results/images/{question}/{world_name}_{task}.png')
    os.makedirs(f'results/images/{question}', exist_ok=True)
    plt.savefig(f'results/images/{question}/{world_name}_{task}.png')