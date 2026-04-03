import matplotlib.pyplot as plt
import os

def plot_imbalance(df, save_path='figures/class_imbalance.png'):
    # Make sure the figures directory actually exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    counts = df['fraudulent'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Genuine (0)', 'Fraud (1)'], counts.values, color=['green', 'red'])
    ax.set_title('Class Distribution: Severe Imbalance', fontsize=14)
    ax.set_ylabel('Count', fontsize=12)
    
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, f'{v:,}', ha='center', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved class imbalance plot to {save_path}")