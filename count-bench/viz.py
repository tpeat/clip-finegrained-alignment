# @author: Claude

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

def save_debug_image(image, output_path, text=None):
    """Save image with optional text overlay for debugging."""
    plt.figure(figsize=(10, 10))
    if isinstance(image, Image.Image):
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        plt.imshow(image_array)
    else:
        plt.imshow(image)
    
    if text:
        plt.title(text, wrap=True)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_single_template_probabilities(templates, probabilities, output_path, extract_number, positive_indices):
    """Plot probabilities for each template in a single sample."""
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(range(len(probabilities)), probabilities)
    
    # Color bars based on positive/negative indices
    for i, bar in enumerate(bars):
        bar.set_color('green' if i in positive_indices else 'red')
    
    for i, (prob, template) in enumerate(zip(probabilities, templates)):
        number = extract_number(template)
        plt.text(i, prob, f'{number}', ha='center', va='bottom')
        plt.text(i, prob/2, f'{prob:.3f}', ha='center', va='center', rotation=90)
    
    plt.title('Template Probabilities by Count')
    plt.xlabel('Template Index')
    plt.ylabel('Probability')
    plt.ylim(0, max(probabilities) * 1.1)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend(['Templates'], ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(true_numbers, pred_numbers, valid_numbers, output_path):
    """Create and save confusion matrix plot."""
    # Remove None predictions
    valid_indices = [i for i, x in enumerate(pred_numbers) if x is not None]
    true_numbers = [true_numbers[i] for i in valid_indices]
    pred_numbers = [pred_numbers[i] for i in valid_indices]

    labels = sorted(list(valid_numbers))
    cm = confusion_matrix(true_numbers, pred_numbers, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Number')
    plt.ylabel('True Number')
    
    plt.savefig(output_path)
    plt.close()

def plot_probability_histograms(results, output_dir, confidence_threshold):
    """Plot histograms of probability distributions."""
    all_pos_probs = []
    all_neg_probs = []
    
    # Get positive/negative probabilities using positive_indices
    for probs, templates, correct in zip(results['all_probs'], results['all_templates'], results['correct']):
        # Get indices of positive templates from the results
        positive_indices = [i for i, t in enumerate(templates) if any(t == pos_t for pos_t in templates[:len(templates)//3])]
        
        # Split probabilities
        pos_probs = probs[positive_indices]
        neg_probs = probs[[i for i in range(len(probs)) if i not in positive_indices]]
        
        all_pos_probs.extend(pos_probs)
        all_neg_probs.extend(neg_probs)

    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.hist(all_pos_probs, bins=50, alpha=0.7, color='green', 
             label=f'Positive Templates (n={len(all_pos_probs)})')
    plt.axvline(x=confidence_threshold, color='r', linestyle='--', 
                label=f'Confidence Threshold ({confidence_threshold})')
    plt.title('Probability Distribution - Positive Templates')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(all_neg_probs, bins=50, alpha=0.7, color='red',
             label=f'Negative Templates (n={len(all_neg_probs)})')
    plt.axvline(x=confidence_threshold, color='r', linestyle='--')
    plt.title('Probability Distribution - Negative Templates')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'))
    plt.close()

def save_probability_stats(stats, output_dir):
    """Save probability distribution statistics to file."""
    with open(os.path.join(output_dir, 'probability_stats.txt'), 'w') as f:
        f.write("Probability Distribution Statistics\n")
        f.write("=================================\n\n")
        
        for template_type, metrics in stats.items():
            f.write(f"{template_type}:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.3f}\n")
            f.write("\n")