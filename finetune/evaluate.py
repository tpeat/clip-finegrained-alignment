import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_batch(model, batch, device, filename=None):
    images = batch['image'].to(device)
    gt_counts = batch['gt_count']
    cf_counts = batch['cf_counts']
    cap = batch['text'].to(device)
    all_cf_caps = batch['cf_text'].to(device)
    
    possible_counts = list(range(1, 11))
    confusion = np.zeros((len(possible_counts), len(possible_counts)))
    count_occurrences = {count: 0 for count in possible_counts}
    results = []

    img_embeddings = model.encode_image(images)
    
    for idx, (img_embedding, true_count, caption, cf_caps, cf_count) in enumerate(zip(img_embeddings, gt_counts, cap, all_cf_caps, cf_counts)):
        scores = {}
        similarities = [0] * len(possible_counts)
        
        text_embedding = model.encode_text(caption.unsqueeze(0))
        similarity = torch.cosine_similarity(img_embedding.unsqueeze(0), text_embedding)
        scores[int(true_count)] = similarity.item()
        similarities[int(true_count) - 1] = similarity.item()

        for count, test_caption in zip(cf_count, cf_caps):
            text_embedding = model.encode_text(test_caption.unsqueeze(0))
            
            similarity = torch.cosine_similarity(img_embedding.unsqueeze(0), text_embedding)
            scores[count] = similarity.item()
            similarities[count - 1] = similarity.item()

        probs = torch.nn.functional.softmax(torch.tensor(similarities), dim=0).numpy()

        # Add to confusion matrix
        true_idx = possible_counts.index(int(true_count))
        confusion[true_idx] += probs
        count_occurrences[int(true_count)] += 1

        # Get predicted count and record results
        pred_count = max(scores.items(), key=lambda x: x[1])[0]
        results.append({
            'true_count': int(true_count),
            'pred_count': pred_count,
            'correct': pred_count == int(true_count),
            'scores': scores
        })

    for count in possible_counts:
        if count_occurrences[count] > 0:
            row_idx = possible_counts.index(count)
            confusion[row_idx] /= count_occurrences[count]

    if filename:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='.2f', 
                    xticklabels=possible_counts, yticklabels=possible_counts,
                    cmap='Blues')
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title('Counting Confusion Matrix (Probabilities)')
        plt.savefig(filename)
        plt.close()
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    return accuracy, confusion, results