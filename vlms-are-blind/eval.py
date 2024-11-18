import os
import sys
import types
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
import logging
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPEvaluator:
    def __init__(self, model_name="openai/clip-vit-base-patch32", confidence=0.5, margin=0.1, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        try:
            # Load checkpoint with specific map_location
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load only the model state dict
            if 'model_state_dict' in checkpoint:
                missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing:
                    logger.warning(f"Missing keys in checkpoint: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
                logger.info("Successfully loaded finetuned weights")
            else:
                logger.error("Checkpoint does not contain model_state_dict")
                raise KeyError("Invalid checkpoint format")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.confidence_threshold = confidence
        self.margin_threshold = margin

        self.valid_values = {
            'Line Plot Intersections': {0, 1, 2},
            'Olympic Counting - Circles': {5, 6, 7, 8, 9},
            'Olympic Counting - Pentagons': {5, 6, 7, 8, 9},
            'Nested Squares': {2, 3, 4, 5},
            'Subway Connections': {0, 1, 2, 3},
            'Circled Letter': set('AaBbCcDdEeGgHhIiKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz')
        }
        
        # from the benchmark website
        self.circled_letter_words = {
            'Acknowledgement',
            'Subdermatoglyphic', 
            'tHyUiKaRbNqWeOpXcZvM'
        }

    def validate_groundtruth(self, task, groundtruth):
        """Validate groundtruth values for each task"""
        try:
            if task == 'Circled Letter':
                # Check if letter is valid
                if groundtruth.lower() not in {c.lower() for c in self.valid_values['Circled Letter']}:
                    logger.warning(f"Invalid letter in groundtruth: {groundtruth}")
                    return False
                return True
            elif task in self.valid_values:
                gt_val = int(groundtruth)
                if gt_val not in self.valid_values[task]:
                    logger.warning(f"Invalid groundtruth {gt_val} for task {task}")
                    return False
            elif task == 'Touching Circles':
                if groundtruth.lower() not in {'yes', 'no'}:
                    logger.warning(f"Invalid groundtruth {groundtruth} for Touching Circles")
                    return False
            elif task.startswith('Counting Grid'):
                try:
                    if ',' in groundtruth:
                        rows, cols = map(int, groundtruth.split(','))
                    else:
                        rows, cols = map(int, groundtruth.split('x'))
                    if not (3 <= rows <= 10 and 3 <= cols <= 10):
                        logger.warning(f"Invalid grid dimensions {rows}x{cols}")
                        return False
                except ValueError:
                    return False
            return True
        except (ValueError, TypeError):
            logger.warning(f"Invalid groundtruth format for task {task}: {groundtruth}")
            return False

    def get_task_templates(self, task, groundtruth):
        """Get task-specific templates based on groundtruth"""
        
        if task == 'Touching Circles':
            is_touching = groundtruth.lower() == "yes"
            state = "touching or overlapping" if is_touching else "separated"
            return [
                f"Two circles that are {state}",
                f"A pair of circles that are {state}",
                f"Two circles {state} from each other",
                f"Two circles in {state} configuration"
            ]

        elif task == 'Circled Letter':
            return [
                f"The letter {groundtruth} is circled in red",
                f"A red circle highlights the letter {groundtruth}",
                f"The character {groundtruth} is marked with a red oval",
                f"Letter {groundtruth} is emphasized with a red circle"
            ]
            
        elif task == 'Line Plot Intersections':
            return [
                f"Two lines intersecting {groundtruth} times",
                f"A graph with {groundtruth} intersection points",
                f"Two line segments with {groundtruth} crossing points",
                f"Two piecewise linear functions with {groundtruth} intersections"
            ]
            
        elif task == 'Subway Connections':
            return [
                f"{groundtruth} different paths between stations A and B",
                f"{groundtruth} unique routes connecting stations A and B",
                f"A subway map showing {groundtruth} paths between A and B",
                f"A transit map with {groundtruth} distinct routes between stations"
            ]
            
        elif task == 'Nested Squares':
            return [
                f"A pattern of {groundtruth} nested squares",
                f"{groundtruth} concentric squares",
                f"{groundtruth} squares inside each other",
                f"A diagram showing {groundtruth} squares nested within each other"
            ]
            
        elif task.startswith('Olympic Counting'):
            shape = "circles" if "Circles" in task else "pentagons"
            return [
                f"An image with {groundtruth} overlapping {shape}",
                f"A logo-like pattern with {groundtruth} {shape}",
                f"{groundtruth} {shape} arranged in an Olympic-like pattern",
                f"A design containing {groundtruth} {shape} in overlapping rows"
            ]
            
        elif task.startswith('Counting Grid'):
            try:
                if "," in groundtruth:
                    rows, cols = map(int, groundtruth.split(','))
                else:
                    rows, cols = map(int, groundtruth.split('x'))
                grid_type = "empty" if "Blank" in task else "filled with text"
                return [
                    f"A {grid_type} grid with {rows} rows and {cols} columns",
                    f"A {grid_type} table layout of {rows} by {cols}",
                    f"A {grid_type} grid of size {rows} rows × {cols} columns",
                    f"A {rows}×{cols} {grid_type} table"
                ]
            except ValueError:
                logger.warning(f"Invalid grid format in groundtruth: {groundtruth}")
                return [f"A grid with {groundtruth}"]
            
        else:
            logger.warning(f"Unknown task: {task}")
            return [f"An image showing {groundtruth}"]

    def generate_negative_templates(self, task, groundtruth):
        """Generate negative templates based on task type"""
        if not self.validate_groundtruth(task, groundtruth):
            return ["Invalid input"]

        if task == 'Touching Circles':
            is_touching = groundtruth.lower() == "yes"
            state = "separated" if is_touching else "touching or overlapping"
            return [f"Two circles that are {state}"]
        
        elif task == 'Circled Letter':
            gt_letter = groundtruth.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            idx = alphabet.find(gt_letter)
            if idx != -1:
                nearby_letters = []
                for offset in [-2, -1, 1, 2]:
                    new_idx = (idx + offset) % len(alphabet)
                    letter = alphabet[new_idx]
                    if letter.lower() in {c.lower() for c in self.valid_values['Circled Letter']}:
                        nearby_letters.append(letter)
                
                return [
                    f"The letter {letter} is circled in red" for letter in nearby_letters[:4]
                ] + [
                    "No letter is circled",
                    "Multiple letters are circled"
                ]
            return ["A different letter is circled"]

        elif task.startswith('Olympic Counting') or task == 'Line Plot Intersections' or task == 'Subway Connections':
            gt_num = int(groundtruth)
            valid_range = self.valid_values[task]
            nearby_nums = [n for n in valid_range if n != gt_num][:4]
            shape = "circles" if "Circles" in task else "pentagons" if "Pentagons" in task else "intersections"
            return [f"An image showing {num} {shape}" for num in nearby_nums]
            
        elif task == 'Nested Squares':
            gt_num = int(groundtruth)
            valid_range = self.valid_values[task]
            other_nums = [n for n in valid_range if n != gt_num]
            return [
                f"{num} nested squares" for num in other_nums
            ] + ["Overlapping squares", "Adjacent squares"]
            
        elif task.startswith('Counting Grid'):
            try:
                if "," in groundtruth:
                    rows, cols = map(int, groundtruth.split(','))
                else:
                    rows, cols = map(int, groundtruth.split('x'))
                nearby_pairs = [
                    (rows+1, cols),
                    (rows-1, cols),
                    (rows, cols+1),
                    (rows, cols-1)
                ]
                grid_type = "empty" if "Blank" in task else "text-filled"
                return [
                    f"A {grid_type} grid of size {r}×{c}" for r, c in nearby_pairs 
                    if 3 <= r <= 9 and 3 <= c <= 9
                ] + [f"A {grid_type} grid with random dimensions"]
            except ValueError:
                return ["A grid with different dimensions"]
                
        return ["Something else entirely", "An unrelated image"]
    
    def evaluate_single(self, image, task, groundtruth):
        """Evaluate a single image"""
        if not self.validate_groundtruth(task, groundtruth):
            return {
                'correct': False,
                'confidence': 0.0,
                'pred_template': "Invalid input",
                'all_probs': np.array([]),
                'all_templates': []
            }

        positive_templates = self.get_task_templates(task, groundtruth)
        negative_templates = self.generate_negative_templates(task, groundtruth)
        all_templates = positive_templates + negative_templates

        inputs = self.processor(
            images=[image],
            text=all_templates,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]

        pos_probs = probs[:len(positive_templates)]
        neg_probs = probs[len(positive_templates):]
        
        best_pos_idx = torch.argmax(pos_probs)
        best_pos_prob = pos_probs[best_pos_idx]
        
        best_neg_prob = torch.max(neg_probs) if len(neg_probs) > 0 else torch.tensor(0.0)
        
        is_correct = (best_pos_prob > self.confidence_threshold and  # Confidence threshold
                     best_pos_prob > best_neg_prob + self.margin_threshold and  # Margin requirement
                     best_pos_prob == torch.max(probs))  # Highest overall

        
        return {
            'correct': is_correct,
            'confidence': float(best_pos_prob),
            'pred_template': all_templates[torch.argmax(probs)],
            'all_probs': probs.cpu().numpy(),
            'all_templates': all_templates
        }

    def evaluate_dataset(self, dataset, task):
        """Evaluate entire dataset for a specific task"""
        # TODO: slow can we cache it? 
        task_dataset = dataset.filter(lambda x: x['task'] == task)
        logger.info(f"Found {len(task_dataset)} samples for task {task}")
        
        results = defaultdict(list)
        
        for item in tqdm(task_dataset, desc=f"Evaluating {task}"):
            image = item['image']
            groundtruth = item['groundtruth']
            
            eval_result = self.evaluate_single(image, task, groundtruth)
            
            results['correct'].append(eval_result['correct'])
            results['confidence'].append(eval_result['confidence'])
            results['pred_templates'].append(eval_result['pred_template'])
            results['groundtruth'].append(groundtruth)
            results['all_probs'].append(eval_result['all_probs'])
            results['all_templates'].append(eval_result['all_templates'])
        
        return results

    def compute_metrics(self, results):
        """Compute evaluation metrics"""
        correct = sum(results['correct'])
        total = len(results['correct'])
        # avoid div by zero
        if total == 0:
            logger.warning("No samples found in results. Returning zero accuracy.")
            return {
                'accuracy': 0.0,
                'total_samples': 0,
                'correct': 0,
                'avg_confidence': 0.0
            }
        
        accuracy = correct / total

        confidences = [conf.cpu() if torch.is_tensor(conf) else conf for conf in results['confidence']]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # TODO: add support for pure argmax or confidence based on benchmark random guess value
        high_conf_mask = np.array(confidences) > 0.5
        correct_array = np.array([x.cpu() if torch.is_tensor(x) else x for x in results['correct']])
        high_conf_correct = sum(correct_array[high_conf_mask])
        high_conf_total = sum(high_conf_mask)
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct': correct,
            'avg_confidence': avg_confidence,
            'high_confidence_accuracy': high_conf_accuracy
        }


def main():
    dataset = load_dataset("XAI/vlmsareblind")

    # TODO: move to argparse function
    parser = argparse.ArgumentParser(description='Evaluate CLIP model on visual reasoning tasks')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model version to use (default: openai/clip-vit-base-patch32)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='If something is more than confidence score confident then we count it as valid')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='must be at least margin more confident then the next negative sample')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to finetuned model checkpoint (optional)')
    args = parser.parse_args()

    output_dir = args.model.replace('/', '_')
    if args.checkpoint:
        output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(args.checkpoint)))
    os.makedirs(output_dir, exist_ok=True)

    evaluator = CLIPEvaluator(
        model_name=args.model,
        confidence=args.confidence,
        margin=args.margin,
        checkpoint_path=args.checkpoint
    )
    
    tasks = [
        'Touching Circles',
        'Line Plot Intersections',
        'Circled Letter',
        'Subway Connections',
        'Nested Squares',
        'Olympic Counting - Circles',
        'Counting Grid - Blank Grids',
        'Counting Grid - Word Grids',
        'Olympic Counting - Pentagons'
    ]
    
    all_results = {}
    
    for task in tasks:
        logger.info(f"\nEvaluating task: {task}")
        results = evaluator.evaluate_dataset(dataset['valid'], task)
        metrics = evaluator.compute_metrics(results)
        
        all_results[task] = {
            'metrics': metrics,
            'results': results
        }
        
        logger.info(f"Task: {task}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Correct: {metrics['correct']}/{metrics['total_samples']}")
        logger.info(f"Average confidence: {metrics['avg_confidence']:.2%}")

        np.save(f"{output_dir}/results_{task.lower().replace(' ', '_').replace('-', '_')}.npy", {
            'metrics': metrics,
            'predictions': results['pred_templates'],
            'groundtruth': results['groundtruth'],
            'confidence': results['confidence'],
            'all_probs': results['all_probs'],
            'all_templates': results['all_templates']
        })

if __name__ == "__main__":
    main()