import os
import torch
import logging
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
from collections import defaultdict
from datasets import load_dataset
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import requests
from io import BytesIO
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from viz import (save_debug_image, plot_single_template_probabilities, 
                plot_confusion_matrix, plot_probability_histograms, 
                save_probability_stats)

class CountBenchEvaluator:
    def __init__(self, model_name="openai/clip-vit-base-patch32", confidence=0.5, margin=0.1, 
                 checkpoint_path=None, number_format="numeric", debug=False, samples_of_interest=None,
                 template_position='first', output_dir="results"):
        """Initialize the CountBench evaluator.
        
        Args:
            model_name (str): Name of the CLIP model to use
            confidence (float): Confidence threshold for predictions
            margin (float): Margin threshold for predictions
            checkpoint_path (str, optional): Path to finetuned model weights
            number_format (str): Format for numbers in templates ("numeric", "word", or "both")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.debug = debug
        self.output_dir = output_dir
        self.debug_dir = os.path.join(output_dir, "debug") if debug else None
        if debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            
        self.samples_of_interest = set(samples_of_interest) if samples_of_interest else set()
        self.template_position = template_position
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if checkpoint_path:
            logger.info(f"Loading finetuned weights from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("Successfully loaded finetuned weights")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                raise

        self.confidence_threshold = confidence
        self.margin_threshold = margin
        self.number_format = number_format
        
        # TODO: make max val a argument
        # countbench max number is 10, but we know VLMs struggle with double digits so we might want to expand this
        self.valid_numbers = set(range(1, 13))
        self.number_words = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 
            18: "eighteen", 19: "nineteen", 20: "twenty"
        }

        self.word_to_number = {word: num for num, word in self.number_words.items()}
        
    def format_number(self, number):
        """Format number according to specified format."""
        if self.number_format == "numeric":
            return [str(number)]
        elif self.number_format == "word":
            return [self.number_words[number]]
        else:  # "both"
            return [str(number), self.number_words[number]]

    def extract_number(self, template):
        """Extract first number from template checking both digits and words.
        Updated to consider the first digit in the sentence"""
        words = template.lower().split()
        
        # Store the index of first occurrence of any valid number
        first_index = float('inf')
        found_number = None
        
        for i, word in enumerate(words):
            # Check numeric form
            for num in self.valid_numbers:
                if str(num) == word and i < first_index:
                    first_index = i
                    found_number = num
                    break
                    
            # Check word form
            if word in self.word_to_number and i < first_index:
                first_index = i
                found_number = self.word_to_number[word]
        
        return found_number

    def arrange_templates(self, positive_templates, negative_templates):
        """Arrange templates according to the specified position strategy."""
        if self.template_position == "first":
            return positive_templates + negative_templates
        elif self.template_position == "random":
            all_templates = positive_templates + negative_templates
            indices = list(range(len(all_templates)))
            random.shuffle(indices)
            return [all_templates[i] for i in indices]
        else:
            raise ValueError(f"Invalid template position strategy: {self.template_position}")
            
    def find_number_in_text(self, text, target_number):
        """Find first occurrence of target_number in text (as word or digit)."""
        words = text.lower().split()
        
        # Store the index of first occurrence of the target number
        first_index = float('inf')
        found_word = None
        
        # Look for both numeric and word forms
        target_as_str = str(target_number)
        target_as_word = self.number_words[target_number].lower()
        
        for i, word in enumerate(words):
            if (word == target_as_str or word == target_as_word) and i < first_index:
                first_index = i
                found_word = word
                
        if found_word is not None:
            return found_word
                
        logger.warning(f"Could not find number {target_number} in text: {text}")
        return str(target_number)

    def generate_templates(self, text, number):
        """Generate positive and negative templates by modifying the original text."""
        positive_templates = []
        negative_templates = []
        
        original_number_text = self.find_number_in_text(text, number)
        
        # Find first occurrence index to only replace the first instance
        words = text.split()
        first_occurrence_idx = -1
        for i, word in enumerate(words):
            if word.lower() == original_number_text.lower():
                first_occurrence_idx = i
                break
                
        # goes two positions above and below
        nearby_numbers = [n for n in [number-2, number-1, number+1, number+2] 
                        if n in self.valid_numbers]
        
        # Generate positive templates
        number_formats = self.format_number(number)
        for num_format in number_formats:
            new_words = words.copy()
            new_words[first_occurrence_idx] = num_format
            positive_templates.append(' '.join(new_words))
            
        # Generate negative templates
        for n in nearby_numbers:
            number_formats = self.format_number(n)
            for num_format in number_formats:
                new_words = words.copy()
                new_words[first_occurrence_idx] = num_format
                negative_templates.append(' '.join(new_words))

        return positive_templates, negative_templates

    def should_debug_sample(self, index):
        """Determine if we should debug this sample."""
        return self.debug and (not self.samples_of_interest or index in self.samples_of_interest)

    def compute_argmax_prediction(self, probs, positive_templates, text, all_templates):
        """Get the predicted number from the argmax template."""
        pred_template = all_templates[torch.argmax(probs)]
        num = None
        num = self.extract_number(pred_template)

        if num is None:  
            logger.warning(f"Could not find number in predicted template: {pred_template}")
        return num

    def evaluate_single(self, image, text, number, index=None):
        """Evaluate a single image.
        Index = the sample to debug
        """

        debug_this_sample = self.should_debug_sample(index)

        if debug_this_sample:
            logger.debug(f"\nEvaluating sample {index}:")
            logger.debug(f"Text: {text}")
            logger.debug(f"Number: {number}")
            logger.debug(f"Template position strategy: {self.template_position}")

            if index is not None:
                debug_image_path = os.path.join(self.debug_dir, f"sample_{index}_image.png")
                save_debug_image(image, debug_image_path, f"Sample {index}\nNumber: {number}\n{text}")
                logger.debug(f"Saved debug image to {debug_image_path}")

        try:
            number = int(number)
            if number not in self.valid_numbers:
                return {
                    'correct': False,
                    'confidence': 0.0,
                    'pred_template': "Invalid number",
                    'all_probs': np.array([]),
                    'all_templates': []
                }
        except ValueError:
            logger.warning(f"Invalid number format: {number}")
            return {
                'correct': False,
                'confidence': 0.0,
                'pred_template': "Invalid input",
                'all_probs': np.array([]),
                'all_templates': []
            }

        positive_templates, negative_templates = self.generate_templates(text, number)
        all_templates = self.arrange_templates(positive_templates, negative_templates)

        positive_indices = [i for i, template in enumerate(all_templates) if template in positive_templates]

        if debug_this_sample:
            logger.debug("\nTemplates arranged:")
            for i, t in enumerate(all_templates):
                template_type = "Positive" if i in positive_indices else "Negative"
                logger.debug(f"  {i}: [{template_type}] {t}")

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

        if debug_this_sample:
            if index is not None:
                debug_prob_path = os.path.join(self.debug_dir, f"sample_{index}_probs.png")
                plot_single_template_probabilities(
                    all_templates, 
                    probs.cpu().numpy(), 
                    debug_prob_path,
                    self.extract_number,
                    positive_indices
                )
                logger.debug(f"Saved probability plot to {debug_prob_path}")

        pos_probs = probs[positive_indices]
        neg_probs = torch.tensor([p for i, p in enumerate(probs) if i not in positive_indices])
        
        best_pos_idx = torch.argmax(pos_probs)
        best_pos_prob = pos_probs[best_pos_idx]
        
        best_neg_prob = torch.max(neg_probs) if len(neg_probs) > 0 else torch.tensor(0.0)
        
        is_correct = (
            best_pos_prob > self.confidence_threshold and
            best_pos_prob > best_neg_prob + self.margin_threshold and
            best_pos_prob == torch.max(probs)
        )

        pred_number = self.compute_argmax_prediction(probs, positive_templates, text, all_templates)
        
        return {
            'correct': is_correct,
            'confidence': float(best_pos_prob),
            'pred_template': all_templates[torch.argmax(probs)],
            'pred_number': pred_number,
            'all_probs': probs.cpu().numpy(),
            'all_templates': all_templates
        }

    def plot_confusion_matrix(self, true_numbers, pred_numbers, output_dir):
        plot_confusion_matrix(
            true_numbers, 
            pred_numbers, 
            self.valid_numbers,
            os.path.join(output_dir, 'confusion_matrix.png')
        )

    def download_image(self, url):
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            return None

    def evaluate_dataset(self, dataset):
        """Evaluate entire dataset."""
        results = defaultdict(list)
        
        none_type_count = 0
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            image = item['image']

            if image is None:
                logger.warning(f"Attempting to download image from {item['image_url']}")
                # currently disabling download bc on pace we can't reach internet from GPU
                # image = self.download_image(item['image_url'])
                none_type_count += 1
                continue

            eval_result = self.evaluate_single(
                image, 
                item['text'],
                item['number'],
                index=idx if self.debug else None
            )
            
            results['correct'].append(eval_result['correct'])
            results['confidence'].append(eval_result['confidence'])
            results['pred_templates'].append(eval_result['pred_template'])
            results['groundtruth'].append(item['number'])
            results['pred_numbers'].append(eval_result['pred_number'])
            results['all_probs'].append(eval_result['all_probs'])
            results['all_templates'].append(eval_result['all_templates'])
        
        logger.info(f"Nonetype count: {none_type_count}")
        return results

    def compute_metrics(self, results):
        """Compute evaluation metrics."""
        correct = sum(results['correct'])
        total = len(results['correct'])
        
        if total == 0:
            logger.warning("No samples found in results. Returning zero accuracy.")
            return {
                'accuracy': 0.0,
                'total_samples': 0,
                'correct': 0,
                'avg_confidence': 0.0
            }
        
        accuracy = correct / total

        # vanilla argmax accuracy
        true_numbers = results['groundtruth']
        pred_numbers = results['pred_numbers']
        valid_predictions = [(true, pred) for true, pred in zip(true_numbers, pred_numbers) 
                            if pred is not None]
        
        if valid_predictions:
            true_vals, pred_vals = zip(*valid_predictions)
            argmax_correct = sum(1 for t, p in zip(true_vals, pred_vals) if t == p)
            argmax_accuracy = argmax_correct / len(valid_predictions)
        else:
            argmax_accuracy = 0.0

        
        confidences = [conf if isinstance(conf, float) else conf.item() 
                      for conf in results['confidence']]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # score high confidnece accuracy
        high_conf_mask = np.array(confidences) > self.confidence_threshold
        
        correct_array = np.array([x.cpu() if torch.is_tensor(x) else x for x in results['correct']])
        high_conf_correct = sum(correct_array[high_conf_mask])
        high_conf_total = sum(high_conf_mask)
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0
        
        number_accuracies = {}
        for num in self.valid_numbers:
            num_mask = np.array(results['groundtruth']) == num
            if sum(num_mask) > 0:
                num_accuracy = sum(correct_array[num_mask]) / sum(num_mask)
                number_accuracies[int(num)] = float(num_accuracy)
        
        return {
            'accuracy': accuracy,
            'argmax_accuracy': argmax_accuracy,
            'total_samples': total,
            'correct': correct,
            'avg_confidence': avg_confidence,
            'high_confidence_accuracy': high_conf_accuracy,
            'per_number_accuracy': number_accuracies
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model on CountBench')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                      help='CLIP model version to use')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Confidence threshold for predictions')
    parser.add_argument('--margin', type=float, default=0.1,
                      help='Margin threshold for predictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to finetuned model checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--number_format', type=str, choices=['numeric', 'word', 'both'],
                      default='numeric', help='Format for numbers in templates')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode: saves images, prints template probabilities, and saves probability plots')
    parser.add_argument('--samples', type=int, nargs='+',
                      help='Indices of samples to debug (e.g., --samples 0 10 42)')
    parser.add_argument('--template_position', type=str, choices=['first', 'random'],
                      default='first', help='Position strategy for positive templates')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.debug:
        os.makedirs(os.path.join(args.output_dir, 'debug'), exist_ok=True)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    dataset = load_dataset("nielsr/countbench")
    logger.info(f"Loaded dataset with {len(dataset['train'])} samples")

    evaluator = CountBenchEvaluator(
        model_name=args.model,
        confidence=args.confidence,
        margin=args.margin,
        checkpoint_path=args.checkpoint,
        number_format=args.number_format,
        debug=args.debug,
        samples_of_interest=args.samples,
        template_position=args.template_position,
        output_dir=args.output_dir
    )

    results = evaluator.evaluate_dataset(dataset['train'])
    metrics = evaluator.compute_metrics(results)

    logger.info(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Argmax Accuracy: {metrics['argmax_accuracy']:.2%}")
    logger.info(f"Total correct: {metrics['correct']}/{metrics['total_samples']}")
    logger.info(f"Average confidence: {metrics['avg_confidence']:.2%}")
    logger.info(f"High-confidence accuracy: {metrics['high_confidence_accuracy']:.2%}")

    evaluator.plot_confusion_matrix(
        results['groundtruth'],
        results['pred_numbers'],
        args.output_dir
    )
    
    output_file = os.path.join(args.output_dir, 'countbench_results.npy')
    np.save(output_file, {
        'metrics': metrics,
        'predictions': results['pred_templates'],
        'groundtruth': results['groundtruth'],
        'confidence': results['confidence'],
        'all_probs': results['all_probs'],
        'all_templates': results['all_templates']
    })
    logger.info(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()