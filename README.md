# CLIP Finegrained Alignment

CLIP's contrastive loss has well-documented limitations such as poor performance on finegrained comprehension. When prompted to classify the existence of small objects, CLIP performs near random. I hypothesize that training CLIP directly for a counting objective will result in improvements in zero-shot small object detection. 

This project explores the interplay between related, but distinctly unique objectives: counting and detection. They both are rooted in detailed visual and text understanding. As such, I've included evaluation methods against the "Vision Language Models are Blind" benchmark, a set of basic tasks that any five-year old could solve; yet suprisingly VLMs struggle.

So far, the codebase supports two improvments upon CLIP:
1) Sparse Fine-grained Contrastive Alignment (SPARC) loss
2) Adam Selective Projection Decay


## Setup

Pytorch, datasets, transformers; typical DL packages will be required

This project assumes that COCO is already downloaded