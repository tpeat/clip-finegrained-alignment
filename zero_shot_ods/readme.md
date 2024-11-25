# Zero shot transfer of CLIP to object detection 

### General Pipeline

1. Extracting CLIP features
2. Develop regional proposals using heuristic and classical techniques like selective search: This allows transfer with minimal overhead, as opposed to training a Regional Proposal Network
3. Regional Feature Pooling using something like ROIAlign
4. Region Classification: Use CLIP's text-image similarity head
5. Bounding Box Refinement using regression
6. Final output processing using non-maximum supression to remove the overlapping boxes


test_loader.py: randomly sample 5 images to be used for testing and debugging
ods_eval.py: main file implementing this pipeline