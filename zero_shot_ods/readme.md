# Zero shot transfer of CLIP to object detection using the following pipeline:

1. Extracting CLIP features
2. Develop regional proposals using heuristic and classical techniques like selective search: This allows transfer with minimal overhead, as opposed to training a Regional Proposal Network
3. Regional Feature Pooling using something like ROIAlign
4. Region Classification: Use CLIP's text-image similarity head
5. Bounding Box Refinement using regression
6. Final output processing using non-maximum supression to remove the overlapping boxes