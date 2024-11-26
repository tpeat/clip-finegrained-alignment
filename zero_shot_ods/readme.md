# Zero shot transfer of CLIP to object detection 

### General Pipeline

1. Extracting CLIP features from baseline and pretrained models. Experiments will be conducted on 
2. Develop regional proposals using heuristic and classical techniques like selective search: This allows transfer with minimal overhead, as opposed to training a Regional Proposal Network. Moreover, for selective search, decreasing the number of iterations decreases the amount of merging into larger candidate regions, potentially leaving smaller regions and therefore a greater likelihood of encapsulating the small object in a given proposal. 
3. Regional Feature Pooling using something like ROIAlign
4. Region Classification: Use CLIP's text-image similarity head
5. Bounding Box Refinement using regression
6. Final output processing using non-maximum supression to remove the overlapping boxes


- test_loader.py: randomly sample 5 images to be used for testing and debugging
- ods_eval.py: main file implementing this pipeline

#### Note: ChatGPT was used to help with implementation level details, such as referring to appropriate syntax/library methods in PyTorch and their usage. However, the hypothesis and ideas, the design of the zero shot object detection pipeline, the code layout, and the gathering of exerimental results was conducted by us.