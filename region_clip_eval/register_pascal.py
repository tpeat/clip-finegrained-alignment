from detectron2.data.datasets import register_pascal_voc
from detectron2.data import MetadataCatalog


# register_pascal_voc("voc_2012_trainval", "/storage/ice1/9/3/kkundurthy3/dataset/pascal/VOCdevkit/VOC2012", "trainval", 2012)
# register_pascal_voc("voc_2012_test", "/storage/ice1/9/3/kkundurthy3/dataset/pascal/VOCdevkit/VOC2012", "test", 2012)

print(MetadataCatalog.get("voc_2012_trainval"))