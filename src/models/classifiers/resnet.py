from torchvision.models import resnet34

from .registry import CLASSIFIERS

CLASSIFIERS.register_module("resnet34")(resnet34) # h