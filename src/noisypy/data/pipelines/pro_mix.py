from copy import deepcopy

import torch
from torchvision import transforms

from .base import AugmentationPipeline


class ProMixify(AugmentationPipeline):
    """
        ProMix implementation based on https://github.dev/Justherozen/ProMix/blob/main/Train_promix.py
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, dataset_cls):

        class ProMixDataset(dataset_cls):
            def __init__(self, mode, *args, **kwargs):
                self.mode = 'init'
                super().__init__(*args, **kwargs)
                self.mode = mode
                self.probabilities = torch.ones(self.data.shape[0])
                self.probabilities2 = torch.ones(self.data.shape[0])
                self.transform_s = deepcopy(self.transform)
                # must be transformed to PIL image
                self.transform_s.transforms.insert(0, transforms.ToPILImage())
                # Use RandAugment with N=3, M=5
                self.transform_s.transforms.insert(1, transforms.RandAugment(3, 5))

                assert hasattr(self, 'data'), "Dataset must have data attribute"


            def __getitem__(self, index):
                if self.mode == 'all':
                    sample = super().__getitem__(index)
                    return *sample, index
                elif self.mode == 'all_lab':
                    sample = super().__getitem__(index)
                    x2 = self.data[index]
                    img2 = self.transform_s(x2)
                    prob = self.probabilities[index]
                    prob2 = self.probabilities2[index]   
                    return sample[0], img2, sample[-1], prob, prob2
                elif self.mode == 'init':
                    # when getitem is called before mode is set return the parent getitem
                    return super().__getitem__(index)
                else:
                    raise ValueError("Invalid mode")
                
        return ProMixDataset