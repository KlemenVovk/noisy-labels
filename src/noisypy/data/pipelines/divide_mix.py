import torch

from .base import AugmentationPipeline


class DivideMixify(AugmentationPipeline):
    """
        DivideMix implementation based on https://openreview.net/pdf?id=HJgExaVtwr
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, dataset_cls):

        class DivideMixCIFAR10Dataset(dataset_cls):
            def __init__(self, mode, *args, **kwargs):
                self.mode = 'init'
                super().__init__(*args, **kwargs)
                self.mode = mode
                self.labeled_indices = torch.arange(self.data.shape[0])
                self.unlabeled_indices = torch.arange(self.data.shape[0])
                self.prediction = torch.ones(self.data.shape[0])
                self.probabilities = torch.ones(self.data.shape[0])

            def __len__(self):
                if self.mode == 'all':
                    return super().__len__()
                elif self.mode == 'labeled':
                    return len(self.labeled_indices)
                elif self.mode == 'unlabeled':
                    return len(self.unlabeled_indices)
                elif self.mode == 'init':
                    # when len is called before mode is set return the parent len
                    return super().__len__()
                else:
                    raise ValueError("Invalid mode")


            def __getitem__(self, index):
                if self.mode == 'all':
                    sample = super().__getitem__(index)
                    return *sample, index
                elif self.mode == 'labeled':
                    indices = self.labeled_indices[index]
                    prob = self.probabilities[indices]
                    sample1 = super().__getitem__(indices)
                    sample2 = super().__getitem__(indices)
                    return sample1[0], sample2[0], sample1[-1], prob
                elif self.mode == 'unlabeled':
                    indices = self.unlabeled_indices[index]
                    sample1 = super().__getitem__(indices)
                    sample2 = super().__getitem__(indices)
                    return sample1[0], sample2[0]
                elif self.mode == 'init':
                    # when getitem is called before mode is set return the parent getitem
                    return super().__getitem__(index)
                else:
                    raise ValueError("Invalid mode")
                
        return DivideMixCIFAR10Dataset