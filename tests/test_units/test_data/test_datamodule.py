import pytest
from data.datamodule import MultiSampleDataModule 
from data.datasets.cifar10 import CIFAR10

# TODO: rewrite tests
"""
def test_ensure_list():
    obj1 = "h"
    obj2 = [obj1]

    assert ensure_list(obj1) == obj2

def test_is_broadcastable():
    obj1 = [0]
    obj2 = [1]
    obj3 = [2, 3]
    obj4 = [2, 3, 4]

    assert is_broadcastable(obj1, obj2) == True
    assert is_broadcastable(obj1, obj3) == True
    assert is_broadcastable(obj4, obj1) == True
    assert is_broadcastable(obj3, obj4) == False

def test_broadcast_init():

    class Dummy:

        def __init__(self, something) -> None:
            self.something = something
    
    class ExtraDummy(Dummy):

        def __init__(self, something) -> None:
            super().__init__(something)
            self.something_else = something

    cls1 = Dummy
    cls2 = ExtraDummy
    args1 = dict(something=1)
    args2 = dict(something=2)

    # haha, probably best to split this one up=

    objs = broadcast_init(cls1, args1)
    assert isinstance(objs, list)
    assert len(objs) == 1
    assert objs[0].something == 1

    objs = broadcast_init(cls1, [args1, args2])
    assert isinstance(objs, list)
    assert len(objs) == 2
    assert objs[0].something == 1
    assert objs[1].something == 2

    objs = broadcast_init([cls1, cls2], args1)
    assert isinstance(objs, list)
    assert len(objs) == 2
    assert isinstance(objs[0], Dummy)
    assert isinstance(objs[1], ExtraDummy)
    assert objs[0].something == 1
    assert objs[1].something == 1

    objs = broadcast_init([cls1, cls2], [args1, args2])
    assert isinstance(objs, list)
    assert len(objs) == 2
    assert isinstance(objs[0], Dummy)
    assert isinstance(objs[1], ExtraDummy)
    assert objs[0].something == 1
    assert objs[1].something == 2

    with pytest.raises(AssertionError):
        objs = broadcast_init([cls1, cls1, cls2], [args1, args2])

def test_MultiSampleDataModule():
    dataset_cls = CIFAR10
    train_args = dict(root="data/cifar")
    test_args = {**train_args, "train":False}

    datamodule = MultiSampleDataModule(
        dataset_cls, [train_args, train_args],
        [dataset_cls, dataset_cls], test_args,
        [dataset_cls, dataset_cls], [test_args, test_args],
        32, 2
    )

    assert datamodule.num_classes == 10
    assert datamodule.num_train_samples == 50000
    assert datamodule.num_val_samples == 10000
    assert datamodule.num_test_samples == 10000
    assert len(datamodule.train_dataloader()) == 2
    assert len(datamodule.val_dataloader()) == 2
    assert len(datamodule.test_dataloader()) == 2
"""