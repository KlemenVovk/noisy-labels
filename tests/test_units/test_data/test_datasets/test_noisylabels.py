from data.datasets.noisylabels import NoisyLabelsLoader


def test_cifar10_load_all(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, True)
    labels = nl.load_all()

    assert isinstance(labels, dict)
    for key in nl.cifar10_label_names:
        assert key in labels

def test_cifar10_load_label(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, True)
    
    for key in nl.cifar10_label_names:
        nl.load_label(key)

def test_cifar100_load_all(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar100", shared_tmp_path, True)
    labels = nl.load_all()

    assert isinstance(labels, dict)
    for key in nl.cifar100_label_names:
        assert key in labels

def test_cifar100_load_label(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar100", shared_tmp_path, True)
    
    for key in nl.cifar100_label_names:
        nl.load_label(key)

def test_download(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, True)
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, False)
    nl.load_all()

def test_integrity_check(shared_tmp_path):
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, True)
    with open(nl.save_path, "w") as f:
        f.write("hhh")
    nl = NoisyLabelsLoader("cifar10", shared_tmp_path, True)
    nl.load_all()