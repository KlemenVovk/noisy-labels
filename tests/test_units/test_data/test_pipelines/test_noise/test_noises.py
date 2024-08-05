import torch

from noisypy.data.pipelines.noise.noises import InstanceNoise, AsymmetricNoise, SymmetricNoise, LambdaNoise

# TODO remember mat2 and calculate 
# teoretical guarantees for n_samples to converge

def test_persistance():
    noise = InstanceNoise(torch.tensor([3, 4]))
    sample1 = ("image1", 1, 0) # image, target, index
    sample2 = ("image2", 1, 1)

    noisy_target1 = noise(*sample1)
    noisy_target2 = noise(*sample1)
    noisy_target3 = noise(*sample2)

    assert noisy_target1 == noisy_target2
    assert noisy_target1 != noisy_target3

def test_instance_noise():
    noise = InstanceNoise(torch.tensor([0, 1, 2]))
    
    for i in range(3):
        assert noise(None, 0, i) == i

def test_asymmetric_noise():
    tr_mtx = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    count_mtx = torch.ones_like(tr_mtx)
    noise = AsymmetricNoise(tr_mtx)

    # run sample n_trials times
    n_trials = 100000
    for i in range(n_trials):
        sample = (None, i%tr_mtx.shape[0], i)
        noisy_target = noise(*sample)
        count_mtx[sample[1], noisy_target] += 1

    # normalize
    est_tr_mtx = count_mtx / count_mtx.sum(axis=1, keepdim=True)
    
    assert torch.allclose(tr_mtx, est_tr_mtx, atol=0.01)
    

def test_symmetric_noise():
    noise = SymmetricNoise(2, 0.1)
    tr_mtx = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    assert torch.allclose(noise.transition_matrix, tr_mtx)

    count_mtx = torch.ones_like(tr_mtx)

    # run sample n_trials times
    n_trials = 100000
    for i in range(n_trials):
        sample = (None, i%tr_mtx.shape[0], i)
        noisy_target = noise(*sample)
        count_mtx[sample[1], noisy_target] += 1

    # normalize
    est_tr_mtx = count_mtx / count_mtx.sum(axis=1, keepdim=True)
    
    assert torch.allclose(tr_mtx, est_tr_mtx, atol=0.01)

def test_lambda_noise():
    
    def noise_fcn(img, tgt, index):
        return [1, 0][tgt]
    
    noise = LambdaNoise(noise_fcn)
    tr_mtx = torch.tensor([[0., 1], [1, 0]])
    count_mtx = torch.ones_like(tr_mtx)

    # run sample n_trials times
    n_trials = 100000
    for i in range(n_trials):
        sample = (None, i%tr_mtx.shape[0], i)
        noisy_target = noise(*sample)
        count_mtx[sample[1], noisy_target] += 1

    # normalize
    est_tr_mtx = count_mtx / count_mtx.sum(axis=1, keepdim=True)
    
    assert torch.allclose(tr_mtx, est_tr_mtx, atol=0.01)
    