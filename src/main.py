import lightning as L
from models.cores2 import SampleSieve
from data.cifar import CIFAR10DataModule
from utils.cores2 import train_cifar10_transform, test_cifar10_transform
from aim.pytorch_lightning import AimLogger

# TODO: test instead of val
# TODO: saving
# TODO: aim logger

NOISE_RATE = 0.6
BATCH_SIZE = 64
INITIAL_LR = 0.1 # TODO: investigate, paper says 0.1, but code says 0.05
MOMENTUM = 0.9 # TODO: investigate, paper says 0.9, but code says 0
WEIGHT_DECAY = 5e-4 # TODO: investigate, paper says 5e-4, but code says 0
EPOCHS = 100

L.seed_everything(0)

aim_logger = AimLogger(
    experiment='cores',
    train_metric_prefix='train_',
    test_metric_prefix='test_',
    val_metric_prefix='val_',
)

data_module = CIFAR10DataModule(train_transform=train_cifar10_transform, test_transform=test_cifar10_transform, noise_rate=NOISE_RATE, batch_size=BATCH_SIZE)
data_module.setup()
model = SampleSieve(INITIAL_LR, MOMENTUM, WEIGHT_DECAY, data_module.num_classes, initial_noise_prior=data_module.noise_prior, num_training_samples=data_module.num_training_samples)
trainer = L.Trainer(max_epochs=EPOCHS, logger=aim_logger)
trainer.fit(model, data_module)
