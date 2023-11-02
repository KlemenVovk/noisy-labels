import lightning as L
from torchvision import transforms
from models.samplesieve import SampleSieve
from data.cifar import CIFAR10DataModule

# TODO: callbacks
# TODO: aim logger
# TODO: data transforms

NOISE_RATE = 0.2
BATCH_SIZE = 64
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
NUM_CLASSES = 10


model = SampleSieve(INITIAL_LR, MOMENTUM, WEIGHT_DECAY, NUM_CLASSES)
trainer = L.Trainer(max_epochs=EPOCHS)
dm = CIFAR10DataModule(train_transform=transforms.ToTensor(), batch_size=BATCH_SIZE, noise_rate=NOISE_RATE, add_synthetic_noise=True)
trainer.fit(model, dm)
