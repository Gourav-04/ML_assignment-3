from torch import nn


def classifier(name):
    """Choose the reference dense network, the overparameterized memoriser, or the CNN."""
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    if name == "memory":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    if name != "cnn":
        raise ValueError(f"Unknown classifier: {name}")
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 96, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.Linear(384, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


class Autoencoder(nn.Module):
    """Compress 3x32x32 pixels through a 48x4x4 feature map to a fixed-width vector.
    The mirrored decoder ends in a sigmoid because the target pixels lie in [0,1].
    """

    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(size, 768),
            nn.ReLU(),
            nn.Unflatten(1, (48, 4, 4)),
            nn.ConvTranspose2d(48, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PatchMLP(nn.Module):
    """Embed each 4x4 RGB patch with the same layer, then classify all 64 embeddings."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(48, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 48), nn.ReLU(), nn.Linear(48, 10))

    def forward(self, x):
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 1, 4, 5)
        return self.head(self.embed(patches.reshape(len(x), 64, 48)))
