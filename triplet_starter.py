# Triplet rule:
# anchor   = Event1, Game1
# positive = Event1, Game2
# negative = Event2, Game1

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


frame_root = ".../Research/Soccer/multigame_event_frames"
folder_layout = "event_game"


# one image plus labels
@dataclass(frozen=True)
class Sample:
    path: Path
    event: str
    game: str


# read labels
def get_labels(path: Path, root: Path, layout: str) -> tuple[str, str]:
    a, b = path.relative_to(root).parts[:2]
    return (a, b) if layout == "event_game" else (b, a)


def load_samples(frame_root: str, layout: str) -> list[Sample]:
    root = Path(frame_root).expanduser()
    images = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in root.rglob(ext))
    samples = [Sample(p, *get_labels(p, root, layout)) for p in images]
    if not samples:
        raise ValueError(f"No images found under {root}")
    return samples


# sample on the fly.
class TripletFrames(Dataset):
    def __init__(self, samples: list[Sample], image_size: int = 112, n_triplets: int = 512):
        self.samples, self.n_triplets, self.rng = samples, n_triplets, random.Random(7)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # input size
                transforms.ToTensor(),  # PIL image to tensor
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # fast lookup tables for triplet sampling.
        self.by_event_game = defaultdict(list)
        self.by_game = defaultdict(list)
        for i, s in enumerate(samples):
            self.by_event_game[(s.event, s.game)].append(i)
            self.by_game[s.game].append(i)

        # valid anchor needs same event in another game and another event in same game.
        self.valid = [i for i, s in enumerate(samples) if self.has_pos(s) and self.has_neg(s)]
        if not self.valid:
            raise ValueError("Need one event in 2+ games and one game with 2+ events.")

    # can this sample find Event1 in a different game?
    def has_pos(self, s: Sample) -> bool:
        return any(event == s.event and game != s.game for event, game in self.by_event_game)

    # can this sample find Event2 in the same game?
    def has_neg(self, s: Sample) -> bool:
        return any(self.samples[i].event != s.event for i in self.by_game[s.game])

    # load one image file as a normalized tensor
    def image(self, i: int) -> torch.Tensor:
        with Image.open(self.samples[i].path) as img:
            return self.transform(img.convert("RGB"))

    def __len__(self) -> int:
        return self.n_triplets

    # return anchor, positive, negative
    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.rng.choice(self.valid)
        anchor = self.samples[a]

        pos_games = [g for e, g in self.by_event_game if e == anchor.event and g != anchor.game]
        p = self.rng.choice(self.by_event_game[(anchor.event, self.rng.choice(pos_games))])

        neg_pool = [i for i in self.by_game[anchor.game] if self.samples[i].event != anchor.event]
        n = self.rng.choice(neg_pool)
        return self.image(a), self.image(p), self.image(n)


# ResNet18 backbone with L2-normalized embeddings
class Embedder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(features, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.head(self.backbone(x)), p=2, dim=1)


# ... device() whatever is available


# train for a few epochs and print whether positives are closer than negatives
def train() -> None:
    samples = load_samples(frame_root, folder_layout)
    data = TripletFrames(samples)
    loader = DataLoader(data, batch_size=16, shuffle=False)

    dev = device()
    model = Embedder().to(dev)
    loss_fn = nn.TripletMarginLoss(margin=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"device={dev} images={len(samples)} valid_anchors={len(data.valid)}")

    for epoch in range(1, 4):
        loss_sum = pos_sum = neg_sum = correct = violations = seen = 0
        for anchor, pos, neg in loader:
            anchor, pos, neg = anchor.to(dev), pos.to(dev), neg.to(dev)
            a, p, n = model(anchor), model(pos), model(neg)
            loss = loss_fn(a, p, n)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # success means pos_dist < neg_dist.
            with torch.no_grad():
                pos_dist = torch.linalg.vector_norm(a - p, dim=1)
                neg_dist = torch.linalg.vector_norm(a - n, dim=1)
                b = anchor.shape[0]
                loss_sum += loss.item() * b
                pos_sum += pos_dist.sum().item()
                neg_sum += neg_dist.sum().item()
                correct += (pos_dist < neg_dist).sum().item()
                violations += (pos_dist + 0.2 >= neg_dist).sum().item()
                seen += b

        print(
            f"epoch={epoch} "
            f"loss={loss_sum / seen:.4f} "
            f"pos_dist={pos_sum / seen:.4f} "
            f"neg_dist={neg_sum / seen:.4f} "
            f"triplet_acc={correct / seen:.3f} "
            f"margin_violations={violations / seen:.3f}"
        )

    torch.save(model.state_dict(), "triplet_model.pt")


if __name__ == "__main__":
    train()
