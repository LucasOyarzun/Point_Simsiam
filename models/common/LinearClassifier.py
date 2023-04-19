import torch
import torch.nn as nn

class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(                                                                                                           
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
