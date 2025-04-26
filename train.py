import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy simple model for now
class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.model.hidden_size, 1)

    def forward(self, x):
        return self.fc(x)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {cfg.run_name}")
    logger.info(f"Training config: {cfg.train}")
    logger.info(f"Model config: {cfg.model}")

    # Set up output directory (Hydra does this automatically)
    output_dir = os.getcwd()
    logger.info(f"Outputs will be saved to: {output_dir}")

    # Instantiate model
    model = SimpleModel(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Fake dataset
    X = torch.randn(100, cfg.model.hidden_size)
    y = torch.randint(0, 2, (100, 1)).float()

    # Training loop
    for epoch in range(cfg.train.epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % cfg.train.log_every == 0:
            logger.info(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {loss.item():.4f}")

    # Save the model
    save_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()