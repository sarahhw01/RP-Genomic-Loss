import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Sequence prediction model
class SequenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.model.hidden_size, config.model.num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        return self.fc(x)  # output: (batch_size, seq_len, num_classes)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {cfg.run_name}")
    logger.info(f"Training config: {cfg.train}")
    logger.info(f"Model config: {cfg.model}")

    output_dir = os.getcwd()
    logger.info(f"Outputs will be saved to: {output_dir}")

    model = SequenceModel(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Fake dataset: (batch, seq_len, hidden_size)
    batch_size = cfg.train.batch_size
    seq_len = cfg.model.seq_len
    hidden_size = cfg.model.hidden_size
    num_classes = cfg.model.num_classes

    X = torch.randn(batch_size, seq_len, hidden_size)
    y = torch.randint(0, num_classes, (batch_size, seq_len))  # target: class indices

    # Training loop
    for epoch in range(cfg.train.epochs):
        optimizer.zero_grad()
        outputs = model(X)  # (batch, seq_len, num_classes)

        # Reshape for loss: treat as (batch * seq_len, num_classes)
        loss = criterion(outputs.view(-1, num_classes), y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % cfg.train.log_every == 0:
            logger.info(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Loss: {loss.item():.4f}")
    model_path = "model_" + cfg.model.name + ".pt"
    # Save the model
    save_path = os.path.join(output_dir, model_path)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()