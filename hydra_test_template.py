import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import torch.nn as nn
from train_sequence import SequenceModel  # Reuse the model definition

@hydra.main(config_path="conf", config_name="config", version_base=None)
def test(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Running test with config: {cfg}")

    # Load model
    model = SequenceModel(cfg)
    model_path = os.path.join(cfg.test.model_dir)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate or load test data
    batch_size = cfg.test.batch_size
    seq_len = cfg.model.seq_len
    hidden_size = cfg.model.hidden_size
    num_classes = cfg.model.num_classes

    X_test = torch.randn(batch_size, seq_len, hidden_size)
    y_test = torch.randint(0, num_classes, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(X_test)  # (batch, seq_len, num_classes)
        predictions = torch.argmax(outputs, dim=-1)  # (batch, seq_len)

    # Accuracy
    correct = (predictions == y_test).float()
    accuracy = correct.sum() / correct.numel()
    logger.info(f"Test accuracy: {accuracy.item():.4f}")

    # Optional: loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs.view(-1, num_classes), y_test.view(-1))
    logger.info(f"Test loss: {loss.item():.4f}")

if __name__ == "__main__":
    test()
