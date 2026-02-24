import torch
from pathlib import Path

def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str
):
    """Saves a PyTorch model to a target directory with a target name.

    Args:
        model: A PyTorch model to be saved.
        target_dir: A directory to save the model to.
        model_name: The name of the saved model file (e.g. "model.pth").
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pth or .pt"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)