import torch
from config import get_config
from train import get_ds
from accuracy import setup_model



def print_model_overview(model, max_depth=4):
    """
    Stampa una panoramica gerarchica del modello e dei suoi parametri.

    Args:
        model: modello PyTorch già costruito o caricato.
        max_depth: profondità massima dei sotto-moduli mostrati.
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 115)
    print(f"MODEL OVERVIEW: {model.__class__.__name__}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 115)

    print(
        f"{'Module':<55} "
        f"{'Type':<28} "
        f"{'Direct params':>15} "
        f"{'Params in subtree':>18}"
    )
    print("-" * 115)

    for name, module in model.named_modules():
        if name == "":
            continue

        depth = name.count(".") + 1
        if depth > max_depth:
            continue

        direct_params = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=False)
        )

        subtree_params = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=True)
        )

        if subtree_params == 0:
            continue

        indentation = "  " * (depth - 1)
        displayed_name = indentation + name

        print(
            f"{displayed_name:<55} "
            f"{module.__class__.__name__:<28} "
            f"{direct_params:>15,} "
            f"{subtree_params:>18,}"
        )

    print("=" * 115 + "\n")



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)
    
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, validation=True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'))

    model = setup_model(config, tokenizer_src, tokenizer_tgt, device)

    print_model_overview(model, max_depth=4)