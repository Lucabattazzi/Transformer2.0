import torch
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode, evaluate_metrics
from temperature import zero_attention_head

def setup_model():
    # Load the pretrained weights
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    return model

def translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()
    with torch.no_grad():
        # Tokenizza la frase sorgente
        enc = tokenizer_src.encode(sentence)
        tokens = [tokenizer_src.token_to_id('[SOS]')] + enc.ids + [tokenizer_src.token_to_id('[EOS]')]
        
        # Padding fino a seq_len (usa il max_len del modello)
        pad_id = tokenizer_src.token_to_id('[PAD]')
        seq_len = max_len  # oppure il seq_len fisso del tuo modello
        padded = tokens + [pad_id] * (seq_len - len(tokens))
        
        encoder_input = torch.tensor([padded], dtype=torch.long).to(device)  # (1, seq_len)
        encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)  # (1,1,1,seq_len)
        
        # Greedy decode
        output_ids = greedy_decode(model, encoder_input, encoder_mask,
                                   tokenizer_src, tokenizer_tgt, max_len, device)
        
        # Decodifica (rimuove SOS/EOS)
        output_text = tokenizer_tgt.decode(output_ids.detach().cpu().numpy())
        return output_text
    

######################## Execution ########################

if __name__ == "__main__":

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, validation=True)

    model = setup_model()

    num_examples = 100

    metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=num_examples, n_gram=2
        )

    print("\n" + "="*50 + "\n")
    print(f"Full Model - BLEU: {metrics['bleu']:.4f}")

    # Layer_0
    hottest_heads = [5, 3]
    coldest_heads = [1, 4]

    print("\n" + "="*50 + "\n")

    # Without 2 hottest heads
    model = setup_model()
    for i in hottest_heads:
        zero_attention_head(model=model, layer_idx=0, head_idx=i)

    metrics = evaluate_metrics(
    model, val_dataloader, tokenizer_src, tokenizer_tgt,
    config['seq_len'], device, num_examples=num_examples, n_gram=2
    )

    print(f"Layer 0 without {len(hottest_heads)} hottest heads - BLEU: {metrics['bleu']:.4f}")
    print("\n" + "="*50 + "\n")

    # Without 2 coldest heads
    model = setup_model()
    for i in coldest_heads:
        zero_attention_head(model=model, layer_idx=0, head_idx=i)

    metrics = evaluate_metrics(
    model, val_dataloader, tokenizer_src, tokenizer_tgt,
    config['seq_len'], device, num_examples=num_examples, n_gram=2
    )

    print(f"Layer 0 without {len(coldest_heads)} coldest heads - BLEU: {metrics['bleu']:.4f}")
    print("\n" + "="*50 + "\n")

    # With only 2 hottest heads
    model = setup_model()
    for i in range(8):
        if i not in hottest_heads:
            zero_attention_head(model=model, layer_idx=0, head_idx=i)

    metrics = evaluate_metrics(
    model, val_dataloader, tokenizer_src, tokenizer_tgt,
    config['seq_len'], device, num_examples=num_examples, n_gram=2
    )

    print(f"Layer 0 with only {len(hottest_heads)} hottest heads - BLEU: {metrics['bleu']:.4f}")
    print("\n" + "="*50 + "\n")

    # With only 2 coldest heads
    model = setup_model()
    for i in range(8):
        if i not in coldest_heads:
            zero_attention_head(model=model, layer_idx=0, head_idx=i)

    metrics = evaluate_metrics(
    model, val_dataloader, tokenizer_src, tokenizer_tgt,
    config['seq_len'], device, num_examples=num_examples, n_gram=2
    )

    print(f"Layer 0 with only {len(coldest_heads)} coldest heads - BLEU: {metrics['bleu']:.4f}")
    print("\n" + "="*50 + "\n")