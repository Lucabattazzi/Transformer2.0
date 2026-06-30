import torch
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode, evaluate_metrics, calculate_validation_loss
from temperature import zero_attention_head

def setup_model(config, tokenizer_src, tokenizer_tgt, device):
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
    


def create_table(bleu_samples, val_samples):

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


    data = [[],[], [], []]

    metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=bleu_samples, n_gram=2
        )

    print("\n" + "="*50 + "\n")
    print(f"Full Model - BLEU: {metrics['bleu']:.4f}")
    full_bleu = metrics['bleu']

    full_val_loss = val_loss = calculate_validation_loss(model, val_dataloader, loss_fn, tokenizer_tgt, device, val_samples)

    # Layer_0
    hottest_heads = [1, 5, 7]
    coldest_heads = [0, 4, 6]

    print("\n" + "="*50 + "\n")

    # Without hottest heads
    model = setup_model(config, tokenizer_src, tokenizer_tgt, device)
    for i, head in enumerate(hottest_heads):
        # print("Removing head", head)
        zero_attention_head(model=model, layer_idx=0, head_idx=head)
        # print(f"Layer 0 without {i+1} hottest head(s) - Evaluating...")

        metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=bleu_samples, n_gram=2
        )

        val_loss = calculate_validation_loss(model, val_dataloader, loss_fn, tokenizer_tgt, device, val_samples)

        data[0].append(metrics['bleu'])
        data[2].append(val_loss)
        # print(f"Layer 0 without {i+1} hottest heads - BLEU: {metrics['bleu']:.4f}")
        # print("\n" + "="*50 + "\n")

    # Without coldest heads
    model = setup_model(config, tokenizer_src, tokenizer_tgt, device)
    for i, head in enumerate(coldest_heads):
        zero_attention_head(model=model, layer_idx=0, head_idx=head)

        metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=bleu_samples, n_gram=2
        )

        val_loss = calculate_validation_loss(model, val_dataloader, loss_fn, tokenizer_tgt, device, val_samples)

        data[1].append(metrics['bleu'])
        data[3].append(val_loss)
        # print(f"Layer 0 without {i+1} coldest heads - BLEU: {metrics['bleu']:.4f}")
        # print("\n" + "="*50 + "\n")

    # data[0] -> BLEU dopo rimozione progressiva dei hottest heads (1,2,3)
    # data[1] -> BLEU dopo rimozione progressiva dei coldest heads (1,2,3)
    # data[2] -> LOSS dopo rimozione progressiva dei hottest heads (1,2,3)
    # data[3] -> LOSS dopo rimozione progressiva dei coldest heads (1,2,3)
    print("\n" + "="*80 + "\n")
    print("BLEU comparison table (Full model vs removed heads)")
    print("="*80)
    print(f"{'Config':<25} | {'BLEU hot':>10} | {'BLEU cold':>10} | {'Loss hot':>10} | {'Loss cold':>10}")
    print('-'*80)
    print(f"{'Full model':<25} | {full_bleu:10.4f} | {full_bleu:10.4f} | {full_val_loss:10.4f} | {full_val_loss:10.4f}")
    for idx in range(max(len(data[0]), len(data[1]))):
        hot = f"{data[0][idx]:.4f}" if idx < len(data[0]) else "-"
        cold = f"{data[1][idx]:.4f}" if idx < len(data[1]) else "-"
        hot_loss = f"{data[2][idx]:.4f}" if idx < len(data[2]) else "-"
        cold_loss = f"{data[3][idx]:.4f}" if idx < len(data[3]) else "-"
        print(f"{'Without '+str(idx+1)+' head(s)':<25} | {hot:>10} | {cold:>10} | {hot_loss:>10} | {cold_loss:>10}")
    print('='*80)

    data = [[],[], [], []]

    # With only hottest heads
    for j in range(1, 4):  # j = 1, 2, 3 teste calde
        model = setup_model(config, tokenizer_src, tokenizer_tgt, device)  # reset fresco ad ogni iterazione
        hottest = hottest_heads[:j]

        # Azzera tutte le teste che NON sono tra le più calde
        for head_idx in range(8):
            if head_idx not in hottest:
                zero_attention_head(model=model, layer_idx=0, head_idx=head_idx)

        metrics = evaluate_metrics(
            model, val_dataloader, tokenizer_src, tokenizer_tgt,
            config['seq_len'], device, num_examples=bleu_samples, n_gram=2
        )

        val_loss = calculate_validation_loss(model, val_dataloader, loss_fn, tokenizer_tgt, device, val_samples)
        
        data[0].append(metrics['bleu'])
        data[2].append(val_loss)
        # print(f"Layer 0 with only {j+1} hottest heads - BLEU: {metrics['bleu']:.4f}")
        # print("\n" + "="*50 + "\n")

    # With only coldest heads
    for j in range(1, 4):  # j = 1, 2, 3 teste calde
        model = setup_model(config, tokenizer_src, tokenizer_tgt, device)  # reset fresco ad ogni iterazione
        coldest = coldest_heads[:j]

        # Azzera tutte le teste che NON sono tra le più calde
        for head_idx in range(8):
            if head_idx not in coldest:
                zero_attention_head(model=model, layer_idx=0, head_idx=head_idx)

        metrics = evaluate_metrics(
            model, val_dataloader, tokenizer_src, tokenizer_tgt,
            config['seq_len'], device, num_examples=bleu_samples, n_gram=2
        )

        val_loss = calculate_validation_loss(model, val_dataloader, loss_fn, tokenizer_tgt, device, val_samples)
        
        data[1].append(metrics['bleu'])
        data[3].append(val_loss)
        # print(f"Layer 0 with only {j+1} coldest heads - BLEU: {metrics['bleu']:.4f}")
        # print("\n" + "="*50 + "\n") 

    # data[0] -> BLEU con solo hottest heads (1,2,3)
    # data[1] -> BLEU con solo coldest heads (1,2,3)
    # data[2] -> LOSS con solo hottest heads (1,2,3)
    # data[3] -> LOSS con solo coldest heads (1,2,3)
    print("\n" + "="*80 + "\n")
    print("BLEU comparison table (Full model vs remaining heads)")
    print("="*80)
    print(f"{'Config':<25} | {'BLEU hot':>10} | {'BLEU cold':>10} | {'Loss hot':>10} | {'Loss cold':>10}")
    print('-'*80)
    print(f"{'Full model':<25} | {full_bleu:10.4f} | {full_bleu:10.4f} | {full_val_loss:10.4f} | {full_val_loss:10.4f}")
    for idx in range(max(len(data[0]), len(data[1]))):
        hot = f"{data[0][idx]:.4f}" if idx < len(data[0]) else "-"
        cold = f"{data[1][idx]:.4f}" if idx < len(data[1]) else "-"
        hot_loss = f"{data[2][idx]:.4f}" if idx < len(data[2]) else "-"
        cold_loss = f"{data[3][idx]:.4f}" if idx < len(data[3]) else "-"
        print(f"{'With only '+str(idx+1)+' head(s)':<25} | {hot:>10} | {cold:>10} | {hot_loss:>10} | {cold_loss:>10}")
    print('='*80)
######################## Execution ########################

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
    # create_table(bleu_samples=10, val_samples=500)