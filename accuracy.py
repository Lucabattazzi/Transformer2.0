import torch
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode
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
    
def evaluate_metrics(model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                     seq_len, device, num_examples=10, prnt=False):
    model.eval()
    
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= num_examples:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, seq_len, device
            )

            source_texts.append(batch["src_text"][0])
            expected.append(batch["tgt_text"][0])
            predicted.append(tokenizer_tgt.decode(model_out.detach().cpu().numpy()))

    # --- Metriche ---
    cer_metric  = CharErrorRate()
    wer_metric  = WordErrorRate()
    bleu_metric = BLEUScore()

    expected_bleu = [[e] for e in expected]

    cer  = cer_metric(predicted, expected).item()
    wer  = wer_metric(predicted, expected).item()
    bleu = bleu_metric(predicted, expected_bleu).item()

    # --- Stampa risultati ---
    sep = "=" * 50
    if prnt:
        print(sep)
        print(f"  Valutazione su {num_examples} esempi")
        print(sep)
        print(f"  BLEU score      : {bleu:.4f}  (↑ migliore)")
        print(f"  Word Error Rate : {wer:.4f}  (↓ migliore)")
        print(f"  Char Error Rate : {cer:.4f}  (↓ migliore)")
        print(sep)

        # --- Qualche esempio per sanity check ---
        print("\nCampione predizioni:\n")
        for src, tgt, pred in zip(source_texts[:3], expected[:3], predicted[:3]):
            print(f"  SRC  : {src}")
            print(f"  TGT  : {tgt}")
            print(f"  PRED : {pred}")
        print()

    return {"bleu": bleu, "wer": wer, "cer": cer}

######################## Execution ########################

if __name__ == "__main__":

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, validation=True)

    model = setup_model()

    metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=10
        )

    print("\n" + "="*50 + "\n")
    print(f"Full Model - BLEU: {metrics['bleu']:.4f}\n")

    for i in range(7):
        model = setup_model()
        zero_attention_head(model=model, layer_idx=0, head_idx=i)

        metrics = evaluate_metrics(
        model, val_dataloader, tokenizer_src, tokenizer_tgt,
        config['seq_len'], device, num_examples=10
        )

        print(f"Layer {i} zeroed - BLEU: {metrics['bleu']:.4f}\n")