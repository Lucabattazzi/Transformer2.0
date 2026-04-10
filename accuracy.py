import torch

from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode

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
    
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename, map_location=device)
model.load_state_dict(state['model_state_dict'])

s = "About two o'clock p.m. I entered the village."
t = translate_sentence(s, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
print("SOURCE: ", s)
print("PREDICTED:", t)