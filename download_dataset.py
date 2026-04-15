

#Scarica una volta il dataset da Hugging Face, lo divide in train/val
#e lo salva in formato nativo Arrow. 


from datasets import load_dataset, load_from_disk
from pathlib import Path
from huggingface_hub import login

from train import get_or_build_tokenizer
from config import get_config

# Login automatico
try:
    with open("_token.txt", "r") as f:
        token = f.read().strip()
    login(token=token)
    print("✓ Login su Hugging Face completato\n")
except FileNotFoundError:
    print("⚠ Token file not found - proceeding without auth\n")
except Exception as e:
    print(f"⚠ Login error: {e}\n")

def download_dataset(
    datasource: str = "Helsinki-NLP/opus-100",
    lang_src: str = "en",
    lang_tgt: str = "it",
    dataset_folder: str = "Dataset",
):
    
    # Crea la cartella Dataset se non esiste
    dataset_path = Path(dataset_folder)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {datasource} ({lang_src}-{lang_tgt})")
    
    # Scarica il dataset completo
    ds_raw = load_dataset(
        datasource, 
        f"{lang_src}-{lang_tgt}", 
        split='train'
    )

    # Salva i tre dataset in formato Arrow (nativo HuggingFace)
    unfiltered_path = dataset_path / "unfiltered_dataset"
    
    print(f"\nSalvataggio dataset...")
    ds_raw.save_to_disk(str(unfiltered_path))
    print(f"✓ Full dataset salvato in {unfiltered_path}\n\n")


def prepare_datasets(
    train_split_ratio: float = 0.9,
    seed: int = 42,
    dataset_folder: str = "Dataset",
    max_samples: int = 300000
):
    
    print("Preparazione dataset...")

    ds_raw = load_from_disk("Dataset/unfiltered_dataset")

    # Crea la cartella Dataset se non esiste
    dataset_path = Path(dataset_folder)

    config = get_config(preload=None)
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # STEP 1: Filtra dataset per mantenere solo frasi < 350 token
    print("\nFiltrando dataset per frasi < 350 token...")
    max_seq_len = 350
    
    def is_valid_length(example):
        src_ids = tokenizer_src.encode(example['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(example['translation'][config['lang_tgt']]).ids
        return len(src_ids) < max_seq_len and len(tgt_ids) < max_seq_len
    
    ds_raw = ds_raw.filter(is_valid_length)
    print(f"✓ Dataset filtrato: {len(ds_raw)} frasi rimaste\n")

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # If max_samples is set, sample a subset of the dataset
    if max_samples and max_samples < len(ds_raw):
        ds_raw = ds_raw.shuffle(seed=seed).select(range(max_samples))
        print(f"Campionati {max_samples} esempi da {len(ds_raw)} totali")
    
    print(f"Dataset downloaded: {len(ds_raw)} esempi")
    
    # Calcola le dimensioni
    train_size = int(train_split_ratio * len(ds_raw))
    val_size = len(ds_raw) - train_size
    
    print(f"   - Full dataset: {len(ds_raw)} examples")
    print(f"   - Train set: {train_size} examples ({train_split_ratio*100:.1f}%)")
    print(f"   - Val set: {val_size} examples ({(1-train_split_ratio)*100:.1f}%)")
    print(f"   - Seed: {seed}")
    
    # Dividi il dataset usando il seed per riproducibilità
    train_ds_raw, val_ds_raw = ds_raw.train_test_split(
        test_size=1 - train_split_ratio,
        seed=seed
    ).values()

    ds_raw.save_to_disk(dataset_path / "full_dataset")
    train_ds_raw.save_to_disk(dataset_path / "train_set")
    val_ds_raw.save_to_disk(dataset_path / "test_set")
    
    # Salva i tre dataset in formato Arrow (nativo HuggingFace)
    full_path = dataset_path / "full_dataset"
    train_path = dataset_path / "train_set"
    val_path = dataset_path / "test_set"
    
    print(f"\nSalvataggio dataset...")
    ds_raw.save_to_disk(str(full_path))
    print(f"✓ Full dataset salvato in {full_path}")
    
    train_ds_raw.save_to_disk(str(train_path))
    print(f"✓ Train set salvato in {train_path}")
    
    val_ds_raw.save_to_disk(str(val_path))
    print(f"✓ Val set salvato in {val_path}")
    
    return {
        "full_dataset_path": str(full_path),
        "train_dataset_path": str(train_path),
        "val_dataset_path": str(val_path)
    }


if __name__ == "__main__":   
    download_dataset() 
    prepare_datasets()
