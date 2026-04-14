

#Scarica una volta il dataset da Hugging Face, lo divide in train/val
#e lo salva in formato nativo Arrow. 


from datasets import load_dataset
from pathlib import Path
from huggingface_hub import login

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


def prepare_datasets(
    datasource: str = "Helsinki-NLP/opus-100",
    lang_src: str = "en",
    lang_tgt: str = "it",
    train_split_ratio: float = 0.9,
    seed: int = 42,
    dataset_folder: str = "Dataset",
    max_samples: int = 300000
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
    prepare_datasets()
