

#Scarica una volta il dataset da Hugging Face, lo divide in train/val
#e lo salva in formato nativo Arrow. 


from datasets import load_dataset
from pathlib import Path


def prepare_datasets(
    datasource: str = "opus_books",
    lang_src: str = "en",
    lang_tgt: str = "it",
    train_split_ratio: float = 0.9,
    seed: int = 42,
    dataset_folder: str = "Dataset"
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
    
    return {
        "full_dataset_path": str(full_path),
        "train_dataset_path": str(train_path),
        "val_dataset_path": str(val_path)
    }


if __name__ == "__main__":    
    prepare_datasets()
