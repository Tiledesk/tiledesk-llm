from huggingface_hub import snapshot_download

def prepare_huggingface_model(model_name: str):
    """Scarica e cachea il modello Hugging Face"""
    return snapshot_download(
        repo_id=model_name,
        local_dir=f"./models/{model_name.replace('/', '_')}"
    )