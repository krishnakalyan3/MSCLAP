from datasets import load_dataset, Audio
from transformers import Trainer, TrainingArguments
from msclap.models.clap import CLAP
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Usage
config_path = '/home/MSCLAP/msclap/configs/config_2023.yml'
config = load_config(config_path)

# Load Model
clap = CLAP(
            audioenc_name=config["audioenc_name"],
            sample_rate=config["sampling_rate"],
            window_size=config["window_size"],
            hop_size=config["hop_size"],
            mel_bins=config["mel_bins"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            classes_num=config["num_classes"],
            out_emb=config["out_emb"],
            text_model=config["text_model"],
            transformer_embed_dim=config["transformer_embed_dim"],
            d_proj=config["d_proj"]
        )



# Load Dataset
dataset = load_dataset("/home/MSCLAP/ears")

# DataLoader
train = dataset["train"].cast_column("flac", Audio(sampling_rate=16000))
val = dataset["test"].cast_column("flac", Audio(sampling_rate=16000))

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    learning_rate=3e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    output_dir="../output"
)

def collate_fn(batch):
    inputs = torch.stack([item['flac'] for item in batch])
    labels = torch.tensor([item['json']["text"] for item in batch])
    return {'input_values': inputs, 'labels': labels}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    data_collator=collate_fn,
)

trainer.train()
