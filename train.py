import argparse
from argparse import Namespace

import numpy as np
import datasets

import espnetez as ez
from espnet2.bin.s2t_inference import Speech2Text


import logging
logging.basicConfig(level=logging.INFO)


def tokenize_fn(text, tokenizer, converter):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="espnet/owsm_v4_base_102M")
    parser.add_argument("--config_path", type=str, default="conf/finetune.yaml")
    parser.add_argument("--exp_dir", type=str, default="./exp/finetune")
    parser.add_argument("--stats_dir", type=str, default="./exp/stats_finetune")
    args = parser.parse_args()

    # Load dataset
    fleurs_language = "en_us"
    owsm_language = "eng"
    train_dataset = datasets.load_dataset("google/fleurs", fleurs_language)["train"]
    valid_dataset = datasets.load_dataset("google/fleurs", fleurs_language)["validation"]

    # Load pretrained model
    pretrained_model = Speech2Text.from_pretrained(
        args.model_name,
        beam_size=1,
        device='cpu'
    )

    def build_model_fn(args):
        model = pretrained_model.s2t_model
        model.train()
        print(f"Trainable parameters: {count_parameters(model)}")
        return model

    # Tokenizer and converter
    tokenizer = pretrained_model.tokenizer
    converter = pretrained_model.converter

    def tokenize(text):
        return tokenize_fn(text, tokenizer, converter)

    # Data mapping
    data_info = {
        "speech": lambda d: d['audio']['array'].astype(np.float32), # 1-D raw waveform
        "text": lambda d: tokenize(f"<{owsm_language}><asr><notimestamps> {d['transcription']}"), # tokenized text mapped to integer ids
        "text_prev": lambda d: tokenize("<na>"), # tokenized text of previous utterance for prompting, unused here
        "text_ctc": lambda d: tokenize(d['transcription']), # tokenized text mapped to integer ids for CTC loss, can be different from "text" depending on task
    }

    train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
    valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)

    # Load finetuning config
    config = Namespace(**ez.config.update_finetune_config(
        "s2t",
        vars(pretrained_model.s2t_train_args),
        args.config_path
    ))

    # Setup trainer
    trainer = ez.Trainer(
        task="s2t",
        train_config=config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=args.exp_dir,
        stats_dir=args.stats_dir,
        ngpu=1,
    )

    # Collect stats and train
    trainer.collect_stats()
    trainer.train()


if __name__ == "__main__":
    main()
