import argparse
from pathlib import Path

import torch
import datasets

from espnet2.bin.s2t_inference import Speech2Text



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="espnet/owsm_v4_base_102M")
    parser.add_argument("--exp_dir", type=str, default="./exp/finetune")
    parser.add_argument("--src", type=str, default="de")
    parser.add_argument("--trg", type=str, default="en")
    args = parser.parse_args()

    # Load dataset
    fleurs_language = "en_us"
    owsm_language = "eng"
    europarl_test = datasets.load_dataset(
        "google/fleurs",
        fleurs_language,
        trust_remote_code=True,
    )["test"]

    # Load pretrained model
    pretrained_model = Speech2Text.from_pretrained(
        args.model_name,
        device="cuda"
    )

    # Evaluate fine-tuned model
    sample = europarl_test[0]

    # Evaluate original model
    # pretrained_model.s2t_model.load_state_dict(torch.load("original.pth"))
    pred = pretrained_model(sample["audio"]["array"])
    print("Original Model")
    print("PREDICTED:", pred[0][3])
    print("REFERENCE:", sample["transcription"])

    # Evaluate fine-tuned model
    pretrained_model.s2t_model.load_state_dict(torch.load(Path(args.output_dir) / "1epoch.pth"))
    pred = pretrained_model(sample["audio"]["array"])
    print("\nFine-tuned Model")
    print("PREDICTED:", pred[0][3])
    print("REFERENCE:", sample["text_ctc"])

