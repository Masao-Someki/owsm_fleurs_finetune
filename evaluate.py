import argparse
from pathlib import Path

import torch
from dataset.dataset import EuroparlSTDataset

from espnet2.bin.s2t_inference import Speech2Text



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="espnet/owsm_v4_base_102M")
    parser.add_argument("--exp_dir", type=str, default="./exp/finetune")
    parser.add_argument("--src", type=str, default="de")
    parser.add_argument("--trg", dtype=str, default="en")
    parser.add_argument("--debug_sample", action="store_true")
    args = parser.parse_args()

    # Load dataset
    europarl_test = EuroparlSTDataset(split=f"{args.src}_{args.trg}_test")

    # Load pretrained model
    pretrained_model = Speech2Text.from_pretrained(
        args.model_name,
    )

    if args.debug_sample:
        # Evaluate fine-tuned model
        id, sample = europarl_test[0]
        pretrained_model.s2t_model.cuda()
        pretrained_model.device = "cuda"

        # Evaluate original model
        pretrained_model.s2t_model.load_state_dict(torch.load("original.pth"))
        pred = pretrained_model(sample["speech"])
        print("Original Model")
        print("PREDICTED:", pred[0][3])
        print("REFERENCE:", sample["text_raw"])

        # Evaluate fine-tuned model
        pretrained_model.s2t_model.load_state_dict(torch.load(Path(args.output_dir) / "1epoch.pth"))
        pred = pretrained_model(sample["speech"])
        print("\nFine-tuned Model")
        print("PREDICTED:", pred[0][3])
        print("REFERENCE:", sample["text_raw"])

