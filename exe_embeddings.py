import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Embeddings
from dataset_embeddings import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI for Embeddings")
parser.add_argument("--config", type=str, default="embeddings_base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--missing_ratio", type=float, default=0.1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--embeddings_file", type=str, required=True, help="Path to embeddings file")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["missing_ratio"] = args.missing_ratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/embeddings_{current_time}/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    embeddings_file=args.embeddings_file,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["missing_ratio"],
    seed=args.seed
)

model = CSDI_Embeddings(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
