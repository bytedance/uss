import argparse
from pathlib import Path

from uss.inference import separate
from uss.utils import get_path

model_paths_dict = {
    "at_soft": {
        "config_yaml": {
            "path": Path(Path.home(), ".cache/uss/scripts/ss_model=resunet30,querynet=at_soft,data=full.yaml"),
            "remote_path": "https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull.yaml?download=1",
            "size": 1558,
        },
        "checkpoint": {
            "path": Path(Path.home(), ".cache/uss/checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=1000000.ckpt"),
            "remote_path": "https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull%2Cdevices%3D8%2Cstep%3D1000000.ckpt",
            "size": 1121024828,
        },
    }
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--audio_path", type=str, required=True)
    parser.add_argument(
        "-c",
        "--condition_type",
        type=str,
        default="at_soft",
        choices=[
            "at_soft",
            "embedding"])
    parser.add_argument("--levels", nargs="*", type=int, default=[])
    parser.add_argument("--class_ids", nargs="*", type=int, default=[])
    parser.add_argument("--queries_dir", type=str, default="")
    parser.add_argument("--query_emb_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args()

    condition_type = args.condition_type

    # Use default pretrained models
    if condition_type == "at_soft":
        args.config_yaml = get_path(
            meta=model_paths_dict[condition_type]["config_yaml"])
        args.checkpoint_path = get_path(
            meta=model_paths_dict[condition_type]["checkpoint"])

    elif condition_type == "embedding":
        pass

    else:
        raise NotImplementedError

    separate(args)
