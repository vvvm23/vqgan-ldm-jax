import re

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

import torch

from .modeling_flax_vqgan import VQModel
from .configuration_vqgan import VQGANConfig

import zipfile
from urllib.request import urlretrieve
from pathlib import Path
from omegaconf import OmegaConf

regex = r"\w+[.]\d+"

NAME_TO_MODEL_URL = {
    'vq-f4': "https://ommer-lab.com/files/latent-diffusion/vq-f4.zip",
    'vq-f4-noattn': "https://ommer-lab.com/files/latent-diffusion/vq-f4-noattn.zip",
    'vq-f8': "https://ommer-lab.com/files/latent-diffusion/vq-f8.zip",
    'vq-f8-n256': "https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip",
    'vq-f16': "https://ommer-lab.com/files/latent-diffusion/vq-f16.zip"
}

NAME_TO_CONFIG_URL = {
    'vq-f4': "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/vq-f4/config.yaml",
    'vq-f4-noattn': "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/vq-f4-noattn/config.yaml",
    'vq-f8': "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/vq-f8/config.yaml",
    'vq-f8-n256': "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/vq-f8-n256/config.yaml",
    'vq-f16': "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/vq-f16/config.yaml",
}

def download_model(name, base: str = 'vqgan-ckpt'):
    base_dir = Path(base)
    base_dir.mkdir(exist_ok=True)

    assert name in NAME_TO_MODEL_URL, f"Unrecognised model configuration '{name}'"
    model_dir = base_dir / name
    model_dir.mkdir(exist_ok = True)

    print(f"Downloading model checkpoint '{name}'.")
    urlretrieve(NAME_TO_MODEL_URL[name], model_dir / 'model.zip')

    print(f"Extracting model checkpoint.")
    zipfile.ZipFile(model_dir / 'model.zip', 'r').extractall(model_dir)

    (model_dir / 'model.zip').unlink()

def download_config_yaml(name, base: str = 'vqgan-ckpt'):
    base_dir = Path(base)
    base_dir.mkdir(exist_ok=True)

    assert name in NAME_TO_CONFIG_URL, f"Unrecognised model configuration '{name}'"
    model_dir = base_dir / name
    model_dir.mkdir(exist_ok = True)

    print(f"Downloading model config '{name}'.")
    urlretrieve(NAME_TO_CONFIG_URL[name], model_dir / 'config.yaml')

def load_config(config_path: str) -> VQGANConfig:
    pt_config = OmegaConf.load(config_path)
    # essentially flatten ddconfig into params
    return VQGANConfig(**pt_config.model.params, **pt_config.model.params.ddconfig)

def rename_key(key):
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


# Adapted from https://github.com/huggingface/transformers/blob/ff5cdc086be1e0c3e2bbad8e3469b34cffb55a85/src/transformers/modeling_flax_pytorch_utils.py#L61
def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    random_flax_state_dict = flatten_dict(flax_model.params)
    flax_state_dict = {}

    remove_base_model_prefix = (
        flax_model.base_model_prefix
        not in flax_model.params) and (flax_model.base_model_prefix in set(
            [k.split(".")[0] for k in pt_state_dict.keys()]))
    add_base_model_prefix = (flax_model.base_model_prefix in flax_model.params
                             ) and (flax_model.base_model_prefix not in set(
                                 [k.split(".")[0]
                                  for k in pt_state_dict.keys()]))

    # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    for pt_key, pt_tensor in pt_state_dict.items():
        pt_tuple_key = tuple(pt_key.split("."))

        has_base_model_prefix = pt_tuple_key[0] == flax_model.base_model_prefix
        require_base_model_prefix = (flax_model.base_model_prefix,
                                     ) + pt_tuple_key in random_flax_state_dict

        if remove_base_model_prefix and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]
        elif add_base_model_prefix and require_base_model_prefix:
            pt_tuple_key = (flax_model.base_model_prefix, ) + pt_tuple_key

        # Correctly rename weight parameters
        if ("norm" in pt_key and (pt_tuple_key[-1] == "bias")
            and (pt_tuple_key[:-1] + ("bias", ) not in random_flax_state_dict)
                and (pt_tuple_key[:-1] + ("scale", ) in random_flax_state_dict)):
            pt_tuple_key = pt_tuple_key[:-1] + ("scale", )
        elif pt_tuple_key[-1] in [
            "weight", "gamma"
        ] and pt_tuple_key[:-1] + ("scale", ) in random_flax_state_dict:
            pt_tuple_key = pt_tuple_key[:-1] + ("scale", )
        if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + (
                "embedding", ) in random_flax_state_dict:
            pt_tuple_key = pt_tuple_key[:-1] + ("embedding", )
        elif pt_tuple_key[
                -1] == "weight" and pt_tensor.ndim == 4 and pt_tuple_key not in random_flax_state_dict:
            # conv layer
            pt_tuple_key = pt_tuple_key[:-1] + ("kernel", )
            pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        elif pt_tuple_key[
                -1] == "weight" and pt_tuple_key not in random_flax_state_dict:
            # linear layer
            pt_tuple_key = pt_tuple_key[:-1] + ("kernel", )
            pt_tensor = pt_tensor.T
        elif pt_tuple_key[-1] == "gamma":
            pt_tuple_key = pt_tuple_key[:-1] + ("weight", )
        elif pt_tuple_key[-1] == "beta":
            pt_tuple_key = pt_tuple_key[:-1] + ("bias", )

        if pt_tuple_key in random_flax_state_dict:
            if pt_tensor.shape != random_flax_state_dict[pt_tuple_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[pt_tuple_key].shape}, but is {pt_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[pt_tuple_key] = jnp.asarray(pt_tensor)

    # TODO: remove this and just change code above
    # definitely a way to do that but my tired brain doesn't want to do right now
    def fix_tuple(t):
        numerics = [i for i, v in enumerate(t) if v.isnumeric()]
        for ni in numerics[::-1]:
            t = list(t[:ni-1]) + [t[ni-1] + '_' + t[ni]] + list(t[ni+1:])

        if t[-2][:-1] in ['norm'] and t[-1] == 'kernel':
            t[-1] = 'scale'
        if t[-2] in ['norm'] and t[-1] == 'kernel':
            t[-1] = 'scale'

        return tuple(t)

    flax_state_dict = {fix_tuple(k): v for k, v in flax_state_dict.items()}
    return unflatten_dict(flax_state_dict)


def load_model(config: VQGANConfig, pt_state_dict_path: str, dtype = jnp.float16):
    model = VQModel(config, dtype=dtype)

    state_dict = torch.load(pt_state_dict_path, map_location="cpu")[
        "state_dict"]
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("loss"):
            state_dict.pop(key)
            continue
        renamed_key = rename_key(key)
        state_dict[renamed_key] = state_dict.pop(key)

    state = convert_pytorch_state_dict_to_flax(state_dict, model)
    model.params = state
    #model.save_pretrained(save_path)
    return model

def load_and_download_model(name, base: str = 'vqgan-ckpt', dtype = jnp.float16):
    model_path = Path(base) / name / 'model.ckpt'
    config_path = Path(base) / name / 'config.yaml'

    if not model_path.is_file(): download_model(name, base)
    if not config_path.is_file(): download_config_yaml(name, base)

    config = load_config(config_path)
    return load_model(config, model_path, dtype = dtype)
