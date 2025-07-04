from omegaconf import OmegaConf

def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True) 