import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data