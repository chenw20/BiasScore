GLOBAL_CONFIG = {}

def update_global_config(config):
    for key in GLOBAL_CONFIG:
        if key in config:
            GLOBAL_CONFIG[key] = config[key]
