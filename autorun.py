import os
from shutil import copy
from subprocess import run

configs_dir = './Configs'

for config_file in os.listdir(configs_dir):
    copy(os.path.join(configs_dir,config_file),"./config.yaml")
    try:
        run(['python','run.py'])
    except:
        print(f"\nError for file {config_file}!\n")
        continue
