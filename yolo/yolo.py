import tensorflow as tf
from tensorflow import keras
import numpy as np

CFG_FILENAME = "yolov3.cfg"

def main():
    pass

def create_blocks(cfg_filename):
    blocks = []
    current_block = {}
    lines = [line.rstrip('\n') for line in open(cfg_filename) if line != '\n' and "#" not in line]
    for line in lines:
        if "[" in line:
            if current_block != {}: #for the case at the beginning, where current_block will be empty
                blocks.append(current_block)
                current_block = {}
            current_block["type"] = line[1:-1] #trims off the [ and ] from the string
        else:
            key, value = line.split("=")
            current_block[key.rstrip()] = value.lstrip()
    return blocks







if __name__ == '__main__':
    main()