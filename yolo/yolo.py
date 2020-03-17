import tensorflow as tf
from tensorflow import keras
import numpy as np

CFG_FILENAME = "yolov3.cfg"

def main():
    blocks = create_blocks(CFG_FILENAME)
    print(blocks)
    model = create_model(blocks)

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

def create_model(blocks):
    model = keras.Sequential()
    for block in blocks:
        if block["type"] == "convolutional":
            model.add(keras.layers.Conv2D(filters=block["filters"], kernel_size=block["size"], strides=block["stride"], padding=block["pad"], activation=keras.layers.LeakyReLU(alpha=0.1)))
        elif block["type"] == "shortcut":
            pass
        elif block["type"] == "upsample":
            pass
        elif block["type"] == "route":
            pass
        else: #implies yolo layer
            pass






if __name__ == '__main__':
    main()