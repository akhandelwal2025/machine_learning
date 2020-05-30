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
    for idx, block in enumerate(blocks):
        if block["type"] == "convolutional":
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            pad = (block["size"] - 1)//2 if int(block["padding"]) else 0
            model.add(keras.layers.Conv2D(filters=block["filters"], kernel_size=block["size"], strides=block["stride"], padding=pad, activation=keras.layers.LeakyReLU(alpha=0.1), bias=bias))
            if batch_normalize == 1:
                model.add(keras.layers.BatchNormalization(axis=1))

        elif block["type"] == "shortcut":
            from_int = int(block["from"])
            output_from_layer = model.layers[idx+from_int].output
            output_prev_layer = model.layers[idx-1].output
            shortcut_output = output_from_layer + output_prev_layer
            shortcut_layer = keras.layers.Dense(len(shortcut_output))(shortcut_output)
            model.add(shortcut_layer)
            model.add(keras.layers.Activation('linear'))

        elif block["type"] == "upsample":
            model.add(keras.layers.UpSampling2D(interpolation='bilinear')) #Bilinear upscaling uses all nearby pixels to figure out new pixel values

        elif block["type"] == "route":
            numbers = block["layers"].split(", ")
            if len(numbers) == 1: #Implies only start connection
                output_from_layer = model.layers[idx+numbers[0]].output
                route_layer = keras.layers.Dense(len(output_from_layer))(output_from_layer)
                model.add(route_layer)
            else:
                output_start_layer = model.layers[idx+numbers[0]].output
                output_end_layer = model.layers[numbers[1]].output
                output = keras.layers.concatenate([output_start_layer, output_end_layer], axis=1)
                route_layer = keras.layers.Dense(output)(output)
                model.add(route_layer)

        else: #implies yolo layer
            pass






if __name__ == '__main__':
    main()