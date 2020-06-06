import sys
import time
import re
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Concatenate, Add, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo_loss_function import yolo_loss
from utils import data_generator_wrapper, get_parent_dir, change_to_new_machine

CONFIG_FILE = "yolo_config.txt"
DARKNET_WEIGHTS_FILE = "darknet_weights.h5"
ANNOTATIONS_FILE = os.path.join(get_parent_dir(0), "training_images", "vott-csv-export", "data_train.txt")
CLASSES_FILE = os.path.join(get_parent_dir(0), "data_classes.txt")
ANCHORS_FILE = os.path.join(get_parent_dir(0), "yolo_anchors.txt")
LOG_DIR = os.path.join(get_parent_dir(0), "log_dir")
LOG_DIR_TIME = os.path.join(LOG_DIR, "{}".format(int(time())))
VAL_PERCENT = 0.1

def main():
    anchors, num_anchors = get_anchors()
    num_classes = get_classes()
    annotation_lines = [line.strip(" ") for line in open(ANNOTATIONS_FILE)]
    change_to_new_machine(annotation_lines, repo="yolo", remote_machine="")
    obj_list = parse_config(num_anchors, num_classes)
    generate_model(anchors, num_classes, annotation_lines, obj_list)
    print(obj_list)


def generate_model(anchors, num_classes, annotation_lines, obj_list):
    image_input = Input(shape=(None, None, 3)) #Specifies input tensor of unknown HxW, but three dimensions
    num_anchors = len(anchors)
    image_height, image_width = 416, 416
    batch_size = 32
    y_true = [Input(shape=(image_height // {0: 32, 1: 16, 2: 8}[l], image_width // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5,))for l in range(3)]
    num_train = len(annotation_lines) * VAL_PERCENT
    num_validation = len(annotation_lines) - num_train
    checkpoint, reduce_lr, early_stopping = create_callbacks()

    model_body = create_yolo_body(image_input, obj_list)

    model_body.load_weights(filepath=DARKNET_WEIGHTS_FILE, by_name=True, skip_mismatch=True)
    for num in range(len(model_body.layers)-3):
        model_body.layers[num].trainable = False

    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss", arguments={
        "anchors": anchors,
        "num_classes": num_classes,
        "ignore_thresh": 0.5,
        },
    )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    train(model, annotation_lines, 1e-3, num_train, num_classes, num_validation, (image_height, image_width), anchors, batch_size, [checkpoint], 51, 0)

    for num in range(len(model_body.layers)):
        model.layers[num].trainable = True
    train(model, annotation_lines, 1e-4, num_train, num_classes, num_validation, (image_height, image_width), anchors, batch_size, [checkpoint, reduce_lr, early_stopping], 102, 51)

def create_callbacks():
    checkpoint = ModelCheckpoint(os.path.join(LOG_DIR, "checkpoint.h5"), monitor="val_loss", save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)
    return checkpoint, reduce_lr, early_stopping

def train(model, annotation_lines, learning_rate, num_train, num_classes, num_val, input_shape, anchors, batch_size, callbacks, epoch_num, initial_epoch_num):
    model.compile(optimizer=Adam(lr=learning_rate), loss={"yolo_loss": lambda y_true, y_pred: y_pred})
    history = model.fit_generator(
        data_generator_wrapper(
            annotation_lines[:num_train], batch_size, input_shape, anchors, num_classes
        ),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(
            annotation_lines[num_train:], batch_size, input_shape, anchors, num_classes
        ),
        validation_steps=max(1, num_val // batch_size),
        epochs=epoch_num,
        initial_epoch=initial_epoch_num,
        callbacks=callbacks,
    )

    model.save_weights(os.path.join(LOG_DIR, "trained_weights_stage_1.h5"))

    step1_train_loss = history.history["loss"]

    file = open(os.path.join(LOG_DIR_TIME, "step1_loss.npy"), "w")
    with open(os.path.join(LOG_DIR_TIME, "step1_loss.npy"), "w") as f:
        for item in step1_train_loss:
            f.write("%s\n" % item)
    file.close()

    step1_val_loss = np.array(history.history["val_loss"])

    file = open(os.path.join(LOG_DIR_TIME, "step1_val_loss.npy"), "w")
    with open(os.path.join(LOG_DIR_TIME, "step1_val_loss.npy"), "w") as f:
        for item in step1_val_loss:
            f.write("%s\n" % item)
    file.close()

def create_yolo_body(inputs, obj_list):
    darknet_model = None
    current_tensor = inputs

    last_layers_part2_list = []
    for obj in obj_list:
        obj_name = list(obj.keys())[0]
        if obj_name == "Darknet":
            darknet_model = Model(inputs, create_obj_layers(inputs, obj[obj_name]))
            current_tensor = darknet_model.output
        elif obj_name == "Concatenate":
            current_tensor = create_obj_layers(current_tensor, obj[obj_name], darknet=darknet_model)
        elif obj_name == "Last_Layers_Part2":
            last_layers_part2_list.append(create_obj_layers(current_tensor, obj[obj_name]))
        else:
            current_tensor = create_obj_layers(current_tensor, obj[obj_name])
    return Model(current_tensor, last_layers_part2_list)

def create_obj_layers(inputs, module_list, darknet=None):
    current_tensor = inputs
    for layer_tup in module_list:
        layer_name = layer_tup[0]
        if layer_name == "Conv2D":
            current_tensor = Conv2D(**layer_tup[1])(current_tensor)
        elif layer_name == "BatchNormalization":
            current_tensor = BatchNormalization(**layer_tup[1])(current_tensor)
        elif layer_name == "LeakyReLU":
            current_tensor = LeakyReLU(**layer_tup[1])(current_tensor)
        elif layer_name == "ZeroPadding2D":
            current_tensor = ZeroPadding2D(**layer_tup[1])(current_tensor)
        elif layer_name == "UpSampling2D":
            current_tensor = UpSampling2D(**layer_tup[1])(current_tensor)
        elif layer_name == "Resblock":
            multiplier = layer_tup[1]
            filters = layer_tup[2]
            for block in range(multiplier):
                temp = create_obj_layers(current_tensor, layer_tup[3])
                current_tensor = Add()([current_tensor, temp])
        else:
            darknet_num = layer_tup[1]["inputs"]
            current_tensor = Concatenate()([current_tensor, darknet.layers[darknet_num].output])
    return current_tensor

def parse_config(num_anchors, num_classes):
    line_count = 0 #Can't use for-loop, so using counter + while loop
    lines = []
    with open(CONFIG_FILE) as txt_wrapper:
        lines = [line.rstrip('\n') for line in txt_wrapper]

    obj_list = []  #[{"Darknet":[("Conv2D", {"filters":64, "kernel_size":(3,3)})]}]
    current_obj = "NONE"
    obj_modules_list = []

    resblock_list = []
    resblock_multiplier = 1
    resblock_filters = 0
    in_resblock = False
    while line_count < len(lines):
        line = lines[line_count]
        if line == "":
            line_count += 1
            continue
        elif line[0] == '<' and line[len(line)-1] == ">": #Start/end of object (EX. <Darknet> OR </Darknet>)
            if line[1] == "/":
                obj_list.append({current_obj:obj_modules_list})
                obj_modules_list=[]
            else:
                current_obj = str(line.strip(" <>")) #Will strip whitespace + delimiter from leading/trailing edges
        elif line[0] == "[" and line[len(line)-1] == "]":
            current_module = str(line.strip(" []"))
            module_attributes = {} #{"filters":64, "kernel_size":(3,3), ....}
            line_count += 1
            while lines[line_count] != "":
                attribute_split = lines[line_count].split(":")
                module_attributes[attribute_split[0]] = attribute_interpret(attribute_split[1], True if current_module == "ZeroPadding2D" else False, num_anchors, num_classes)
                line_count += 1
            if in_resblock:
                resblock_list.append((current_module, module_attributes)) #Tuple in form (Conv2D, {filters:64, "kernel_size":(3,3)}). This is in case whether a resblock is present
            else:
                obj_modules_list.append((current_module, module_attributes)) #If not in resblock, append to module lists
        elif line[0] == "/" and line[len(line)-1] == "/": #indicates /x2x1024/ or /*x2/ (multiplier, filters)
            if line[1] == "*":
                obj_modules_list.append(("Resblock", resblock_multiplier, resblock_filters, resblock_list))
                resblock_list = []
                resblock_multiplier = 1
                resblock_filters = 0
                in_resblock = False
            else:
                parts = line.strip(" /").split("x")
                resblock_multiplier = int(parts[1])
                resblock_filters = int(parts[2])
                in_resblock = True
        else:
            line_count += 1
            continue
        line_count += 1
    return obj_list

def attribute_interpret(attribute, zero_padding=False, num_anchors=0, num_classes=0):
    if attribute == "False" or attribute == "True":
        return False if attribute == "False" else "True"
    elif attribute == "valid" or attribute == "same":
        return attribute
    elif attribute == "tf.keras.regularizer.l2(5e-4)":
        return tf.keras.regularizers.l2(5e-4)
    elif attribute[0] == "(" and attribute[len(attribute)-1] == ")":
        if zero_padding:
            return ((1,0),(1,0))
        else:
            nums = attribute.strip(" ()").split(",")
            return (nums[0], nums[1])
    elif attribute == "0.1":
        return float(0.1)
    elif attribute == "num_anchors * (num_classes + 5)":
        return num_anchors*(num_classes + 5)
    elif "current_tensor" in attribute:
        stripped_line = re.sub(r'[a-z,. \[\]_]+', '', attribute)
        return int(stripped_line)
    else:
        return int(attribute)

def get_classes():
    with open(CLASSES_FILE) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors():
    with open(ANCHORS_FILE) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    main()