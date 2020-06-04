import sys
import time
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Concatenate, Add
CONFIG_FILE = "yolo_newv3.txt"

NUM_ANCHORS = 0
NUM_CLASSES = 0

def main():
    obj_list = parse_config()
    create_model()
    print(obj_list)

def create_model(inputs, obj_list):
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

def parse_config():
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
                module_attributes[attribute_split[0]] = attribute_interpret(attribute_split[1], True if current_module == "ZeroPadding2D" else False)
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

def attribute_interpret(attribute, zero_padding=False):
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
        return NUM_ANCHORS*(NUM_CLASSES + 5)
    elif "current_tensor" in attribute:
        stripped_line = re.sub(r'[a-z,. \[\]_]+', '', attribute)
        return int(stripped_line)
    else:
        return int(attribute)

if __name__ == '__main__':
    main()