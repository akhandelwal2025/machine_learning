import os

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

path_way = get_parent_dir(0)
print(path_way)
path_way = get_parent_dir(1)
print(path_way)
