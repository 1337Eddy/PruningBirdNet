from abc import abstractclassmethod
import re

class SelectMask():

    def __init__(self):
        self.resstack_pattern = "module\.classifier\.[0-9]+\.classifier\."
        self.resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\."
        self.fst_bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[3].weight"
        self.snd_bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[7].weight"
        self.bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[3|7].weight"
        self.scaling_block_pattern = "module\.classifier\.[0-9]+\.classifier\.[1-9]+\.W"
        self.len_resstack = len("module.classifier.1")
        self.len_resblock = len("module.classifier.1.classifier.2")

    def group_key_name_list_in_stacks(self, layer_name_list):
        stacks = []
        temp = []
        name = layer_name_list[0]
        for key in layer_name_list:
            if key[:self.len_resstack] == name[:self.len_resstack]:
                temp.append(key)
            else: 
                stacks.append(temp)
                temp = [] 
                name = key[:self.len_resstack]
                temp.append(key)
        stacks.append(temp)
        return stacks 

    def sort_dict_by_key(self, dict):
        buffer = {}
        keys = dict.keys()
        keys = sorted(keys)
        for key in keys:
            buffer[key] = dict[key] 
        return buffer

    def select_layers(self, model_state_dict, pattern):
        sub_dict = {}
        for key, value in model_state_dict.items():
            element = re.search(pattern, key)
            if element:
                sub_dict[key] = value 
        return sub_dict

    @classmethod
    @abstractclassmethod
    def get_mask_key_tuple(self, model_state_dict, ratio, block_temperature):
        pass 


