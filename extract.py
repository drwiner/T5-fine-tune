from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForQuestionAnswering
import os
import torch.nn as nn
import dataclasses
from typing import Optional

def convert_to_text(full_model_path: str, class_text: dict):
    """ Converts a model to text files, recursively"""
    previous_line_started: bool = False
    with open(full_model_path, 'r') as f:
        for line in f:
            if line.startswith("class") or previous_line_started:
                if not previous_line_started:
                    model_name = line.split("(")[0].split(" ")[1]
                    model_text = []
                    model_text.append(line)
                else:
                    previous_line_started = False

                while True:
                    line = f.readline()
                    if len(line) == 0:
                        break

                    if line[0] == " " or line[0] == "\n":
                        model_text.append(line)
                    else:
                        class_text[model_name] = model_text
                        if line.startswith("class"):
                            model_name = line.split("(")[0].split(" ")[1]
                            model_text = []
                            model_text.append(line)
                            previous_line_started = True
                        break

@dataclasses.dataclass
class TextModule:
    name: str
    module: Optional[nn.Module]
    text: list[str] = dataclasses.field(default_factory=list)
    children: list['TextModule'] = dataclasses.field(default_factory=list)

def breakdown_module(class_name: str, module: nn.Module, class_text: dict) -> TextModule:
    if class_name not in class_text:
        print(f"No answer for : {class_name}")
        return None

    text_module = TextModule(name=class_name, module=module, text=class_text[class_name])

    for name, child in module.named_children():
        if isinstance(child, nn.ModuleList):
            module_list = TextModule(name=name, module=None, text=[""])
            for sub_module in child:
                mod_child: TextModule = breakdown_module(sub_module._get_name(), sub_module, class_text)
                # mod_child.name = f"{name}_{mod_child.name}"
                if mod_child is not None:
                    module_list.children.append(mod_child)
            text_module.children.append(module_list)
            continue
        mod_child: TextModule = breakdown_module(child._get_name(), child, class_text)
        if mod_child is not None:
            # mod_child.name = f"{name}_{mod_child.name}"
            text_module.children.append(mod_child)

    return text_module


def text_module_to_files(text_module: TextModule, file_path: str, flat_folder: bool = True):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    file_text = os.path.join(file_path, f"{text_module.name}.py")
    with open(file_text, 'w') as f:
        for line in text_module.text:
            f.write(line)

    for child in text_module.children:
        if flat_folder:
            text_module_to_files(child, file_path, flat_folder)
        else:
            text_module_to_files(child, os.path.join(file_path, child.name), flat_folder)

if __name__ == "__main__":

    path = "/users/drw/transformers/src/transformers/models/"
    pytorch_path = "/users/drw/pytorch/torch/nn/modules/"

    modeling_files = []
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        for file in os.listdir(os.path.join(path, folder)):
            if file.startswith(f"modeling_{folder}"):
                model_file = os.path.join(path, folder, file)
                print(model_file)
                modeling_files.append(model_file)

    for file in os.listdir(pytorch_path):
        if os.path.isdir(file):
            continue
        model_file = os.path.join(pytorch_path, file)
        print(model_file)
        modeling_files.append(model_file)

    class_to_text = {}
    for file in modeling_files:
        convert_to_text(file, class_to_text)

    # config = AutoConfig.from_pretrained("google/flan-ul2")
    # config.num_heads=2
    # config.num_layers=2
    # config.num_decoder_layers=2
    # config.vocab_size=24
    # bertModel = AutoModelForSeq2SeqLM.from_config(config)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")#, load_in_8bit=True, device_map="auto")
    seed: str = llm_model._get_name()
    tm = breakdown_module(seed, llm_model, class_to_text)
    print(tm)
    text_module_to_files(tm, f"./flan-t5-base/", flat_folder=False)




