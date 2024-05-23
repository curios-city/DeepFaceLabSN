from core import pathex
from core.interact import interact as io
import os
from pathlib import Path

def main(saved_models_path):
    saved_models_names = []

    for filepath in pathex.get_file_paths(saved_models_path):
        filepath_name = filepath.name
        if filepath_name.endswith('AMP_data.dat'):
            saved_models_names.append(filepath_name.split('_')[0])

    io.log_info("选择一个保存的模型，将其从 AMP 转换为 AMPLegacy")
    io.log_info ("")

    for i, model_name in enumerate(saved_models_names):
        io.log_info(f"[{i}] : {model_name}")

    inp = io.input_str(f"Insert a value: ", "0", show_default_value=False )

    while True:
        inp = int(inp)
        if inp > len(saved_models_names)-1 or inp < 0:
            io.log_info(f"请插入一个介于0和{len(saved_models_names)-1}之间的值")
            inp = io.input_str(f"插入一个值: ", "0", show_default_value=False )
        else:
            break

    for filepath in pathex.get_paths(saved_models_path):
        filepath_name = filepath.name
        parent_str = str(filepath.parent)

        if len(filepath_name.split('_')) <= 1 : continue
        original_model_name = filepath_name.split('_')[0]
        if filepath_name.split('_')[1] != 'AMP':
            continue
        rest_of_name = filepath_name.split('_', 2)[2]
        new_name = f"{original_model_name}_AMPLegacy_{rest_of_name}"
        new_path = Path(f"{parent_str}/{new_name}")

        if Path.exists(new_path):
            name_suffix = [original_model_name[c] for c in range(len(original_model_name)-3, len(original_model_name))]
            if name_suffix[0] == '(' and name_suffix[2] == ')':
                number = int(name_suffix[1]) + 1
                old_suffix = ''
                for c in name_suffix:
                    old_suffix += c
                name_suffix[1] = str(number)
                new_suffix = ''
                for c in name_suffix:
                    new_suffix += c
                new_model_name = original_model_name.replace(old_suffix, new_suffix)
            else:
                new_model_name = original_model_name + '(1)'
            new_name = f"{new_model_name}_AMPLegacy_{rest_of_name}"
            new_path = Path(f"{parent_str}/{new_name}")

        if original_model_name == saved_models_names[inp]:
            os.rename(filepath, new_path)
