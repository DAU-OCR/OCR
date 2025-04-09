import os
import json

def convert_json_to_txt(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            path = os.path.join(json_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                value = data['value']
                base_name = os.path.splitext(fname)[0]
                txt_path = os.path.join(output_dir, f'{base_name}.txt')
                with open(txt_path, 'w', encoding='utf-8') as out:
                    out.write(value)

# 변환 실행
convert_json_to_txt('data/labels/train_json', 'data/labels/train')
convert_json_to_txt('data/labels/val_json', 'data/labels/val')