input_path = "TrainLabel/train_label.txt"
output_path = "TrainLabel/labels.csv"

with open(input_path, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

with open(output_path, "w", encoding="utf-8") as fout:
    fout.write("filename,words\n")
    for line in lines:
        if ',' not in line:
            continue
        fout.write(line.strip() + "\n")

print("✅ 변환 완료 (UTF-8 기준)")
