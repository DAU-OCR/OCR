
import ast

def sort_char_list_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        char_list = ast.literal_eval(content)  # 문자열을 리스트로 변환

    char_list_sorted = sorted(char_list)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(char_list_sorted))

    print(f"정렬 완료: {file_path}")

# 사용 예시
file_path = "G:/Projects/crnn_ocr/test/char_list.txt"
sort_char_list_file(file_path)
