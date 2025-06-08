import os

def extract_unique_chars_in_order(folder_path, output_file):
    seen = set()
    char_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            title = os.path.splitext(filename)[0]
            for char in title:
                if char != '-' and char not in seen:
                    seen.add(char)
                    char_list.append(char)

    # 결과를 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(char_list))  # 리스트 형태로 저장

    print(f"{len(char_list)}개의 고유 문자를 저장했습니다 → {output_file}")

# 사용 예시
folder_path = "data/labels/train"  # 실제 폴더 경로
output_file = "test/char_list.txt"
extract_unique_chars_in_order(folder_path, output_file)
