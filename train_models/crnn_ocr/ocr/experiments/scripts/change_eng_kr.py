import os

# === 1. 파일이 들어 있는 폴더 경로 ===
folder_path = './testfile'  # 여기를 실제 폴더 경로로 바꿔주세요

# === 2. 영문 → 한글 매핑 테이블 ===
kor_map = {
    "ga": "가", "na": "나", "da": "다", "ra": "라", "ma": "마",
    "ba": "바", "sa": "사", "ah": "아", "ja": "자", "cha": "차",
    "ka": "카", "ta": "타", "pa": "파", "ha": "하",
    "geo": "거", "neo": "너", "deo": "더", "reo": "러", "meo": "머",
    "beo": "버", "seo": "서", "eo": "어", "jeo": "저",
    "go": "고", "no": "노", "do": "도", "ro": "로", "mo": "모",
    "bo": "보", "so": "소", "o": "오", "jo": "조", "cho": "초",
    "ko": "코", "to": "토", "po": "포", "ho": "호",
    "bu": "부", "du": "두"
}

# === 3. 길이가 긴 키부터 먼저 대체되도록 정렬 ===
sorted_keys = sorted(kor_map.keys(), key=lambda x: -len(x))

# === 4. 파일 이름 변경 실행 ===
renamed_count = 0

for filename in os.listdir(folder_path):
    name, ext = os.path.splitext(filename)
    new_name = name

    for key in sorted_keys:
        if key in new_name:
            new_name = new_name.replace(key, kor_map[key])
            break  # 한 번만 변경

    if new_name != name:
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name + ext)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name + ext}")
        renamed_count += 1

print(f"\n✅ 총 {renamed_count}개의 파일명을 변경했습니다.")
