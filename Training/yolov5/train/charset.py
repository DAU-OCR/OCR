import json

# TXT 경로 입력
txt_path = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/charset_korean.txt'
json_path = 'charset_korean.json'

with open(txt_path, 'r', encoding='utf-8') as f:
    charset = [line.strip() for line in f if line.strip()]

# 중복 제거 + 정렬
charset = sorted(set(charset))

# JSON 저장
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({'char_list': charset}, f, ensure_ascii=False, indent=2)

print(f'✅ {json_path} 생성 완료!')
print(f'총 문자 수: {len(charset)}')
print(charset)