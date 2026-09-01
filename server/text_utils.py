import re
from collections import Counter

import cv2

# 한글 매핑 딕셔너리
dict_map = {
    'gyeongnam': '경남', 'busan': '부산', 'jeo': '저', 'meo': '머', 'seo': '서', 'beo': '버',
    'geo': '거', 'neo': '너', 'deo': '더', 'bu': '부', 'do': '도', 'no': '노', 'go': '고', 'ro': '로',
    'bo': '보', 'jo': '조', 'gu': '구', 'na': '나', 'ma': '마', 'ba': '바', 'sa': '사', 'ah': '아',
    'ja': '자', 'cha': '차', 'ka': '카', 'ta': '타', 'pa': '파', 'ha': '하', 'la': '라', 'ra': '라',
    'me': '머', 'mu': '무', 'su': '수', 'ho': '호', 'ru': '루', 'mo': '모', 'ke': '커', 'ne': '네',
    'je': '제', 'yu': '유', 'se': '서', 'mi': '미', 'ju': '주', 'de': '데', 'oe': '외', 'wa': '와',
    'wi': '위', 'ri': '리', 'ye': '예', 'yi': '이', 'u': '우', 'eo': '어', 'heo': '허', 'du': '두',
    'leo': '러', 'lu': '루', 'so': '소', 'da': '다', 'lo': '로', 'nu': '누', 'o': '오', 'ga': '가'
}
dict_sorted = sorted(dict_map.items(), key=lambda x: len(x[0]), reverse=True)

# 번호판에 사용되는 모든 한글 문자 집합
VALID_HANGUL_CHARS = {
    # Private vehicles
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조', '구', '누', '두', '루', '무',
    '부', '수', '우', '주',
    # Rental cars
    '허', '하', '호',
    # Commercial vehicles (Taxis, Buses)
    '바', '사', '아', '자',
    # Delivery
    '배',
    # Military (optional)
    '육', '해', '공', '국', '합'
}


# OCR 유틸 함수
def roman_to_korean(text):
    t = text.lower()
    for roman, kor in dict_sorted:
        t = t.replace(roman, kor)
    return t

def normalize(text):
    return re.sub(r'[^가-힣0-9]', '', text)

def is_valid_plate(text):
    return bool(re.fullmatch(r'\d{2,3}[가-힣]\d{4}', text))

def reorder_blocks(blocks):
    if len(blocks) == 2:
        if re.search(r'[가-힣]', blocks[1]) and not re.search(r'[가-힣]', blocks[0]):
            return [blocks[1], blocks[0]]
    return blocks

def insert_hangul_fixed(digits: str, hangul: str) -> str:
    if len(digits) >= 7:
        pos = -5
        return digits[:pos] + hangul + digits[pos+1:]
    return digits

def patch_hangul(t1, t2):
    d1 = ''.join(re.findall(r'\d', t1))
    d2 = ''.join(re.findall(r'\d', t2))
    h1 = re.findall(r'[가-힣]', t1)
    h2 = re.findall(r'[가-힣]', t2)
    if len(d1) in [7, 8] and len(h2) == 1:
        p = insert_hangul_fixed(d1, h2[0])
        if is_valid_plate(p): return p
    if len(d2) in [7, 8] and len(h1) == 1:
        p = insert_hangul_fixed(d2, h1[0])
        if is_valid_plate(p): return p
    return None

def get_filtered_ocr(reader, image, resize):
    resized = cv2.resize(image, resize)
    result = reader.readtext(resized)
    if not result:
        return '', 0.0
    blocks = reorder_blocks([t for (_, t, _) in result])
    merged = ''.join(blocks)
    norm = normalize(merged)
    conf = max([c for (_, _, c) in result])
    return norm, round(conf, 2)

def apply_plate_selection_logic(t1, c1, t2, c2, t3, c3, hangul_dict):
    results = [
        {'text': t1, 'conf': c1, 'name': '모델1'},
        {'text': t2, 'conf': c2, 'name': '모델2'},
        {'text': t3, 'conf': c3, 'name': '모델3(CRNN)'}
    ]

    # --- 1단계: 다수결 투표 ---
    # 정규식과 한글 사전을 모두 통과한 결과만 투표 자격 부여
    vote_candidates = []
    for r in results:
        if is_valid_plate(r['text']):
            hangul_char = re.findall(r'[가-힣]', r['text'])
            if hangul_char and hangul_char[0] in hangul_dict:
                vote_candidates.append(r['text'])

    if len(vote_candidates) >= 2:
        vote_counts = Counter(vote_candidates)
        most_common = vote_counts.most_common(1)[0]
        if most_common[1] >= 2: # 2개 이상 동의
            return most_common[0], f"다수결({most_common[1]}표)"

    # --- 2단계: 모델1 우선 규칙 ---
    is_t1_valid = is_valid_plate(t1)
    if is_t1_valid:
        hangul_char = re.findall(r'[가-힣]', t1)
        if hangul_char and hangul_char[0] in hangul_dict:
            return t1, "모델1우선"

    # --- 3단계: 가중치 적용 비교 ---
    # 2단계에서 모델1이 채택되지 않았으므로, 모델1을 제외한 나머지로 비교
    remaining_results = [r for r in results if r['name'] != '모델1']
    valid_results = [r for r in remaining_results if is_valid_plate(r['text'])]

    if valid_results:
        for r in valid_results:
            hangul_char = re.findall(r'[가-힣]', r['text'])
            r['in_dict'] = hangul_char and hangul_char[0] in hangul_dict

        dict_results = [r for r in valid_results if r['in_dict']]

        def get_adjusted_conf(r):
            if r['name'] == '모델3(CRNN)':
                return r['conf'] * 0.9
            return r['conf']

        if dict_results:
            best = max(dict_results, key=get_adjusted_conf)
            return best['text'], f"사전+가중치({best['name']})"
        else:
            best = max(valid_results, key=get_adjusted_conf)
            return best['text'], f"정규식+가중치({best['name']})"

    # --- 4단계: 패치 및 최종 선택 ---
    patch_pairs = [(t1, t2), (t1, t3), (t2, t3)]
    for p1, p2 in patch_pairs:
        patched = patch_hangul(p1, p2)
        if patched and is_valid_plate(patched):
            hangul_char = re.findall(r'[가-힣]', patched)
            if hangul_char and hangul_char[0] in hangul_dict:
                return patched, '패치'

    def get_adjusted_conf_final(r):
        if r['name'] == '모델3(CRNN)':
            return r['conf'] * 0.9
        return r['conf']
    best = max(results, key=get_adjusted_conf_final)
    return best['text'], f"conf+가중치({best['name']})"
