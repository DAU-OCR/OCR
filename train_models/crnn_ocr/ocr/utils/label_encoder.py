# label_encoder.py - 텍스트 라벨 인코딩/디코딩 클래스 정의

class LabelEncoder:
    """
    텍스트 라벨을 숫자 인덱스로 변환하고 (encode),
    숫자 인덱스를 다시 텍스트로 변환 (decode) 하는 클래스.

    CTC (Connectionist Temporal Classification) 방식의 문자 인식에서 사용됨.
    """

    def __init__(self, charset=None):
        """
        charset: 사용할 문자 집합 (없으면 한글 차량 번호판용 기본 문자셋 사용)
        """
        if charset is None:
            # 한글 차량 번호판 문자셋 + 숫자 + 알파벳
            self.charset = '가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바버배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천처초추충카커코쿠타터토투파평퍼포푸하허호홀후후히0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        else:
            self.charset = charset

        # 문자 → 인덱스 (CTC에서 0은 blank)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.charset)}  # 0은 CTC blank

        # 인덱스 → 문자
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.charset)}

        # CTC blank token 인덱스 (항상 0)
        self.blank_idx = 0  # CTC blank

    def encode(self, text):
        """
        텍스트 문자열을 인덱스 리스트로 변환
        예: "서울12가3456" → [10, 5, ..., 30]
        """
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode_ctc_standard(self, indices):
        """
        CTC decoding 방식 (표준):
        - 연속된 같은 인덱스는 1개로 축약
        - blank(0)는 무시
        """
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != self.blank_idx and idx != prev_idx:
                decoded.append(self.idx_to_char.get(idx, ''))
            prev_idx = idx
        return ''.join(decoded)
    
    def decode_keep_repeats(self, indices):
        """
        CTC decoding 방식 (단순):
        - blank(0)만 제거하고 같은 인덱스 반복은 유지
        """
        decoded = [self.idx_to_char.get(idx, '') for idx in indices if idx != self.blank_idx]
        return ''.join(decoded)
    
    def get_charset(self):
        """현재 charset 문자열 반환"""
        return self.charset

    def num_classes(self):
        """
        CTC 모델의 출력 클래스 수 반환
        (전체 문자 수 + 1 for blank)
        """
        return len(self.charset) + 1  # +1 for CTC blank