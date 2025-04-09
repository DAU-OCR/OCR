class LabelEncoder:
    def __init__(self, charset=None):
        if charset is None:
            # 기본 한글 번호판용 문자 집합
            self.charset = '가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        else:
            self.charset = charset

        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.charset)}  # 0은 CTC blank
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.charset)}
        self.blank_idx = 0  # CTC blank

    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode_ctc_standard(self, indices):
        # CTC decoding: 중복 제거 + blank 제거
        decoded = []
        prev_idx = None
        for idx in indices:
            if idx != self.blank_idx and idx != prev_idx:
                decoded.append(self.idx_to_char.get(idx, ''))
            prev_idx = idx
        return ''.join(decoded)
    
    def decode_keep_repeats(self, indices):
        # blank 제거만 하고 중복은 유지
        decoded = [self.idx_to_char.get(idx, '') for idx in indices if idx != self.blank_idx]
        return ''.join(decoded)
    
    def get_charset(self):
        return self.charset

    def num_classes(self):
        return len(self.charset) + 1  # +1 for CTC blank