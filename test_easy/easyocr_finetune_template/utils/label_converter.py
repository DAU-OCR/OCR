
class CTCLabelConverter:
    def __init__(self, character_list):
        # 문자 → 인덱스
        self.char2idx = {char: i + 1 for i, char in enumerate(character_list)}
        self.char2idx['blank'] = 0
        self.idx2char = {i: char for char, i in self.char2idx.items()}

    def encode(self, text_list):
        lengths = [len(s) for s in text_list]
        encoded = [self.char2idx[c] for s in text_list for c in s]

        # 🔍 디버깅 체크
        if len(encoded) != sum(lengths):
            print("🚨 encode 오류 발생:")
            print(f" - text_list: {text_list}")
            print(f" - lengths: {lengths}")
            print(f" - sum(lengths): {sum(lengths)}")
            print(f" - len(encoded): {len(encoded)}")

        return encoded, lengths

    def decode(self, preds_idx, preds_prob=None):
        # CTC Greedy Decoding
        texts = []
        for pred in preds_idx:
            chars = []
            prev_idx = -1
            for idx in pred:
                if idx != prev_idx and idx != 0:
                    chars.append(self.idx2char.get(idx, ''))
                prev_idx = idx
            texts.append(''.join(chars))
        return texts
