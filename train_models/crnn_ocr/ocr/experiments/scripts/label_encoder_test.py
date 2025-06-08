from utils.label_encoder import LabelEncoder

encoder = LabelEncoder()
text = "01가0785"

encoded = encoder.encode(text)
decoded = encoder.decode(encoded)

print("ENCODED:", encoded)
print("DECODED:", decoded)

print("문자 리스트:", encoder.get_charset())
print("문자 '0' 인덱스:", encoder.encode("0"))  # 이게 0이면 문제 있음!