import lmdb
import os
import shutil

# 테스트용 LMDB 경로
output_path = 'C:/Users/HOME/Desktop/testyolo/lmdb/lmdb_test'

# 이전에 있던 폴더 삭제
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

try:
    # LMDB 열기
    env = lmdb.open(output_path, map_size=1024 * 1024)  # 1MB
    with env.begin(write=True) as txn:
        txn.put(b'key1', b'value1')
        txn.put(b'key2', b'value2')
        txn.put(b'key3', b'value3')
    print("✅ LMDB 쓰기 성공")

    # LMDB 읽기 테스트
    with env.begin(write=False) as txn:
        value = txn.get(b'key2')
        print("✅ key2의 값:", value)

except lmdb.Error as e:
    print("❌ LMDB 에러 발생:", e)
