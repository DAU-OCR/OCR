# logic/image_filter.py

def select_best_image(results):
    # TODO: 이미지 중 가장 좋은 OCR 결과 선택 로직 구현 예정
    if not results:
        return None

    # confidence 가장 높은 결과 리턴 (더미)
    best = max(results, key=lambda r: r["confidence"])
    return best
