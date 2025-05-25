# import argparse

# def get_opt():
#     parser = argparse.ArgumentParser()

#     # ✅ 데이터 경로
#     # LMDB 경로로 직접 지정 (이게 핵심!)
#     parser.add_argument('--train_data', default='C:/Users/HOME/Desktop/testyolo/easyOCR/dataset/training', help='학습용 LMDB 경로')
#     parser.add_argument('--valid_data', default='C:/Users/HOME/Desktop/testyolo/easyOCR/dataset/validation', help='검증용 LMDB 경로')


#     # ✅ 저장 경로 및 실험명
#     parser.add_argument('--experiment_name', default='korean_lp_finetune', help='저장 폴더 이름')
#     parser.add_argument('--saved_model', default='saved_models/korean_g2.pth', help='pretrained 모델 경로')

#     # ✅ 문자셋
#     parser.add_argument('--character', type=str,
#         default="0123456789가강거경계고공관광교구금기김나남너노누다대더도동두등라러로루릉리마머명모목무문미바배백버보부북사산서성세소수시아악안양어여연영오외용우울원육의이인자작저전정제조종주준중지차창천청초추춘충카타태통파평포표하해허협호홀홍흥히",
#         help='사용할 문자셋')

#     # ✅ 모델 구조
#     parser.add_argument('--Transformation', default='TPS', type=str)
#     parser.add_argument('--FeatureExtraction', default='ResNet', type=str)
#     parser.add_argument('--SequenceModeling', default='BiLSTM', type=str)
#     parser.add_argument('--Prediction', default='CTC', type=str)

#     # ✅ 입력 이미지 설정
#     parser.add_argument('--imgH', type=int, default=32)
#     parser.add_argument('--imgW', type=int, default=100)
#     parser.add_argument('--rgb', action='store_true', help='컬러 이미지 사용 여부')

#     # ✅ 학습 설정
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--valInterval', type=int, default=2000, help='validation 주기')
#     parser.add_argument('--num_iter', type=int, default=30000, help='총 학습 스텝')
#     parser.add_argument('--workers', type=int, default=4)
#     parser.add_argument('--grad_clip', type=float, default=5.0)

#     # ✅ 옵티마이저 & 학습률
#     parser.add_argument('--lr', type=float, default=1.0)
#     parser.add_argument('--beta1', type=float, default=0.9)
#     parser.add_argument('--rmsprop', action='store_true')
#     parser.add_argument('--adam', action='store_true')

#     # ✅ 기타 설정
#     parser.add_argument('--PAD', action='store_true', help='비율 유지한 padding 적용')
#     parser.add_argument('--contrast_adjust', type=float, default=0.0)
#     parser.add_argument('--sensitive', action='store_true', help='대소문자 구분')
#     parser.add_argument('--manualSeed', type=int, default=1111)
#     parser.add_argument('--new_prediction', action='store_true', help='CTC 마지막 레이어 새로 초기화')
#     parser.add_argument('--FT', action='store_true', help='feature extractor만 freezing')

#     parser.add_argument('--data_filtering_off', action='store_true', help='문자셋 외의 문자는 필터링하지 않음')
#     parser.add_argument('--select_data', type=str, default='/', help='LMDB 하위 전체 사용')
#     parser.add_argument('--batch_ratio', type=str, default='1', help='각 서브 데이터셋의 배치 비율')
#     parser.add_argument('--num_fiducial', type=int, default=20, help='TPS에서 사용할 fiducial points 개수')
#     parser.add_argument('--input_channel', type=int, default=1, help='입력 이미지 채널 수 (흑백은 1, 컬러는 3)')
#     parser.add_argument('--output_channel', type=int, default=512, help='FeatureExtractor 출력 채널 수')
#     parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden state 크기')
#     parser.add_argument('--batch_max_length', type=int, default=25, help='최대 라벨 길이')
#     parser.add_argument('--total_data_usage_ratio', type=float, default=1.0, help='전체 데이터 중 사용할 비율 (0~1 사이)')
#     opt = parser.parse_args()
#     return opt


import argparse

def get_opt():
    parser = argparse.ArgumentParser()

    # ✅ 데이터 경로
    parser.add_argument('--train_data', default='C:/Users/HOME/Desktop/testyolo/easyOCR/dataset/training', help='학습용 LMDB 경로')
    parser.add_argument('--valid_data', default='C:/Users/HOME/Desktop/testyolo/easyOCR/dataset/validation', help='검증용 LMDB 경로')

    # ✅ 저장 경로 및 실험명
    parser.add_argument('--experiment_name', default='korean_lp_finetune', help='저장 폴더 이름')
    parser.add_argument('--saved_model', default='saved_models/korean_g2.pth', help='pretrained 모델 경로')

    # ✅ 문자셋
    parser.add_argument('--character', type=str,
        default="0123456789가강거경계고공관광교구금기김나남너노누다대더도동두등라러로루릉리마머명모목무문미바배백버보부북사산서성세소수시아악안양어여연영오외용우울원육의이인자작저전정제조종주준중지차창천청초추춘충카타태통파평포표하해허협호홀홍흥히",
        help='사용할 문자셋')

    # ✅ pretrained korean_g2.pth 모델 구조에 맞춰 수정
    parser.add_argument('--Transformation', default='None', type=str)
    parser.add_argument('--FeatureExtraction', default='VGG', type=str)
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str)
    parser.add_argument('--Prediction', default='CTC', type=str)
    parser.add_argument('--decode', type=str, default='greedy', help='디코딩 방식: greedy 또는 beamsearch')
    # ✅ 입력 이미지 설정
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--rgb', action='store_true', help='컬러 이미지 사용 여부')

    # ✅ 학습 설정
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valInterval', type=int, default=2000, help='validation 주기')
    parser.add_argument('--num_iter', type=int, default=30000, help='총 학습 스텝')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # ✅ 옵티마이저 & 학습률
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--rmsprop', action='store_true')
    parser.add_argument('--adam', action='store_true')

    # ✅ 기타 설정
    parser.add_argument('--PAD', action='store_true', help='비율 유지한 padding 적용')
    parser.add_argument('--contrast_adjust', type=float, default=0.0)
    parser.add_argument('--sensitive', action='store_true', help='대소문자 구분')
    parser.add_argument('--manualSeed', type=int, default=1111)
    parser.add_argument('--new_prediction', action='store_true', help='CTC 마지막 레이어 새로 초기화')
    parser.add_argument('--FT', action='store_true', help='feature extractor만 freezing')

    parser.add_argument('--data_filtering_off', action='store_true', help='문자셋 외의 문자는 필터링하지 않음')
    parser.add_argument('--select_data', type=str, default='/', help='LMDB 하위 전체 사용')
    parser.add_argument('--batch_ratio', type=str, default='1', help='각 서브 데이터셋의 배치 비율')
    # ✅ 옵티마이저 설정
    parser.add_argument('--optim', type=str, default='adam', help='사용할 optimizer (adam 또는 adadelta)')
    # ✅ korean_g2 모델 구조 파라미터
    parser.add_argument('--num_fiducial', type=int, default=20, help='TPS에서 사용할 fiducial points 개수')
    parser.add_argument('--input_channel', type=int, default=1, help='입력 이미지 채널 수 (흑백은 1, 컬러는 3)')
    parser.add_argument('--output_channel', type=int, default=256, help='FeatureExtractor 출력 채널 수')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden state 크기')
    parser.add_argument('--batch_max_length', type=int, default=25, help='최대 라벨 길이')
    parser.add_argument('--total_data_usage_ratio', type=float, default=1.0, help='전체 데이터 중 사용할 비율 (0~1 사이)')

    opt = parser.parse_args()
    return opt
