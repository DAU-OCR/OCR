## 실험 결과 요약

---

### 🔬 Experiment 2 (epoch 60)

- **Train Accuracy**: 0.015 이상 도달하지 못함 → 학습 정체
- **Train Norm Edit Distance**: 약 0.3에서 평탄화 → 큰 진전 없음
- **Train Loss**: 꾸준히 감소 중 → 학습 자체는 진행됨
- **Learning Rate**: StepDecay 정상 작동

**결론**:  
학습이 진행되었지만 성능 정체로 인해 조기 중단

**개선점**:  
- Backbone 변경: `MobileNetV3 → ResNet`

---

### 🚀 Experiment 3

- **Train Accuracy**: 직선형 상승 → 학습 잘됨
- **Train Norm Edit Distance**: 0.9 이상 → 예측 문자 매우 유사
- **Train Loss**: 초기 급감 → 이후 완만한 감소
- **Learning Rate**: Warmup → 유지 → Decay 정상 적용

**결론**:  
학습 성공적으로 완료  
**최종 정확도**: 0.84

**개선점**:
- Backbone: `ResNet18 → ResNet34`
- Neck hidden size: `128 → 256`
- Head FC decay: `1e-5 → 1e-4`
- Learning Scheduler: `LinearWarmupCosine → OneCycle`
- Warmup steps: `3125 → 5000`
- Batch size per card: `128 → 192`
- AMP(float16) 사용

**요약**:  
모델 표현력, 학습 스케줄링, GPU 활용 극대화를 통한 성능 향상

---

### ⚠️ Experiment 4

- **Step 19490**: acc = 0.851, edit_dis = 0.963 → 매우 양호
- **Step 19500**: acc = 0.848, edit_dis = 0.962 → 안정적 유지
- **Eval 이후 (Step 19500)**:  
  `eval/acc = 0.797`, `edit_dis = 0.947` → 급격한 하락

- **Best Accuracy**: 0.803 (epoch 46)에서 더 이상 개선되지 않음

**결론**:  
**과적합 발생** → 학습 중단

**개선점**:
| 항목 | 기존 | 변경 |
|------|------|------|
| epoch_num | 100 | 120 |
| optimizer.lr.name | OneCycle | LinearWarmupCosine |
| warmup_steps | 5000 | 3000 |
| regularizer.factor | 0.01 | 1e-5 |
| batch_size_per_card | 192 | 128 |
| transforms | RecAug 단독 | RandAugment, RandomRotation, RandomNoise 추가 |
| fc_decay (CTCHead) | 0.0001 | 1e-5 |

**요약**:  
일반화 능력 향상과 과적합 방지를 목표로 세팅 조정  
'exp5'는 더 안정적인 학습 곡선과 강력한 일반화 능력을 지향

---

### 🧪 Experiment 5 (ResNet18 + 안정화 중심 설정)

- **Train Accuracy**: 약 0.81 전후에서 정체
- **Train Norm Edit Distance**: 0.94 이상으로 유지되나 더 이상 개선 없음
- **Eval Accuracy**: 최대 0.798 수준 → 이후 하락
- **Train Loss**: 학습 후반부에서 더디게 감소
- **Learning Rate**: LinearWarmupCosine 적용 → 의도한 대로 작동
- **Epoch**: 총 120으로 설정했으나, 성능 개선 없음 판단으로 중단

**결론**:  
성능 정체로 인해 조기 중단  
학습 진행 자체는 안정적이나 성능 한계가 뚜렷하게 나타남

**개선점** 
| 항목 | Exp5 | Exp6 |
|------|------|------|
| warmup_steps | 3000 | **2000** |
| batch_size_per_card | 128 | **192** |
| checkpoints | 사용함 (latest.pdparams) | **사용 안함 (null)** |
| 초기화 방식 | 이어서 학습 | **완전 초기화 학습** |
| 목표 | 안정성 중시 | **빠른 수렴 + 일반화 강화** |

**요약**:  
'exp5'는 학습 곡선을 안정적으로 유지했으나, 정확도 및 일반화 개선이 한계에 도달. 성능 향상을 위해 더 강한 학습률 제어 및 batch 설정 조정 필요

---

### 📈 Experiment 6

- **Train Accuracy**: 약 0.80 도달
- **Eval Accuracy (eval/best_acc)**: 0.6654
- **Train-Eval 차이**: 약 13%p 이상 → 일반화 실패
- **Norm Edit Distance**: 7500 step 이후 거의 변화 없음
- **Best Accuracy 정체**: 7500 ~ 11500 step 구간에서 완전 정체

**결론**:  
학습 자체는 안정적이었으나, **Train과 Eval 간 성능 괴리**가 심화  
→ 일반화 실패로 인해 추가적인 구조적 개선 필요

**⚠️ 문제점 요약**:

| 문제 | 설명 |
|------|------|
| 과적합 가능성 | train acc는 꾸준히 상승했으나, eval acc는 거의 향상되지 않음 |
| 데이터 다양성 부족 | 두 줄 번호판 제거로 난이도는 낮아졌지만, 단조로운 학습으로 이어짐 |
| 증강 부족 | RecAug 단독 사용으로 다양한 환경 대응이 어려움 |
| 모델 용량 제한 | hidden_size=256, SE=False로 표현력 한계 존재 가능 |
| scheduler 조기 정체 | warmup 이후 학습률이 너무 일찍 감소했을 가능성 (warmup=2000) |

**개선점** (Exp6 → Exp7 비교):

| 항목 | Exp6 | Exp7 |
|------|------|------|
| warmup_steps | 2000 | **1000** (더 빠른 수렴 유도) |
| min_lr | 1e-6 | **1e-7** (더 낮은 학습률 유지로 세밀한 수렴) |
| regularizer.factor | 1e-5 | **1e-4** (과적합 억제 강화) |
| batch_size_per_card | 192 | **128** (일반화 향상 목적) |
| Backbone.disable_se | true | **false** (SE 활성화로 표현력 향상) |
| Neck.hidden_size | 256 | **384** (모델 용량 증가) |
| Head.fc_decay | 1e-5 | **1e-4** |
| 데이터 증강 | RecAug 단독 | **RecAug + RandAugment** (num_ops=2, magnitude=9) |

**요약**:  
`exp6`는 학습 안정화에는 성공했지만, 일반화 성능에서 한계가 드러남.  
`exp7`에서는 **모델 구조 확장** , **증강 전략 강화** 사용 성능 도약 목표

---

### 🧪 Experiment 7

- **목표**:  
  `exp6`에서 확인된 일반화 성능 한계를 극복하고자  
  **데이터 증강 다양화 + 모델 용량 증가**를 동시에 적용
---

### ⚠️ 학습 중 문제 발생

- 이미지 증강 다양화를 적용한 초기 설정에서 **값 변환 관련 error**가 반복적으로 발생
- 디버깅 도중 증강 파이프라인의 연산 충돌 가능성 제기 → **RandAugment 제거 후 학습 지속**
- 이후에도 학습 중 다음과 같은 **구조적 오류** 확인됨:

| 문제 | 증상 |
|------|------|
| label 무효화 | `label`이 loss 계산에 반영되지 않음 |
| loss 정체 | 학습 시작 이후 loss가 전혀 감소하지 않음 |
| acc 정체 | train/eval accuracy 모두 0에서 시작하여 그대로 유지됨 |
| norm_edit_distance 정체 | 일정 step 이후 0.5 수준에서 더 이상 상승 없음 |

→ **모델 내 label encoding 또는 학습 라벨 전처리 오류 가능성** 발생

---

### 🔍 Exp8로의 전환

- 위 문제 해결을 위해 `exp8`에서는 새로운 증강을 도입하지 않고,  
  **label → encoded → decode 과정의 무결성 검증**만을 위한 코드 삽입  
- 학습은 일시 보류하고, 오류 원인 분석을 목적으로 한 실험으로 설계됨

---

**요약**:  
`exp7`은 구조적 개선을 목표로 다양한 세팅을 적용했으나,  
학습 자체가 정상적으로 이루어지지 않는 **치명적인 label 처리 문제** 발생  
→ 이후 실험(`exp8`)에서는 구조 검증을 통해 원인 추적을 우선시함

---

### 🧪 Experiment 8

- **목표**:  
  `exp7`에서 발생한 loss 및 accuracy 정체 문제의 원인을 추적하기 위해  
  **label → encoded → decode** 전처리 전반에 대한 오류 검증 실시

---

### 🧪 점검 내용

- 학습에 사용되는 전체 train label에 대해 **CTCLabelEncode → CTCLabelDecode** 순환 테스트 진행
- 한글자 단위로 정확히 매칭되는지, 길이(length) 값이 의도대로 들어가는지 확인
- 예외/에러/공백 포함 여부까지 검사 완료

→ **결과: 모든 인코딩 및 디코딩 정상 확인**

---

### ⚠️ 문제 해결 실패

- 라벨 처리 문제는 아닌 것으로 판단되었으나, 여전히 학습 정체 문제 해결 불가
- 오류 원인을 찾지 못한 채 다음과 같은 조치를 진행:

> ✅ **TrainingImg를 제외한 나머지 학습 관련 리소스 파일을 전부 클린(clean) 상태로 교체**

- label 파일, eval 이미지, 설정 파일 등을 재작성 및 정비하여  
  학습 환경의 문제 가능성을 배제하고자 함

---

### 🔁 다음 실험: `exp9`

- 기존과 동일한 구조로 학습을 재진행하여,  
  문제 원인이 **데이터 or 학습 환경 자체였는지 검증**하는 실험
- 증강 및 네트워크 설정은 `exp7` 이전 수준으로 복귀시켜 **최소 조건 검증** 목표
---

**요약**:  
`exp8`은 구조적 오류를 확인하기 위한 라벨 검증 실험으로, 인코딩 로직은 문제 없음을 확인  
그러나 학습 실패의 근본 원인을 파악하지 못해, `exp9`에서 **전체 환경 리셋 및 재검증**으로 전환

---

---

### 📈 Experiment 9

- **Train Accuracy**: 약 0.92까지 빠르게 도달  
- **Eval Accuracy (eval/best_acc)**: 최종 **0.84**
- **Norm Edit Distance**: 약 **0.96**까지 상승  
- **Loss**: 빠르게 하락 후 일정 구간부터 정체
- **Best Accuracy 수렴 시점**: step 17000 ~ 21000 구간

**결론**:  
데이터 정비 및 환경 초기화를 통해 **정상적인 학습곡선 복구**에 성공  
초기 학습 속도나 정확도 상승 폭은 **exp3 대비 더 가팔랐으나**,  
최종적으로는 **exp3과 유사한 수준에서 수렴** → 모델 표현력 한계 가능성 제기

---

### 📊 비교: Exp3 vs Exp9

| 항목 | Exp3 | Exp9 |
|------|------|------|
| max acc | 0.84 | 0.84 |
| norm_edit_distance | 약 0.94 | 약 0.96 |
| loss 곡선 | 안정적인 감소 | 더 빠른 초기 감소 후 정체 |
| 수렴속도 | 점진적 상승 | 빠른 상승 후 수렴 |
| 데이터 | 8만장 전체 | 약 5.5만장 (단일행 번호판만) |

---

**개선점** (Exp9 → Exp10 비교):

| 항목 | Exp9 | Exp10 |
|------|------|-------|
| hidden_size | 256 | **384** |
| disable_se | True | **False** |
| fc_decay | 0.0001 | 0.0001 (동일) |
| warmup_steps | 5000 | **2500** |
| batch_size_per_card | 192 | 192 (동일) |
| aug_prob | 0.7 | **1.0** |
| use_tia | True | **False** |
| regularizer.factor | 0.01 | 0.01 (동일) |
| scheduler | OneCycle | OneCycle (동일) |
| 전체 컨셉 | 안정적인 학습 복구 | **일반화 성능 극대화 + 표현력 확장** |

**요약**:  
`exp10`은 `exp9`에서 확보된 안정된 학습 기반 위에,  
모델의 **표현력 확장 (hidden_size, SE 구조 활성화)** 및  
**강화된 증강 적용 (TIA 제거 + 100% 증강 적용)**을 통해  
**성능 상한 돌파**를 노린 실험이다.

---

### 🚀 Experiment 10

- **Train Accuracy**: 약 0.93까지 빠르게 도달  
- **Eval Accuracy (eval/best_acc)**: 최종 **0.84**  
- **Norm Edit Distance**: 약 0.96 수준 도달  
- **Loss**: 빠르게 감소하였으나, 일정 구간 이후 정체  
- **수렴 성향**: `exp9`보다 더 빠르게 상승했으나, **결국 비슷한 수준에서 수렴**

**결론**:  
모델 용량 증가(`hidden_size=384`, `SE=False`) 및 증강 확장(`aug_prob=1.0`, `TIA 제거`)을 통해  
학습 안정성은 유지되었으나, **최종 성능 상한은 여전히 벽 존재**  
→ 더 근본적인 구조 확장을 통한 성능 극대화 필요

---

### 🛠 개선점 (Exp10 → Exp11)

| 항목 | Exp10 | Exp11 |
|------|--------|--------|
| hidden_size | 384 | **512** |
| backbone.layers | ResNet18 | **ResNet34** |
| disable_se | False | **True** |
| regularizer.factor | 0.01 | **0.001** |
| warmup_steps | 2500 | **1000** |
| batch_size_per_card | 192 | **160** |
| use_tia | False | **True (tia_prob=0.8)** |
| aug_prob | 1.0 | 1.0 (동일) |
| 추가 증강 | 없음 | ✅ **blur, hsv, jitter, crop, reverse 등 복합 RecAug 적용**

**요약**:  
`exp10`에서 수렴 성능 한계가 명확해짐에 따라,  
`exp11`은 모델 용량을 더욱 키우고 (ResNet34 + hidden 512),  
다양한 이미지 증강을 조합하여 **최대의 best_acc 달성**을 목표로 설정됨

// exp11_1 : Rec Aug 수치 대폭 하향




