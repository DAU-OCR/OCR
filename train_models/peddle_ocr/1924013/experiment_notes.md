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