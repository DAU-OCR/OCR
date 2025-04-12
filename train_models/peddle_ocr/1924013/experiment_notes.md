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