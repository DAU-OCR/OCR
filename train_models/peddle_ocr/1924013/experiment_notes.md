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
**exp5**는 더 안정적인 학습 곡선과 강력한 일반화 능력을 지향

---

### 🔮 Experiment 5

> 실험 예정  
→ 위 개선점을 바탕으로 **정답률 95% 달성**을 목표로 학습 진행 예정
