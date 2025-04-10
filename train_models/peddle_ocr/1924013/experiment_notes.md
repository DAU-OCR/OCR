exp2(epoch 60)

    train/acc: 0.015 이상은 못 넘음 → 학습이 멈춘 상태

    train/norm_edit_dis: 0.3 근처에서 플랫 → 비슷한 추세

    train/loss: 꾸준히 감소함 (학습은 되고 있음)

    train/lr: StepDecay 잘 적용되고 있음

    결과 
        학습 중지

    개선점
        Backbone 교체 (MobileNetV3 → ResNet)

exp3
    train/acc: 거의 직선에 가까운 상승 → 학습 잘됨

    train/norm_edit_dis: 0.9 이상 도달 → 예측 문자 거의 유사

    train/loss: 처음 급하강 후 서서히 감소 → 이상적

    train/lr: warmup → 유지 → decay 순서로 정상 작동 확인됨

    결과
        학습 완료 
        최종 정확도 : 0.84

    개선점
        ResNet18 -> ResNet34
        
        Neck.hidden_size
            기존: 128
            변경: 256

        Head.fc_decay
            기존: 0.00001
            변경: 0.0001   

        lr.name
            기존: LinearWarmupCosine
            변경: OneCycle  

        lr.warmup_steps
            기존: 3125
            변경: 5000

        Train.loader.batch_size_per_card
            기존: 128
            변경: 192

        AMP (float16) 사용
        
    모델 표현력 향상(RNN hidden size 증가, fc_decay 조정), 학습 스케줄러 개선(OneCycle 적용), GPU 활용 극대화(batch_size 증가)를 통해 성능 향상을 목표

exp4
    step 19490	acc = 0.851, edit_dis = 0.963 (정상)

    step 19500	acc = 0.848, edit_dis = 0.962 (살짝 감소, 학습은 안정적)

    step 19500 → eval 직후	eval/acc = 0.797, edit_dis = 0.947 ← 급하락

    best_acc	그대로 유지됨 (0.803 from epoch 46)

    결과
        과적합 발생
        학습 중지

    개선점
        epoch_num	100	120	더 긴 학습 시간으로 수렴을 충분히 유도하고, 정답률 95% 도달 가능성 향상

        optimizer.lr.name	OneCycle	LinearWarmupCosine	OneCycle은 초반에 빠르게 학습하지만 불안정할 수 있음. LinearWarmupCosine은 안정적이며 과적합 방지에도 유리

        warmup_steps	5000	3000	이전 warmup이 너무 길어 초반 학습이 느렸음. 3000으로 줄여 더 빠르게 유효한 학습 시작 유도

        regularizer.factor	0.01	1e-5	과적합 방지 L2 정규화 강도를 약하게 조정해 모델 표현력을 살림

        Train.loader.batch_size_per_card	192	128	너무 큰 배치로 메모리 압박 및 일반화 어려움 → 줄이면 성능 안정화 및 regularization 효과

        Train.transforms	RecAug 단독	RandAugment, RandomRotation, RandomNoise 추가	데이터 다양성 증가로 과적합 방지 및 일반화 능력 강화

        fc_decay (CTCHead)	0.0001	1e-5	지나치게 큰 FC decay가 성능 저하 유발 가능 → 감소시켜 유연한 학습 유도
    
    exp5는 더 안정적인 학습 곡선과 더 강력한 일반화 능력을 목표로 함

exp5
