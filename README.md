# 🫁 딥러닝 기반 폐렴 진단 모델 (Chest X-ray Pneumonia Detection using Deep Learning (ResNet50 + Grad-CAM))

📌 프로젝트 개요
주제: 흉부 X-ray 이미지 기반 폐렴 진단
목표: 높은 일반화 성능을 갖춘 모델 개발 (Test 정확도 90% 이상)

📌 데이터셋
데이터 출처: Kaggle Chest X-ray Pneumonia Dataset
구성: Train: 5216장, Test: 624장, Validation: 16장 (Validation이 부족하기 때문에 Train에서 10%를 따로 Validation으로 구성)

📌 모델 구축, 튜닝 및 연구

| 버전 | 주요 특징 | 결과 요약 |
|:----|:---------|:--------|
| **Baseline CNN** | - 간단한 Conv2D + MaxPooling 기반 CNN 모델<br>- EarlyStopping, ModelCheckpoint 사용 | - Train 데이터에 과적합<br>- Pneumonia는 잘 맞추나 Normal 분류 성능 낮음 |
| **Transfer Learning (ResNet50)** | - ImageNet으로 사전 학습된 ResNet50 모델 사용<br>- 전체 레이어 Freeze 후, Output Head만 학습 | - Pneumonia는 모두 맞췄지만<br>- Normal 클래스 완전히 분류 실패 |
| **Fine-Tuning v1** | - ResNet50 상위 50개 레이어만 학습 가능<br>- Dropout 0.3 추가 | - 약간 개선<br>- 여전히 Normal 분류가 어려움 |
| **Fine-Tuning v2** | - 상위 100개 레이어 학습 시도<br>- Learning Rate 축소 (1e-6) | - Normal 분류 성능이 오히려 하락 |
| **Fine-Tuning v3** | - 데이터 증강(Data Augmentation) 강화<br>- Rotation, Shift, Brightness 조정 | - 여전히 Normal 분류 성능의 개선이 없음 |
| **Fine-Tuning v4** | - Dropout 비율 증가 (0.5)<br>- 조기 종료(EarlyStopping) 조건 완화 | - 해당 방법들로는 Normal 분류 성능 개선의 여지가 없음을 인지|
| **Fine-Tuning v5** | - 클래스 가중치(class_weight) 적용<br>- Pneumonia와 Normal 비율 보정 | - Normal 클래스 정확도 큰 폭 향상<br>- 테스트 정확도 84% 이상 도달 |
| **Fine-Tuning v6 (최종)** | - Label Smoothing 추가<br>- ReduceLROnPlateau 적용<br>- 데이터 증강 최적화 | - Test 정확도 **90% 돌파**<br>- 모델 일반화 성능 최상 |

---
**Reaserch Summary**
- **Baseline → Transfer Learning → Fine-Tuning → Regularization 강화 → 클래스 가중치 조정** 흐름으로 점진적 성능 개선
- **Fine-Tuning v6**가 가장 좋은 결과 (Test Accuracy ≈ 90.7%)

- 최종 모델 (Fine-Tuning v6)
	•	Base Model: ResNet50 (ImageNet 사전학습)
	•	Fine-Tuning: 하위 레이어 Freeze, 상위 50개 레이어만 학습
	•	Optimizer: Adam (Learning Rate 1e-4)
	•	Loss: BinaryCrossentropy(Label Smoothing=0.1)
	•	Data Augmentation: Rotation, Shift, Zoom, Brightness Change
	•	Regularization: Dropout 0.5
	•	Scheduler: ReduceLROnPlateau 사용 (EarlyStopping 병행)

	•	Test Accuracy: 약 90.7%
	•	ROC Curve: AUC(Area Under Curve) 0.957





흉부 X-ray 이미지를 활용해 폐렴 여부를 자동으로 분류하는 딥러닝 기반 이진 분류 모델을 개발했습니다.

기존의 pretrained ResNet50을 활용하여, Transfer Learning, fine-tuning을 통해 Accuracy를 높였으며, Grad-CAM을 통해 모델의 예측 결과를 시각적으로 해석할 수 있도록 했습니다.

* 기본적인 pretrained ResNet50은 가중치가 Imagenet으로 설정되어 있음, 이는 사물 분류용으로 훈련된 CNN (강아지, 고양이, 차 등 1000개 클래스를 분류) -> Transfer Learning을 통해 의료 이미지에 더욱 적합하도록 설정.

<img src="https://github.com/user-attachments/assets/8df53b3d-681f-4f92-8538-eaad42521c9f" width="500" height="500">

![fef685ae-09c3-4e8e-8bc4-67a4160b46be](https://github.com/user-attachments/assets/1155b95b-e52a-4b88-91e5-9da9b869a4e4)
![026ce00f-3821-429a-a4da-44cfba0197e7](https://github.com/user-attachments/assets/6acd185c-9380-4e72-8f86-cbdb38eef6e8)




# 💡Insight

* Baseline CNN 모델의 Accuracy: 0.74 (학습시간 약 8분, 4epoch, early stop)
* Transfer Learning 모델의 Accuracy: 0.62 (학습시간 약 30분, 10epoch)
* Fine-tuned 모델의 Accuracy: 0.90 (학습시간 약 100분, 20epoch)
  

* EarlyStopping, CallBack, ModelCheckpoint (best model saving) 을 적절히 활용하자. - 학습 시간 단축
* 모델에 따라 적절한 fine-tuning 기법을 파악하고, 이를 활용하여 Accuracy를 높이자.
* CNN에서는 상위 레이어의 trainable을 조정 (True)하여 훈련하자. -> 상위 레이어는 출력층에 가깝고, 클래스 특화된 특징을 감지하며, 이는 도메인 특화에 최적화가 될 수 있다. (하위 레이어는 입력층에 가깝고, 기본적인 시각적 패턴을 감지한다)
* Batch Size, Epoch만 조정하는 것은 학습 효율 튜닝일 뿐, 신경망의 가중치를 조정하는 fine-tuning이 아님
* fine-tuning한 모델의 confusion matrix와 classification report를 보면서 test 데이터에 대해 맞추지 못하는 부분을 캐치하고, 해당 부분에 대해 클래스 가중치를 부여하여 더욱 잘 맞출 수 있도록 조정


* 서로 다른 모델을 비교하기 위해서는 같은 데이터셋을 같은 조건에서 학습해야 함
* train(train_generator) -> 학습 시 사용하는 학습용
* validation(val_generator) -> 학습 중 모니터링용 (early stopping, best weight 저장), 학습에 영향을 받았으므로 성능 측정용으로 부적합
* test(test_generator) -> 최종 성능 평가용 (처음 보는 데이터를 모델에 적용하여 성능 검증 가능)

* 경량 모델을 활용하여 fine-tuning을 시도하는 방법도 좋을 것 같다.
* 앙상블 기법을 적용하면 추가 성능 향상을 기대할 수 있다.


* 단순히 사전 학습된 모델을 사용한다고 baseline 모델보다 반드시 좋은 성과를 가져오는 것은 아니다. (도메인 특화된 모델이 되도록 Fine-tuning할 필요가 있다.)
* 어떻게 하면 모델의 정확도를 높일 수 있을 지 고민하며 다양한 방법들의 튜닝들을 시도했고, 연구의 흥미를 느끼게 되었다.
