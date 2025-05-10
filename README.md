# 🫁 딥러닝 기반 폐렴 진단 모델 (Chest X-ray Pneumonia Detection using Deep Learning (ResNet50 + Grad-CAM))

📌 프로젝트 개요

흉부 X-ray 이미지를 활용해 폐렴 여부를 자동으로 분류하는 딥러닝 기반 이진 분류 모델을 개발했습니다.

기존의 pretrained ResNet50을 활용하여, Transfer Learning ResNet50, fine-tuned ResNet50을 통해 Accuracy를 높였으며, Grad-CAM을 통해 모델의 예측 결과를 시각적으로 해석할 수 있도록 했습니다.

* 기본적인 pretrained ResNet50은 가중치가 Imagenet으로 설정되어 있음, 이는 사물 분류용으로 훈련된 CNN (강아지, 고양이, 차 등 1000개 클래스를 분류) -> Transfer Learning을 통해 의료 이미지에 더욱 적합하도록 설정.

<img src="https://github.com/user-attachments/assets/8df53b3d-681f-4f92-8538-eaad42521c9f" width="500" height="500">

![fef685ae-09c3-4e8e-8bc4-67a4160b46be](https://github.com/user-attachments/assets/1155b95b-e52a-4b88-91e5-9da9b869a4e4)
![026ce00f-3821-429a-a4da-44cfba0197e7](https://github.com/user-attachments/assets/6acd185c-9380-4e72-8f86-cbdb38eef6e8)




# 💡Insight

* Transfer Learning 모델과 Fine-tuned 모델의 Accuracy: 0.72 -> 0.95
* Transfer Learning 모델과 Fine-tuned 모델의 학습 시간: 약 22분 (8epoch, earlystop) -> 약 35분 (7epoch, earlystop)
* EarlyStopping, CallBack, ModelCheckpoint (best model saving) 을 적절히 활용하자. - 학습 시간 단축
* 모델에 따라 적절한 fine-tuning 기법을 파악하고, 이를 활용하여 Accuracy를 높이자.
* CNN에서는 상위 레이어의 trainable을 조정 (True)하여 훈련하자. -> 상위 레이어는 출력층에 가깝고, 클래스 특화된 특징을 감지하며, 이는 도메인 특화에 최적화가 될 수 있다. (하위 레이어는 입력층에 가깝고, 기본적인 시각적 패턴을 감지한다)
* Batch Size, Epoch만 조정하는 것은 학습 효율 튜닝일 뿐, 신경망의 가중치를 조정하는 fine-tuning이 아님


* validation(val) -> 학습 중 모니터링용 (early stopping, best weight 저장), 학습에 영향을 받았으므로 성능 측정용으로 부적합
* test(test) -> 최종 성능 평가용 (처음 보는 데이터를 모델에 적용하여 성능 검증 가능)
