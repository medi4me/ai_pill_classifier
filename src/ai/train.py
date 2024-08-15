from model import create_model
from data_loader import load_data

## 모델 학습
# 데이터 로드
dataset_dir = 'path_to_dataset_directory'  # 데이터셋 디렉토리 경로
X_train, X_val, X_test, y_train, y_val, y_test, classes = load_data(dataset_dir)

# 모델 생성
num_classes = len(classes)  # 클래스 수 계산
model = create_model(num_classes)  # 모델 생성

# 모델 학습
history = model.fit(
    train_generator,  # 학습 데이터 제너레이터
    steps_per_epoch=len(X_train) // 32,  # 에포크당 수행할 스텝 수
    epochs=10,  # 학습할 에포크 수
    validation_data=val_generator,  # 검증 데이터 제너레이터
    validation_steps=len(X_val) // 32  # 검증할 때 수행할 스텝 수
)

# 학습된 모델 저장
model.save('pill_recognition_model.h5')  # 모델 파일 저장
