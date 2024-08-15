from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50

## 모델 정의
def create_model(num_classes):
    # ResNet50 모델을 베이스로 사용 (사전 학습된 가중치 사용, 최종 분류 레이어는 제거)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    model = Sequential()  # Sequential 모델 생성
    model.add(base_model)  # ResNet50 베이스 모델 추가
    model.add(Flatten())  # 2D 데이터를 1D로 변환
    model.add(Dense(256, activation='relu'))  # 완전 연결 레이어 (256개의 노드, ReLU 활성화 함수 사용)
    model.add(Dropout(0.5))  # 과적합 방지를 위한 Dropout 레이어 (50% 비율)
    model.add(Dense(num_classes, activation='softmax'))  # 클래스 수에 따른 최종 출력 레이어 (softmax 활성화 함수 사용)

    base_model.trainable = False  # ResNet50의 가중치를 고정 (사전 학습된 가중치를 그대로 사용)

    # 모델 컴파일 (Adam 최적화, 다중 클래스 분류를 위한 손실 함수 설정)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
