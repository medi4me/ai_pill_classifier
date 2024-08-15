import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img


## 데이터 로드 및 분할
img_width, img_height = 224, 224  # 이미지의 크기 설정


def load_data(dataset_dir):
    images = []  # 이미지 데이터를 저장할 리스트
    labels = []  # 라벨 데이터를 저장할 리스트
    classes = os.listdir(dataset_dir)  # 각 클래스(라벨) 폴더의 이름을 가져옴

    # 각 클래스 폴더의 이미지를 읽어서 리스트에 추가
    for label, class_dir in enumerate(classes):
        class_path = os.path.join(dataset_dir, class_dir)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path, target_size=(img_width, img_height))  # 이미지 로드 및 크기 조정
            img = img_to_array(img)  # 이미지를 numpy 배열로 변환
            images.append(img)  # 이미지 리스트에 추가
            labels.append(label)  # 라벨 리스트에 추가

    images = np.array(images)  # 리스트를 numpy 배열로 변환
    labels = np.array(labels)  # 리스트를 numpy 배열로 변환

    # 학습, 검증, 테스트 세트로 분할
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, classes  # 분할된 데이터와 클래스 이름 반환
