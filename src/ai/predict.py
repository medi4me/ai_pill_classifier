import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_width, img_height = 224, 224
model = load_model('pill_recognition_model.h5')

## 예측 함수
def predict_pill(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)

    # 실제 구현에서는 classes 목록에서 해당 클래스 이름을 반환 예정
    # 예시로 임시 값을 반환.
    classes = ['Class1', 'Class2', 'Class3']  # 실제 클래스 이름 리스트로 대체해야 함
    return classes[predicted_class[0]]
