from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('models/cat_dog_cnn.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

loss, accuracy = model.evaluate(test_set)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
