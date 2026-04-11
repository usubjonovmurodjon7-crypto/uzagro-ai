import tensorflow as tf
import numpy as np
import cv2

print("UzAgroAI rasm tekshirish boshlandi")

# modelni yuklash
model = tf.keras.models.load_model("../models/uzagro_ai_model.h5")

# rasmni o‘qish
img = cv2.imread("test.jpg")

# o‘lchamni moslash
img = cv2.resize(img, (224,224))

# normalizatsiya
img = img / 255.0
img = np.expand_dims(img, axis=0)

# prediction
prediction = model.predict(img)

# eng katta ehtimollikni olish
class_index = np.argmax(prediction)

# natijani o‘zbek tilida chiqarish
if class_index == 0:
    print("Natija: Pomidor sog'lom")
else:
    print("Natija: Pomidor kasallangan")