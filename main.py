from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load your trained model here
model = tf.keras.models.load_model('cifar10_model_85acc.h5')

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

app = FastAPI()

# Function to preprocess and predict image
def predict_image(image_file):
    img = image.load_img(image_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    predicted_class = predict_image("temp.jpg")
    return JSONResponse(content={"prediction": predicted_class})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
