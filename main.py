import telebot
bot = telebot.TeleBot('NUH UH')

from keras.models import load_model 
from PIL import Image, ImageOps
import numpy as np

def detect_bird(path):
  
    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    classs = "Class:" + class_name[2:]
    conf_score = "Confidence Score:", confidence_score
    return classs, conf_score
@bot.message_handler(content_types=['photo'])
def classif(message):
    i = message.photo[-1]
    
    file_info = bot.get_file(i.file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name, 'wb') as new_file: 
        new_file.write(downloaded_file)
    detected_bird = detect_bird(file_name)

    bot.reply_to(message, detected_bird)

bot.infinity_polling()
