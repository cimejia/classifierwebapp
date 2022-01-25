from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template
#from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input, MobileNetV2
import cv2

# MODELO
img_size = 224
channels = 3
conv_base = tensorflow.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, channels))
conv_base.trainable = False

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

model = tf.keras.Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(7, activation='softmax')
    ])


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)
#model = MobileNetV2(input_shape=(224,224,3), weights=None)
#model.load_weights('static/model_weights.h5')
model.load_weights('static/best_model_tl.h5')

class UploadForm(FlaskForm):
    upload = FileField('Seleccionar una imagen:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Solamente imágenes')
    ])
    submit = SubmitField('Clasificar')


def get_prediction(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    #img_preprocessed = preprocess_input(img_resized)
    #img_reshaped = img_preprocessed.reshape((1, 224, 224, 3))
    img_reshaped = img_resized.reshape((1, 224, 224, 3))
    prediction = model.predict(img_reshaped)
    #decoded = decode_predictions(prediction)

    #top_3 = [(cat.capitalize(), round(prob*100, 2)) for (code, cat, prob) in decoded[0]][:3]
    #return top_3
    return prediction

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename
        )
        f.save(file_url)
        form = None
        prediction = get_prediction(file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)