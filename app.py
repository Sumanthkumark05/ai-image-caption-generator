from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)



def generate_caption(image_path):

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    output = blip_model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption



@app.route('/', methods=['GET', 'POST'])

def home():

    caption = None

    image_file = None

    if request.method == 'POST':

        file = request.files['image']

        if file:

            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(image_path)

            caption = generate_caption(image_path)

            image_file = file.filename

    return render_template(
        'index.html',
        caption=caption,
        image_file=image_file
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)