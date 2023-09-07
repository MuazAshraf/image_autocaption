from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import piexif

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def set_caption(image_path, caption):
    # Open the image
    image = Image.open(image_path)

    # Check if the image has EXIF data
    if "exif" in image.info:
        exif_dict = piexif.load(image.info["exif"])
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    # Add the caption to the 'ImageDescription' field
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = caption.encode("utf-8")

    # Convert the Exif data back to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save the image with the updated Exif data
    image.save(image_path, exif=exif_bytes)
@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    print(request.files.keys())
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    file = request.files.get('image')
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image and generate caption
        raw_image = Image.open(file_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Store the caption in the image's metadata
        set_caption(file_path, caption)

        return jsonify({
        "caption": caption,
        "message": "Caption has been stored in the image's metadata."
        })


    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

