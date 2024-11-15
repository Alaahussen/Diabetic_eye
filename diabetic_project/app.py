# app.py
from flask import Flask, request, render_template, redirect, url_for
from model import classifier
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Predict class
            prediction = classifier.predict(image_path)
            return render_template('index.html', prediction=prediction, image_path=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
