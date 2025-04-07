import os
from flask import Flask, render_template, request
from predict import predict_tumor
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded!'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file!'
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)

    # Get prediction
    predicted_class, confidence, message = predict_tumor(file_path)

    return render_template('result.html',
                           image_path=file_path,
                           prediction=predicted_class,
                           confidence=confidence,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)
