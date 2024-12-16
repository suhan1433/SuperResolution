from flask import Flask, request, render_template, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
import io
import requests

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://user:password@db/image_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ONNX_SERVER_URL = "http://onnx:6000/inference"

# Image model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_image = db.Column(db.LargeBinary, nullable=False)
    result_image = db.Column(db.LargeBinary, nullable=True)  # Allow null for result initially


first_request_handled = False

@app.before_request
def create_tables():
    global first_request_handled
    if not first_request_handled:
        db.create_all()
        print("이 함수는 첫 번째 요청 전에만 실행됩니다.")
        first_request_handled = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    uploaded_image = file.read()
    image_entry = Image(filename=file.filename, uploaded_image=uploaded_image)
    db.session.add(image_entry)
    db.session.commit()

    try:
        response = requests.post(ONNX_SERVER_URL, files={'image': (file.filename, io.BytesIO(uploaded_image))})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "ONNX inference failed.", "details": str(e)}), 500

    result_image = response.content
    image_entry.result_image = result_image
    db.session.commit()

    return render_template('index.html', uploaded_image=image_entry.id, result_image=image_entry.id)

@app.route('/image/<int:image_id>')
def get_image(image_id):
    image = Image.query.get_or_404(image_id)
    return send_file(io.BytesIO(image.uploaded_image), mimetype='image/jpeg')

@app.route('/result/<int:image_id>')
def get_result(image_id):
    image = Image.query.get_or_404(image_id)
    if not image.result_image:
        return jsonify({"error": "Result image not found."}), 404
    return send_file(io.BytesIO(image.result_image), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
