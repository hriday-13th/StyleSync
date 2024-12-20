from flask import Flask, render_template, request, session, Response, redirect, url_for
from body_shape_classifier import classify_body_shape
from face_shape_classifier import classify_face_shape
from recommender import similar
from try_on import generate_frames
import os
import mediapipe as mp

app = Flask(__name__)
app.secret_key = '0987654321'

images = {
        "body": {
            "Men": {
                "Topwear": [
                    'images/15970.jpg',
                    'images/1855.jpg',
                    'images/12369.jpg',
                    'images/7990.jpg',
                    'images/15984.jpg'
                ],
                "Bottomwear": [
                    'images/39386.jpg',
                    'images/21379.jpg',
                    'images/27625.jpg',
                    'images/10257.jpg',
                    'images/11349.jpg'
                ]
            },
            "Women": {
                "Topwear": [
                    'images/3953.jpg',
                    'images/33019.jpg',
                    'images/58514.jpg',
                    'images/25527.jpg',
                    'images/7160.jpg'
                ],
                "Bottomwear": [
                    'images/51499.jpg',
                    'images/49654.jpg',
                    'images/32590.jpg',
                    'images/32564.jpg',
                    'images/28458.jpg'
                ]
            }
        },
        "face": {
            "Oblong": [
                'glasses/Oblong/glass(1).jpg',
                'glasses/Oblong/glass(2).jpg',
                'glasses/Oblong/glass(3).jpg',
                'glasses/Oblong/glass(4).jpg',
                'glasses/Oblong/glass(5).jpg',
                'glasses/Oblong/glass(6).jpg',
                'glasses/Oblong/glass(7).jpg'
            ],
            "Diamond": [
                'glasses/Diamond/glass(1).jpg',
                'glasses/Diamond/glass(2).jpg',
                'glasses/Diamond/glass(3).jpg',
                'glasses/Diamond/glass(4).jpg',
                'glasses/Diamond/glass(5).jpg',
                'glasses/Diamond/glass(6).jpg',
                'glasses/Diamond/glass(7).jpg'
            ],
            "Square": [
                'glasses/Square/glass(1).jpg',
                'glasses/Square/glass(2).jpg',
                'glasses/Square/glass(3).jpg',
                'glasses/Square/glass(4).jpg',
                'glasses/Square/glass(5).jpg',
                'glasses/Square/glass(6).jpg',
                'glasses/Square/glass(7).jpg'
            ],
            "Round": [
                'glasses/Round/glass(1).jpg',
                'glasses/Round/glass(2).jpg',
                'glasses/Round/glass(3).jpg',
                'glasses/Round/glass(4).jpg',
                'glasses/Round/glass(5).jpg',
                'glasses/Round/glass(6).jpg',
                'glasses/Round/glass(7).jpg'
            ]
        }
    }


@app.route("/login")
def login():
    # To be done
    return None

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/face_shape', methods=['GET', "POST"])
def face_shape():
    return render_template('face_shape.html')

@app.route('/body_shape', methods=['GET', "POST"])
def body_shape():
    return render_template('body_shape.html')

@app.route('/virtualtryon', methods=['GET', 'POST'])
def virtual_tryon():
    result = session.get('result')
    imgs = session.get('imgs')

    if 'num' not in session:
        session['num'] = 1

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'increment':
            session['num'] = min(session['num'] + 1, 15)
        elif action == 'decrement':
            session['num'] = max(session['num'] - 1, 1)

        return redirect(url_for('virtual_tryon'))

    return render_template('face_recommend.html', show_feed=True, result=result, image_filenames=imgs, num=session['num'])

@app.route('/video_feed')
def video_feed():
    result = session.get('result')
    num = session.get('num')
    return Response(generate_frames(result, num), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/glasses_recommendations", methods=['POST'])
def face():
    if "image" not in request.files:
        return "No image uploaded", 400
    
    image = request.files["image"]

    if image.filename == "":
        return "No selected file", 400
    
    upload_folder = 'uploads/'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image.filename)
    image.save(image_path)

    if image:
        result = classify_face_shape(image_path)
        imgs = images["face"][result]

        session['result'] = result
        session['imgs'] = imgs

        return render_template("face_recommend.html", result = result, image_filenames = imgs, show_feed=False)

@app.route("/fashion_recommendations", methods=['POST'])
def body():
    if "image" not in request.files:
        return "No image uploaded", 400
    
    image = request.files["image"]
    gender = request.form.get("gender")
    subCategory = request.form.get("subCategory")
    color = request.form.get("color")
    season = request.form.get("season")

    if image.filename == "":
        return "No selected file", 400
    
    upload_folder = 'uploads/'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image.filename)
    image.save(image_path)

    if image and gender:
        mp_image = mp.Image.create_from_file(image_path)
        result = classify_body_shape(image_path, mp_image, gender)
        imgs = images["body"][gender][subCategory]

        session['gender'] = gender
        session['result'] = result
        session['subCategory'] = subCategory
        session['color'] = color
        session['season'] = season
        session['imgs'] = imgs

        return render_template("recommend.html", gender = gender, result = result, image_filenames = imgs)

@app.route('/rec', methods=['POST'])
def plot_image_route():
    gender = session.get('gender')
    result = session.get('result')
    subCategory = session.get('subCategory')
    color = session.get('color')
    season = session.get('season')
    imgs = session.get('imgs')
    image_index = int(request.form.get('image_index'))
    image_paths = similar(image_index, gender, subCategory, color, season)
    return render_template("recommend.html", gender = gender, result = result, image_filenames = imgs, image_paths=image_paths, message=f"Plotted Image {image_index}")

if __name__ == "__main__":
    app.run(debug = True)