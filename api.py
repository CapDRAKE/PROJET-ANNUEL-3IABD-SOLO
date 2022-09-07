from mlp_np import *
import urllib.request
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = './upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mdl =loadModel("bestmdl.pkl")

#upload the image
@app.route('/',methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        files = request.files.getlist('files[]')
        errors = {}
        success = False
        images=[]
        for file in files:
            filename = secure_filename(file.filename)
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            images.append(path)
            success = True

        X = preprocess_images(images) 
        predictions = mdl.predict(X,True)
        results =[ [labels[i],predictions[1][index][i]] for index,i in enumerate(np.argmax(predictions[1], axis=1)) ]
        return jsonify(results)
    return '''
    <h1>Projet annuel sans CSS oups...</h1>
    <form method="post" enctype="multipart/form-data">
    <input type="file" id="multiFiles" name="files[]" multiple="multiple">
    <input type="submit">
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)


#Matrice de confusion