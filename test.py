from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/uploader', methods = ['GET','POST'])
def uploader():
    """
    This function gets the filename of the selected file, stores the image in the static directory and also displayes the image
    """
    f = request.files['file1']
    f.save(os.path.join('static', f.filename))
    filename = os.path.join('static', f.filename)
    return render_template("display.html", sudoku_image = filename)

if __name__ == '__main__':
   app.run()
