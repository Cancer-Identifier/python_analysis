from flask import Flask
from processamento_imagem import process_image  
app = Flask(__name__)

@app.route('/teste')
def index():
    result = process_image()  
    return result

if __name__ == '__main__':
    app.run(debug=True)
