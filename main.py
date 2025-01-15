from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'chave_secreta_para_flash_messages'

# Carregar o modelo treinado
model = load_model('skin_cancer_model.keras')

# Definir as classes 
classes = ['Carcinoma Basocelular', 'Dermatofibroma', 'Nevo Melanocítico', 'Melanoma', 'Lesões Vasculares', 
           'Queratose Actínica', 'Queratose Benigna', 'Carcinoma Intraepitelial', 'Desconhecido']

# Diretório para armazenar uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('envio.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('Nenhuma imagem foi selecionada!', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('Nenhuma imagem foi selecionada!', 'danger')
        return redirect(url_for('index'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Processar e prever a imagem
            img = image.load_img(file_path, target_size=(75, 100))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            
            result = classes[predicted_class[0]]
            return render_template('resultado.html', result=result)
        except Exception as e:
            flash(f'Ocorreu um erro durante o processamento da imagem: {e}', 'danger')
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
