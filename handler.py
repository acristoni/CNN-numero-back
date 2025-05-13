import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import logging
import os
import tensorflow as tf  # Importando TensorFlow

# Configura o logger para enviar logs ao CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mnist_model.keras') # Atualize a extensão para .keras

# Carrega o modelo Keras
def load_model():
    logger.info('Carregando o modelo Keras...')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info('Modelo Keras carregado com sucesso.')
        return model
    except Exception as e:
        logger.error(f'Erro ao carregar o modelo Keras: {str(e)}')
        raise

model = load_model()

def predict(image_np):
    logger.info('Realizando predição com o modelo Keras...')
    image_np = image_np.astype(np.float32)
    image_np = np.expand_dims(image_np, axis=0) # Adiciona a dimensão do batch
    logger.debug(f'Shape da imagem para predição: {image_np.shape}')

    predictions = model.predict(image_np)
    prediction = int(np.argmax(predictions))
    logger.info(f'Predição realizada pelo modelo Keras: {prediction}')
    return prediction

# Função para converter imagem base64 em um array NumPy (mantém a mesma)
def decode_base64_image(base64_string):
    logger.info('Decodificando a imagem base64...')
    img_data = base64.b64decode(base64_string.split(',')[1])  # Remove a parte "data:image/png;base64,"
    image = Image.open(BytesIO(img_data)).convert('L')  # Converte para escala de cinza
    image = image.resize((28, 28))  # Redimensiona para 28x28 pixels
    image_np = np.array(image)  # Converte para um array NumPy
    logger.info('Imagem decodificada com sucesso.')
    return image_np

def handler(event, context):
    try:
        logger.info('Iniciando o processamento da solicitação...')

        body = json.loads(event['body'])
        base64_image = body['image']
        logger.info("Imagem base64 extraída com sucesso.")

        image_np = decode_base64_image(base64_image)
        logger.info(f"Shape da imagem original: {image_np.shape}")

        image_np = image_np / 255.0
        # Não precisa mais do reshape aqui, o modelo Keras cuidará disso na predição

        result = predict(image_np)

        logger.info(f'Predição retornada: {result}')
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result})
        }

    except Exception as e:
        logger.error(f'Erro ao processar a solicitação: {str(e)}')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }