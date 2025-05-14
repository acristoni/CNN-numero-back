import json
import tflite_runtime.interpreter as tflite  # Importando tflite-runtime
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import logging
import os

# Configura o logger para enviar logs ao CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mnist_model.tflite')

# Carrega o modelo embutido usando tflite-runtime
def load_model():
    logger.info('Carregando o modelo TensorFlow Lite...')
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    logger.info('Modelo carregado com sucesso.')
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_np):
    logger.info('Realizando predição...')
    image_np = image_np.astype(np.float32)

    expected_shape = input_details[0]['shape']
    logger.debug(f'Shape esperado pelo modelo: {expected_shape}')
    logger.debug(f'Shape fornecido: {image_np.shape}')

    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = int(np.argmax(output))
    logger.info(f'Predição realizada: {prediction}')
    return prediction

# Função para converter imagem base64 em um array NumPy
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

        image_np = image_np.reshape(1, 28, 28, 1)

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
