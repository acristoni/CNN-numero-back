FROM public.ecr.aws/lambda/python:3.8

# Copia os arquivos do projeto para o diretório de trabalho da Lambda
COPY handler.py mnist_model.keras requirements.txt ./

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Define o handler da função
CMD ["handler.handler"]
