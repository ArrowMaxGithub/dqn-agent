FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

WORKDIR /app

COPY docker_requirements.txt .

# separate requirements without torch => image has torch already installed
RUN pip install -r docker_requirements.txt 

COPY /src /src

CMD ["python", "src/main.py"]