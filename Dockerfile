FROM rocm/pytorch:rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.9.1

WORKDIR /app

COPY docker_requirements.txt .

# separate requirements without torch => image has torch already installed
RUN pip install -r docker_requirements.txt 

COPY /src /src

CMD ["/opt/venv/bin/python", "src/main.py"] 