FROM rocm/pytorch:rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.9.1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY /src /src

CMD ["/opt/venv/bin/python", "src/main.py"]