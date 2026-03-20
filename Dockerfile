FROM kubricdocker/kubric

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["python", "dataset-generator.py"]
