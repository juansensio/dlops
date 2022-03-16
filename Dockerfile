FROM continuumio/miniconda3

RUN conda install -y -c conda-forge \ 
    pillow \
    onnxruntime \
    fastapi \ 
    uvicorn \
    python-multipart 

COPY ./models /models
COPY ./app.py /app.py

CMD uvicorn app:app --host=0.0.0.0 --port=$PORT