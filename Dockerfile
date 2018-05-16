FROM python:3
FROM continuumio/miniconda3
RUN mkdir -p cnn-cell-counting/
RUN conda update -n base conda
RUN pip install --upgrade pip
COPY . cnn-cell-counting/
WORKDIR cnn-cell-counting/
RUN pip install -r requirements.txt
RUN conda install matplotlib
EXPOSE 1883
