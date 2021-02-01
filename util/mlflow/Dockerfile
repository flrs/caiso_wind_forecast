FROM continuumio/miniconda3

RUN conda install mlflow -c conda-forge

ENV MLFLOW_HOME /opt/mlflow

COPY ./mlruns ${MLFLOW_HOME}/mlruns

WORKDIR ${MLFLOW_HOME}

CMD ["conda", "run", "--no-capture-output", "mlflow", "server", "--port", "5000", "--host", "0.0.0.0"]