
FROM continuumio/miniconda3

WORKDIR /app

COPY . .
RUN apt-get update && apt-get install -y libgl1
RUN conda env create -f environment.yml


SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

EXPOSE 9000

CMD ["conda", "run", "--no-capture-output", "-n", "env", "python", "check/app.py"]