FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY ABFormer_env.yml .

RUN conda env create -f ABFormer_env.yml
RUN conda clean -afy

COPY . .

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "ABFormer", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]