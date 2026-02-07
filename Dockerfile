FROM continuumio/miniconda3

WORKDIR /code

# 1. Install System Tools
RUN apt-get update && apt-get install -y cmake && apt-get clean

# 2. Create Environment (Renamed to 'classic_env' to force fresh build)
# We install dlib via Conda to avoid memory crashes.
RUN conda create -n classic_env python=3.9 dlib face_recognition -c conda-forge -y

# 3. Activate the environment
SHELL ["conda", "run", "-n", "classic_env", "/bin/bash", "-c"]

# 4. THE TIME MACHINE FIX
# We uninstall the "bad" auto-installed versions first.
RUN pip uninstall -y gradio huggingface_hub

# We install the specific "Late 2023" versions that we know are stable.
RUN pip install --no-cache-dir \
    "gradio==3.50.2" \
    "huggingface_hub==0.19.3" \
    "numpy<1.27.0" \
    opencv-python-headless \
    Pillow

# 5. Copy App
COPY . .

# 6. Permissions
RUN useradd -m -u 1000 user || true
RUN chown -R user:user /code
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

CMD ["conda", "run", "-n", "classic_env", "python", "app.py"]
