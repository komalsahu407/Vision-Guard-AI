# VisionGuard AI: public demo image (Hugging Face Spaces, Docker SDK)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    VG_HOST=0.0.0.0 \
    PORT=7860 \
    VG_PUBLIC=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p data/videos data/cache

EXPOSE 7860
CMD ["python", "app.py", "--no-browser"]
