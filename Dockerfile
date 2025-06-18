# ---------------------------------------------------------
# Dockerfile do OdontoBot – imagem leve e reprodutível
# ---------------------------------------------------------
FROM python:3.11-slim

# 1. Ajustes de ambiente básicos
ENV PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo

# 2. Dependências de sistema (tzdata para timezone strings)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 3. Diretório de trabalho
WORKDIR /app

# 4. Copia requirements primeiro (cache de camadas)
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o código restante
COPY . .

# 6. Comando de inicialização
#    - 1 worker assíncrono suficiente p/ IO-bound
#    - --timeout-keep-alive 120 evita fechar conexões longas
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", \
     "--workers", "1", "--timeout-keep-alive", "120", "--access-log"]
