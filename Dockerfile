# ---------------------------------------------------------
# Dockerfile do OdontoBot – imagem leve e previsível
# ---------------------------------------------------------
FROM python:3.11-slim

# 1. Diretório de trabalho
WORKDIR /app

# 2. Copia todo o código
COPY . .

# 3. Instala dependências (arquivo requirements.txt na raiz)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4. Comando de inicialização
#    - 1 worker já é suficiente para IO-bound e economiza RAM
#    - --timeout-keep-alive 120 evita fechar conexões longas
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", \
     "--workers", "1", "--timeout-keep-alive", "120", "--access-log"]