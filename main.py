"""
API principal para o OdontoBot AI.

Este módulo contém a aplicação FastAPI que serve como webhook para a Z-API,
processa mensagens de usuários do WhatsApp usando a IA da OpenAI e interage
com um banco de dados para gerenciar pacientes e agendamentos.
"""

import os
import json
from datetime import datetime
from typing import Optional

import httpx
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# --- 1. CONFIGURAÇÃO E VARIÁVEIS DE AMBIENTE ---
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZAPI_API_URL = os.getenv("ZAPI_API_URL")
ZAPI_INSTANCE_ID = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN = os.getenv("ZAPI_CLIENT_TOKEN")

if not all([DATABASE_URL, OPENAI_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise ValueError("Verifique seu arquivo .env, uma ou mais variáveis de ambiente não foram definidas.")

openai.api_key = OPENAI_API_KEY

# --- 2. BANCO DE DADOS (SQLALCHEMY) ---
Base = declarative_base()

class Paciente(Base):
    """Modelo da tabela de Pacientes."""
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True)
    telefone = Column(String, unique=True, index=True, nullable=False)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan")

class Agendamento(Base):
    """Modelo da tabela de Agendamentos."""
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True, index=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora = Column(DateTime, index=True, nullable=False)
    procedimento = Column(String, nullable=False)
    status = Column(String, default="confirmado")
    paciente = relationship("Paciente", back_populates="agendamentos")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def criar_tabelas_db():
    """Cria as tabelas no banco de dados se elas não existirem."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependência do FastAPI para obter uma sessão do banco de dados."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. FUNÇÕES DE SERVIÇO (AS "FERRAMENTAS" DA IA) ---

def buscar_ou_criar_paciente(db: Session, telefone: str) -> Paciente:
    """Busca um paciente pelo telefone. Se não existir, cria um novo."""
    paciente = db.query(Paciente).filter(Paciente.telefone == telefone).first()
    if not paciente:
        paciente = Paciente(telefone=telefone, nome=f"Paciente {telefone}")
        db.add(paciente)
        db.commit()
        db.refresh(paciente)
    return paciente

def agendar_consulta(db: Session, telefone_paciente: str, data_hora_agendamento: str, procedimento: str) -> str:
    """Agenda uma nova consulta para um paciente."""
    try:
        data_hora_obj = datetime.strptime(data_hora_agendamento, '%Y-%m-%d %H:%M')
    except ValueError:
        return "Formato de data e hora inválido. Por favor, use AAAA-MM-DD HH:MM."

    paciente = buscar_ou_criar_paciente(db, telefone_paciente)

    novo_agendamento = Agendamento(
        paciente_id=paciente.id, data_hora=data_hora_obj, procedimento=procedimento
    )
    db.add(novo_agendamento)
    db.commit()

    return f"Agendamento para '{procedimento}' confirmado para {data_hora_obj.strftime('%d/%m/%Y às %H:%M')}."

def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    """Consulta os agendamentos futuros de um paciente."""
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)
    agendamentos = db.query(Agendamento).filter(
        Agendamento.paciente_id == paciente.id,
        Agendamento.data_hora >= datetime.now(),
        Agendamento.status == "confirmado"
    ).order_by(Agendamento.data_hora).all()

    if not agendamentos:
        return "Você não possui agendamentos futuros."

    lista_formatada = [
        f"- ID {ag.id}: {ag.procedimento} em {ag.data_hora.strftime('%d/%m/%Y às %H:%M')}"
        for ag in agendamentos
    ]
    return "Seus próximos agendamentos são:\n" + "\n".join(lista_formatada)

def cancelar_agendamento(db: Session, telefone_paciente: str, id_agendamento: int) -> str:
    """Cancela um agendamento específico de um paciente."""
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)
    agendamento = db.query(Agendamento).filter(
        Agendamento.id == id_agendamento, Agendamento.paciente_id == paciente.id
    ).first()

    if not agendamento:
        return f"O agendamento com ID {id_agendamento} não foi encontrado."
    if agendamento.status == "cancelado":
        return f"O agendamento {id_agendamento} já está cancelado."

    agendamento.status = "cancelado"
    db.commit()
    return f"Agendamento {id_agendamento} ({agendamento.procedimento}) foi cancelado."

# --- 4. CONFIGURAÇÃO DA API (FASTAPI) ---
app = FastAPI(
    title="OdontoBot AI",
    description="API para automação de atendimento de consultório odontológico com IA.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    """Executa ao iniciar a API para criar as tabelas do banco."""
    print("Iniciando API e criando tabelas do banco de dados...", flush=True)
    criar_tabelas_db()
    print("Tabelas prontas.", flush=True)

@app.get("/")
def health_check():
    """Endpoint de verificação de saúde da API."""
    return {"status": "OdontoBot AI está no ar!"}

# --- 5. LÓGICA DO WEBHOOK E DA IA ---

class ZapiWebhookPayload(BaseModel):
    """Modelo do payload recebido da Z-API."""
    phone: str
    text: Optional[str] = Field(None, alias='message')

available_functions = {
    "agendar_consulta": agendar_consulta,
    "consultar_meus_agendamentos": consultar_meus_agendamentos,
    "cancelar_agendamento": cancelar_agendamento,
}

tools = [
    {"type": "function", "function": {"name": "agendar_consulta", "description": "Agenda uma consulta.", "parameters": {"type": "object", "properties": {"data_hora_agendamento": {"type": "string", "description": "Data/hora no formato 'YYYY-MM-DD HH:MM'."}, "procedimento": {"type": "string", "description": "Procedimento a agendar."}}, "required": ["data_hora_agendamento", "procedimento"]}}},
    {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Consulta agendamentos futuros.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "cancelar_agendamento", "description": "Cancela um agendamento por ID.", "parameters": {"type": "object", "properties": {"id_agendamento": {"type": "integer", "description": "ID do agendamento."}}, "required": ["id_agendamento"]}}}
]

async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    """Envia uma mensagem de texto para um número de telefone via Z-API."""
    print(f"-> INICIANDO ENVIO PARA Z-API: {mensagem}", flush=True)
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": telefone, "message": mensagem}
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"-> RESPOSTA ENVIADA COM SUCESSO PARA {telefone}.", flush=True)
        except httpx.HTTPStatusError as exc:
            print(f"-> ERRO HTTP AO ENVIAR PARA Z-API: {exc.response.status_code} - {exc.response.text}", flush=True)
        except Exception as exc:
            print(f"-> ERRO INESPERADO AO CHAMAR Z-API: {exc}", flush=True)

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(payload: ZapiWebhookPayload, db: Session = Depends(get_db)):
    """Recebe, processa e responde mensagens do WhatsApp."""
    print("-> WEBHOOK RECEBIDO", flush=True)
    telefone_usuario, mensagem_usuario = payload.phone, payload.text
    if not mensagem_usuario:
        return {"status": "ignorado", "motivo": "sem mensagem de texto"}

    print(f"-> MENSAGEM DE {telefone_usuario}: '{mensagem_usuario}'", flush=True)
    messages = [
        {"role": "system", "content": "Você é um atendente da 'Clínica Odonto Feliz'. Hoje é "
                                      f"{datetime.now().strftime('%d/%m/%Y')}. Use as ferramentas "
                                      "disponíveis para agendar, consultar ou cancelar consultas."},
        {"role": "user", "content": mensagem_usuario}
    ]

    try:
        print("-> CHAMANDO OPENAI...", flush=True)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, tools=tools, tool_choice="auto"
        )
        response_message = response.choices[0].message
        messages.append(response_message)
        print("-> OPENAI RESPONDEU. Verificando se há ferramentas...", flush=True)

        while response_message.tool_calls:
            print(f"-> IA SOLICITOU FERRAMENTA: {response_message.tool_calls[0].function.name}", flush=True)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                if not function_to_call: raise HTTPException(500, f"Função inválida: {function_name}")
                function_args = json.loads(tool_call.function.arguments)
                function_args.update({"telefone_paciente": telefone_usuario, "db": db})
                function_response = function_to_call(**function_args)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
            
            print("-> RE-CHAMANDO OPENAI COM RESULTADO DA FERRAMENTA...", flush=True)
            second_response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            response_message = second_response.choices[0].message
            messages.append(response_message)
            print("-> OPENAI GEROU RESPOSTA FINAL.", flush=True)

        resposta_final = response_message.content

    except Exception as e:
        print(f"-> ERRO NO BLOCO TRY/EXCEPT DA IA: {e}", flush=True)
        resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente mais tarde."

    await enviar_resposta_whatsapp(telefone_usuario, resposta_final)
    print("-> PROCESSAMENTO DO WEBHOOK CONCLUÍDO.", flush=True)
    return {"status": "processado", "resposta_enviada": resposta_final}
