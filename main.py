"""
API principal para o OdontoBot AI.

FastAPI que serve de webhook para a Z-API, processa mensagens do WhatsApp via
OpenAI e interage com o banco para gerenciar pacientes e agendamentos.
"""

import os, json, asyncio
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ───────────────── 1. VARIÁVEIS DE AMBIENTE ────────────────── #
load_dotenv()

DATABASE_URL      = os.getenv("DATABASE_URL")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ZAPI_API_URL      = os.getenv("ZAPI_API_URL")
ZAPI_INSTANCE_ID  = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN        = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN = os.getenv("ZAPI_CLIENT_TOKEN")

if not all([DATABASE_URL, OPENAI_API_KEY, ZAPI_API_URL,
            ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente obrigatória não foi definida.")

# SDK OpenAI – funciona tanto com openai>=1.0 quanto 0.x
try:
    import openai
    if hasattr(openai, "OpenAI"):  # SDK >= 1.0
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        def chat_completion(**kwargs):
            return openai_client.chat.completions.create(**kwargs)
    else:                           # SDK 0.x
        openai.api_key = OPENAI_API_KEY
        def chat_completion(**kwargs):
            return openai.ChatCompletion.create(**kwargs)
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 2. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id       = Column(Integer, primary_key=True, index=True)
    nome     = Column(String, index=True)
    telefone = Column(String, unique=True, index=True, nullable=False)
    agendamentos = relationship(
        "Agendamento", back_populates="paciente", cascade="all, delete-orphan"
    )

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id          = Column(Integer, primary_key=True, index=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora   = Column(DateTime, index=True, nullable=False)
    procedimento= Column(String, nullable=False)
    status      = Column(String, default="confirmado")
    paciente    = relationship("Paciente", back_populates="agendamentos")

engine        = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal  = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def criar_tabelas():
    Base.metadata.create_all(bind=engine)

# ───────────────── 3. FERRAMENTAS (FUNÇÕES) ────────────────── #
def buscar_ou_criar_paciente(db: Session, telefone: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=telefone).first()
    if not paciente:
        paciente = Paciente(telefone=telefone, nome=f"Paciente {telefone}")
        db.add(paciente)
        db.commit(); db.refresh(paciente)
    return paciente

def agendar_consulta(db: Session, telefone_paciente: str,
                     data_hora_agendamento: str, procedimento: str) -> str:
    try:
        data_hora = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)
    db.add(Agendamento(paciente_id=paciente.id,
                       data_hora=data_hora,
                       procedimento=procedimento))
    db.commit()
    return (f"Agendamento '{procedimento}' confirmado para "
            f"{data_hora.strftime('%d/%m/%Y às %H:%M')}.")

def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(
        Agendamento.paciente_id == paciente.id,
        Agendamento.data_hora >= datetime.now(),
        Agendamento.status == "confirmado"
    ).order_by(Agendamento.data_hora).all()
    if not ags:
        return "Você não possui agendamentos futuros."
    linhas = [
        f"- ID {ag.id}: {ag.procedimento} em {ag.data_hora.strftime('%d/%m/%Y às %H:%M')}"
        for ag in ags
    ]
    return "Seus próximos agendamentos:\n" + "\n".join(linhas)

def cancelar_agendamento(db: Session, telefone_paciente: str,
                         id_agendamento: int) -> str:
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)
    ag = db.query(Agendamento).filter_by(
        id=id_agendamento, paciente_id=paciente.id
    ).first()
    if not ag:
        return f"Agendamento ID {id_agendamento} não encontrado."
    if ag.status == "cancelado":
        return "Esse agendamento já está cancelado."
    ag.status = "cancelado"; db.commit()
    return f"Agendamento {id_agendamento} cancelado com sucesso."

# Dicionário para a IA chamar
available_functions: Dict[str, Any] = {
    "agendar_consulta": agendar_consulta,
    "consultar_meus_agendamentos": consultar_meus_agendamentos,
    "cancelar_agendamento": cancelar_agendamento,
}

# Especificação JSON das ferramentas
tools = [
    {
        "type": "function",
        "function": {
            "name": "agendar_consulta",
            "description": "Agenda uma consulta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_hora_agendamento": {"type": "string"},
                    "procedimento": {"type": "string"},
                },
                "required": ["data_hora_agendamento", "procedimento"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_meus_agendamentos",
            "description": "Consulta agendamentos futuros.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancelar_agendamento",
            "description": "Cancela um agendamento.",
            "parameters": {
                "type": "object",
                "properties": {"id_agendamento": {"type": "integer"}},
                "required": ["id_agendamento"],
            },
        },
    },
]

# ───────────────── 4. APP FASTAPI ───────────────────────────── #
app = FastAPI(
    title="OdontoBot AI",
    description="Automação de WhatsApp para clínica odontológica.",
    version="1.0.1",
)

@app.on_event("startup")
async def startup_event():
    print(">> Criando tabelas…", flush=True)
    await asyncio.to_thread(criar_tabelas)
    print(">> Tabelas OK.", flush=True)

@app.get("/")
def health_get():
    return {"status": "ok"}

@app.head("/")
def health_head():
    # Render faz HEAD / no health-check
    return Response(status_code=200)

# ───────────────── 5. MODELO DO WEBHOOK ─────────────────────── #
class ZapiWebhookPayload(BaseModel):
    phone: str
    message: Optional[str] = Field(None, alias="text")

# ───────────────── 6. UTIL ───────────────────────────────────── #
async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": telefone, "message": mensagem}
    headers = {
        "Content-Type": "application/json",
        "Client-Token": ZAPI_CLIENT_TOKEN,
    }
    # ASCII-safe para evitar problemas de charset
    payload_ascii = json.dumps(payload, ensure_ascii=True)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, content=payload_ascii, headers=headers)
            r.raise_for_status()
        except Exception as exc:
            print(f"Falha ao enviar Z-API: {exc}", flush=True)

# ───────────────── 7. WEBHOOK ───────────────────────────────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    payload: ZapiWebhookPayload, db: Session = Depends(get_db)
):
    if not payload.message:
        return {"status": "ignorado", "motivo": "mensagem vazia"}

    telefone = payload.phone
    user_msg = payload.message

    msgs = [
        {
            "role": "system",
            "content": (
                "Você é atendente da 'Clínica Odonto Feliz'. "
                f"Hoje é {datetime.now().strftime('%d/%m/%Y')}. "
                "Use as ferramentas para agendar, consultar ou cancelar consultas."
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    try:
        # 1ª chamada
        resp = chat_completion(model="gpt-3.5-turbo",
                               messages=msgs,
                               tools=tools,
                               tool_choice="auto")
        ai_msg = resp.choices[0].message
        msgs.append(ai_msg)

        # Loop ferramenta(s) → IA
        while ai_msg.tool_calls:
            for call in ai_msg.tool_calls:
                fname = call.function.name
                f_args = json.loads(call.function.arguments)
                func  = available_functions.get(fname)
                if not func:
                    raise HTTPException(500, f"Função desconhecida: {fname}")
                result = func(db=db, telefone_paciente=telefone, **f_args)
                msgs.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": fname,
                    "content": result,
                })
            # nova resposta da IA
            resp = chat_completion(model="gpt-3.5-turbo", messages=msgs)
            ai_msg = resp.choices[0].message
            msgs.append(ai_msg)

        resposta_final = ai_msg.content

    except Exception as e:
        print("Erro na interação IA:", e, flush=True)
        resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."

    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
