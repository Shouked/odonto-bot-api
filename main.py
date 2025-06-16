"""
API principal do OdontoBot AI.

FastAPI que serve de webhook para a Z-API, processa mensagens de WhatsApp via
OpenAI e interage com o banco para gerenciar pacientes e agendamentos.
"""

import os, json, asyncio
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response, Request
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

# SDK OpenAI – compatível com openai>=1.0 ou 0.x
try:
    import openai
    if hasattr(openai, "OpenAI"):          # SDK 1.x
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        def chat_completion(**kw): return openai_client.chat.completions.create(**kw)
    else:                                  # SDK 0.x
        openai.api_key = OPENAI_API_KEY
        def chat_completion(**kw): return openai.ChatCompletion.create(**kw)
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 2. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id       = Column(Integer, primary_key=True, index=True)
    nome     = Column(String, index=True)
    telefone = Column(String, unique=True, index=True, nullable=False)
    agendamentos = relationship("Agendamento", back_populates="paciente",
                                cascade="all, delete-orphan")

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id          = Column(Integer, primary_key=True, index=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora   = Column(DateTime, index=True, nullable=False)
    procedimento= Column(String, nullable=False)
    status      = Column(String, default="confirmado")
    paciente    = relationship("Paciente", back_populates="agendamentos")

engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def criar_tabelas():
    Base.metadata.create_all(bind=engine)

# ───────────────── 3. FERRAMENTAS ──────────────────────────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente:
        paciente = Paciente(telefone=tel, nome=f"Paciente {tel}")
        db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente

def agendar_consulta(db: Session, telefone_paciente: str,
                     data_hora_agendamento: str, procedimento: str) -> str:
    try:
        dt = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    db.add(Agendamento(paciente_id=pac.id, data_hora=dt, procedimento=procedimento))
    db.commit()
    return (f"Agendamento '{procedimento}' confirmado para "
            f"{dt.strftime('%d/%m/%Y às %H:%M')}.")

def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(
        Agendamento.paciente_id == pac.id,
        Agendamento.data_hora >= datetime.now(),
        Agendamento.status == "confirmado",
    ).order_by(Agendamento.data_hora).all()
    if not ags:
        return "Você não possui agendamentos futuros."
    linhas = [f"- ID {a.id}: {a.procedimento} em "
              f"{a.data_hora.strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos:\n" + "\n".join(linhas)

def cancelar_agendamento(db: Session, telefone_paciente: str,
                         id_agendamento: int) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ag  = db.query(Agendamento).filter_by(id=id_agendamento,
                                          paciente_id=pac.id).first()
    if not ag:
        return f"Agendamento ID {id_agendamento} não encontrado."
    if ag.status == "cancelado":
        return "Esse agendamento já está cancelado."
    ag.status = "cancelado"; db.commit()
    return f"Agendamento {id_agendamento} cancelado com sucesso."

available_functions: Dict[str, Any] = {
    "agendar_consulta": agendar_consulta,
    "consultar_meus_agendamentos": consultar_meus_agendamentos,
    "cancelar_agendamento": cancelar_agendamento,
}

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
    version="1.0.2",
)

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(criar_tabelas)

@app.get("/")
def health_get(): return {"status": "ok"}

@app.head("/")
def health_head(): return Response(status_code=200)

# ───────────────── 5. MODELOS DE PAYLOAD ───────────────────── #
class ZapiText(BaseModel):
    # Use 'message' OU 'body' conforme sua instância Z-API
    message: Optional[str] = None
    body:    Optional[str] = None

class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None

# ───────────────── 6. UTIL ──────────────────────────────────── #
async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": telefone, "message": mensagem}
    headers = {
        "Content-Type": "application/json",
        "Client-Token": ZAPI_CLIENT_TOKEN,
    }
    payload_ascii = json.dumps(payload, ensure_ascii=True)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, content=payload_ascii, headers=headers)
            r.raise_for_status()
        except Exception as exc:
            print("Falha ao enviar para Z-API:", exc, flush=True)

# ───────────────── 7. WEBHOOK ───────────────────────────────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request, db: Session = Depends(get_db)
):
    raw = await request.json()
    print(">>> PAYLOAD RECEBIDO:", raw, flush=True)

    try:
        payload = ZapiWebhookPayload(**raw)
    except Exception as e:
        print("Erro ao validar payload:", e, flush=True)
        raise HTTPException(422, "Formato de payload inválido")

    telefone = payload.phone
    txt_obj  = payload.text
    mensagem_usuario = None
    if txt_obj:
        # Prioriza 'message', depois 'body'
        mensagem_usuario = txt_obj.message or txt_obj.body
    if not mensagem_usuario:
        return {"status": "ignorado", "motivo": "sem mensagem de texto"}

    msgs = [
        {
            "role": "system",
            "content": (
                "Você é atendente da 'Clínica Odonto Feliz'. "
                f"Hoje é {datetime.now().strftime('%d/%m/%Y')}. "
                "Use as ferramentas para agendar, consultar ou cancelar consultas."
            ),
        },
        {"role": "user", "content": mensagem_usuario},
    ]

    try:
        resp = chat_completion(model="gpt-3.5-turbo",
                               messages=msgs,
                               tools=tools,
                               tool_choice="auto")
        ai_msg = resp.choices[0].message
        msgs.append(ai_msg)

        while ai_msg.tool_calls:
            for call in ai_msg.tool_calls:
                fname = call.function.name
                f_args = json.loads(call.function.arguments)
                func   = available_functions.get(fname)
                if not func:
                    raise HTTPException(500, f"Função desconhecida: {fname}")
                result = func(db=db, telefone_paciente=telefone, **f_args)
                msgs.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": fname,
                    "content": result,
                })
            resp   = chat_completion(model="gpt-3.5-turbo", messages=msgs)
            ai_msg = resp.choices[0].message
            msgs.append(ai_msg)

        resposta_final = ai_msg.content

    except Exception as e:
        print("Erro na interação IA:", e, flush=True)
        resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."

    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
