"""
API principal do OdontoBot AI. Versão com Histórico e CRUD completo.

FastAPI que serve de webhook para a Z-API, processa mensagens de WhatsApp via
OpenAI e interage com o banco para gerenciar pacientes e agendamentos.
"""

import os, json, asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text
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

# SDK OpenAI
try:
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    def chat_completion(**kw): return openai_client.chat.completions.create(**kw)
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 2. BANCO DE DADOS (COM NOVO MODELO) ─────── #
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id       = Column(Integer, primary_key=True, index=True)
    nome     = Column(String, index=True)
    telefone = Column(String, unique=True, index=True, nullable=False)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan")
    historico    = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id          = Column(Integer, primary_key=True, index=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora   = Column(DateTime, index=True, nullable=False)
    procedimento= Column(String, nullable=False)
    status      = Column(String, default="confirmado")
    paciente    = relationship("Paciente", back_populates="agendamentos")

# NOVO MODELO PARA HISTÓRICO
class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id          = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    role        = Column(String, nullable=False)  # 'user' ou 'assistant'
    content     = Column(Text, nullable=False)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    paciente    = relationship("Paciente", back_populates="historico")


engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def criar_tabelas():
    Base.metadata.create_all(bind=engine)

# ───────────────── 3. FERRAMENTAS (COM NOVA FERRAMENTA) ────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    # ... (sem alterações)
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente:
        paciente = Paciente(telefone=tel, nome=f"Paciente {tel}")
        db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente

def agendar_consulta(db: Session, telefone_paciente: str, data_hora_agendamento: str, procedimento: str) -> str:
    # ... (sem alterações)
    try: dt = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    db.add(Agendamento(paciente_id=pac.id, data_hora=dt, procedimento=procedimento))
    db.commit()
    return f"Agendamento '{procedimento}' confirmado para {dt.strftime('%d/%m/%Y às %H:%M')}."

def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    # ... (sem alterações)
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.data_hora >= datetime.now(), Agendamento.status == "confirmado").order_by(Agendamento.data_hora).all()
    if not ags: return "Você não possui agendamentos futuros."
    linhas = [f"- ID {a.id}: {a.procedimento} em {a.data_hora.strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos:\n" + "\n".join(linhas)

def cancelar_agendamento(db: Session, telefone_paciente: str, id_agendamento: int) -> str:
    # ... (sem alterações)
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ag  = db.query(Agendamento).filter_by(id=id_agendamento, paciente_id=pac.id).first()
    if not ag: return f"Agendamento ID {id_agendamento} não encontrado."
    if ag.status == "cancelado": return "Esse agendamento já está cancelado."
    ag.status = "cancelado"; db.commit()
    return f"Agendamento {id_agendamento} cancelado com sucesso."

# NOVA FERRAMENTA DE UPDATE
def reagendar_consulta(db: Session, telefone_paciente: str, id_agendamento: int, novo_data_hora_agendamento: str) -> str:
    """Reagenda uma consulta existente para uma nova data ou hora."""
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ag = db.query(Agendamento).filter_by(id=id_agendamento, paciente_id=pac.id, status="confirmado").first()
    if not ag:
        return f"Não foi possível encontrar um agendamento ativo com o ID {id_agendamento} para reagendar."
    try:
        nova_dt = datetime.strptime(novo_data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    
    ag.data_hora = nova_dt
    db.commit()
    return f"Agendamento ID {id_agendamento} reagendado com sucesso para {nova_dt.strftime('%d/%m/%Y às %H:%M')}."

# ATUALIZADO: Dicionário de funções disponíveis
available_functions: Dict[str, Any] = {
    "agendar_consulta": agendar_consulta,
    "consultar_meus_agendamentos": consultar_meus_agendamentos,
    "cancelar_agendamento": cancelar_agendamento,
    "reagendar_consulta": reagendar_consulta, # Adicionada nova função
}

# ATUALIZADO: Lista de ferramentas para a IA
tools = [
    # ... (ferramentas anteriores sem alteração)
    {"type": "function", "function": {"name": "agendar_consulta", "description": "Agenda uma nova consulta.", "parameters": {"type": "object", "properties": {"data_hora_agendamento": {"type": "string"}, "procedimento": {"type": "string"}}, "required": ["data_hora_agendamento", "procedimento"]}}},
    {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Consulta os agendamentos futuros do paciente.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "cancelar_agendamento", "description": "Cancela um agendamento existente pelo ID.", "parameters": {"type": "object", "properties": {"id_agendamento": {"type": "integer"}}, "required": ["id_agendamento"]}}},
    # NOVA DEFINIÇÃO DE FERRAMENTA
    {
        "type": "function",
        "function": {
            "name": "reagendar_consulta",
            "description": "Altera a data ou hora de uma consulta já existente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id_agendamento": {"type": "integer", "description": "O ID do agendamento a ser modificado."},
                    "novo_data_hora_agendamento": {"type": "string", "description": "A nova data e hora no formato AAAA-MM-DD HH:MM."}
                },
                "required": ["id_agendamento", "novo_data_hora_agendamento"],
            },
        },
    },
]

# ───────────────── 4. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp para clínica odontológica.", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    # Isso agora também criará a tabela 'historico_conversas' na primeira vez
    await asyncio.to_thread(criar_tabelas)
    print("Tabelas verificadas/criadas.", flush=True)

@app.get("/")
def health_get(): return {"status": "ok"}

@app.head("/")
def health_head(): return Response(status_code=200)

# ───────────────── 5. MODELOS DE PAYLOAD ───────────────────── #
class ZapiText(BaseModel):
    message: Optional[str] = None
    body:    Optional[str] = None
class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None

# ───────────────── 6. UTIL ──────────────────────────────────── #
async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    # ... (sem alterações)
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": telefone, "message": mensagem}
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
        except Exception as exc: print("Falha ao enviar para Z-API:", exc, flush=True)

# ───────────────── 7. WEBHOOK (TOTALMENTE ATUALIZADO) ──────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    raw = await request.json()
    print(">>> PAYLOAD RECEBIDO:", raw, flush=True)

    try: payload = ZapiWebhookPayload(**raw)
    except Exception as e:
        print("Erro de validação Pydantic:", e, flush=True)
        raise HTTPException(422, "Formato de payload inválido")

    telefone = payload.phone
    mensagem_usuario = payload.text.message if payload.text else None
    if not mensagem_usuario:
        return {"status": "ignorado", "motivo": "sem mensagem"}

    # --- LÓGICA DE HISTÓRICO ---
    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    
    # 1. Salvar a mensagem atual do usuário no histórico
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario))
    db.commit()
    
    # 2. Carregar as últimas 24h de conversa
    limite_tempo = datetime.utcnow() - timedelta(hours=24)
    historico_recente = (
        db.query(HistoricoConversa)
        .filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= limite_tempo)
        .order_by(HistoricoConversa.timestamp)
        .all()
    )
    
    # 3. Montar a lista de mensagens para a IA, incluindo o histórico
    mensagens_para_ia: List[Dict[str, str]] = [
        {"role": "system", "content": f"Você é atendente da 'Clínica Odonto Feliz'. Hoje é {datetime.now().strftime('%d/%m/%Y')}. Use o histórico da conversa para ter contexto. Use as ferramentas para interagir com o sistema."},
    ]
    for msg in historico_recente:
        mensagens_para_ia.append({"role": msg.role, "content": msg.content})

    # --- LÓGICA DA IA (semelhante a antes, mas usando `mensagens_para_ia`) ---
    try:
        resp = chat_completion(model="gpt-4-turbo", messages=mensagens_para_ia, tools=tools, tool_choice="auto")
        ai_msg = resp.choices[0].message
        
        # Loop para ferramentas
        while ai_msg.tool_calls:
            msgs_com_ferramentas = mensagens_para_ia + [ai_msg]
            for call in ai_msg.tool_calls:
                fname = call.function.name
                f_args = json.loads(call.function.arguments)
                func = available_functions.get(fname)
                if not func: raise HTTPException(500, f"Função desconhecida: {fname}")
                result = func(db=db, telefone_paciente=telefone, **f_args)
                msgs_com_ferramentas.append({"tool_call_id": call.id, "role": "tool", "name": fname, "content": result})
            
            resp = chat_completion(model="gpt-4-turbo", messages=msgs_com_ferramentas)
            ai_msg = resp.choices[0].message
        
        resposta_final = ai_msg.content

    except Exception as e:
        print("Erro na interação IA:", e, flush=True)
        resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."

    # 4. Salvar a resposta final da IA no histórico
    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final))
    db.commit()

    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
