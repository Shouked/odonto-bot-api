"""
API principal do OdontoBot AI. Versão com OpenRouter (Gemini) e OpenAI (Whisper).

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
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ───────────────── 1. VARIÁVEIS DE AMBIENTE ────────────────── #
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ZAPI_API_URL = os.getenv("ZAPI_API_URL")
ZAPI_INSTANCE_ID = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN = os.getenv("ZAPI_CLIENT_TOKEN")

if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente obrigatória não foi definida.")

# ───────────────── 2. CONFIGURAÇÃO DOS CLIENTES DE IA ───────── #
try:
    import openai
    openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://github.com/Shouked/odonto-bot-api", "X-Title": "OdontoBot AI"},
    )
    def openrouter_chat_completion(**kw): return openrouter_client.chat.completions.create(**kw)
    async def transcrever_audio_whisper(audio_url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=30)
                response.raise_for_status()
                audio_bytes = response.content
            transcription = await asyncio.to_thread(
                openai_whisper_client.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.ogg", audio_bytes, "audio/ogg")
            )
            return transcription.text
        except Exception as e:
            print(f"Erro ao transcrever áudio: {e}", flush=True)
            return None
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 3. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()
class Paciente(Base):
    __tablename__ = "pacientes"
    id, nome, telefone = Column(Integer, primary_key=True), Column(String), Column(String, unique=True, nullable=False)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan")
    historico = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")
class Agendamento(Base):
    __tablename__ = "agendamentos"
    id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora, procedimento, status = Column(DateTime, nullable=False), Column(String, nullable=False), Column(String, default="confirmado")
    paciente = relationship("Paciente", back_populates="agendamentos")
class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    role, content, timestamp = Column(String, nullable=False), Column(Text, nullable=False), Column(DateTime, default=datetime.utcnow)
    paciente = relationship("Paciente", back_populates="historico")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
def criar_tabelas(): Base.metadata.create_all(bind=engine)

# ───────────────── 4. FERRAMENTAS ──────────────────────────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente: paciente = Paciente(telefone=tel, nome=f"Paciente {tel}"); db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente
def agendar_consulta(db: Session, telefone_paciente: str, data_hora_agendamento: str, procedimento: str) -> str:
    try: dt = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    db.add(Agendamento(paciente_id=pac.id, data_hora=dt, procedimento=procedimento))
    db.commit()
    return f"Agendamento '{procedimento}' confirmado para {dt.strftime('%d/%m/%Y às %H:%M')}."
def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.data_hora >= datetime.now(), Agendamento.status == "confirmado").order_by(Agendamento.data_hora).all()
    if not ags: return "Você não possui agendamentos futuros."
    linhas = [f"- ID {a.id}: {a.procedimento} em {a.data_hora.strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos são:\n" + "\n".join(linhas)
def cancelar_agendamento(db: Session, telefone_paciente: str, id_agendamento: int) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ag = db.query(Agendamento).filter_by(id=id_agendamento, paciente_id=pac.id).first()
    if not ag: return f"Agendamento ID {id_agendamento} não encontrado."
    if ag.status == "cancelado": return "Esse agendamento já está cancelado."
    ag.status = "cancelado"; db.commit()
    return f"Agendamento {id_agendamento} cancelado com sucesso."
def consultar_e_reagendar_inteligente(db: Session, telefone_paciente: str, novo_data_hora_agendamento: str) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    agendamentos_ativos = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.data_hora >= datetime.now(), Agendamento.status == "confirmado").all()
    if not agendamentos_ativos: return "Você não tem nenhum agendamento futuro para reagendar."
    if len(agendamentos_ativos) > 1: return "Encontrei mais de um agendamento futuro. Qual deles você gostaria de reagendar? Por favor, informe o ID.\n" + consultar_meus_agendamentos(db, telefone_paciente)
    agendamento_para_reagendar = agendamentos_ativos[0]
    try: nova_dt = datetime.strptime(novo_data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "O formato da nova data e hora parece inválido. Por favor, use AAAA-MM-DD HH:MM."
    id_antigo = agendamento_para_reagendar.id; agendamento_para_reagendar.data_hora = nova_dt; db.commit()
    return f"Pronto! Seu agendamento (ID {id_antigo}) foi reagendado com sucesso para {nova_dt.strftime('%d/%m/%Y às %H:%M')}."
available_functions = {"agendar_consulta": agendar_consulta, "consultar_meus_agendamentos": consultar_meus_agendamentos, "cancelar_agendamento": cancelar_agendamento, "consultar_e_reagendar_inteligente": consultar_e_reagendar_inteligente}
tools = [{"type": "function", "function": {"name": "agendar_consulta", "description": "Agenda uma nova consulta.", "parameters": {"type": "object", "properties": {"data_hora_agendamento": {"type": "string"}, "procedimento": {"type": "string"}}, "required": ["data_hora_agendamento", "procedimento"]}}}, {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Lista agendamentos futuros do paciente, com IDs.", "parameters": {"type": "object", "properties": {}}}}, {"type": "function", "function": {"name": "cancelar_agendamento", "description": "Cancela um agendamento por ID.", "parameters": {"type": "object", "properties": {"id_agendamento": {"type": "integer"}}, "required": ["id_agendamento"]}}}, {"type": "function", "function": {"name": "consultar_e_reagendar_inteligente", "description": "Ferramenta inteligente para reagendar uma consulta, encontrando o agendamento automaticamente.", "parameters": {"type": "object", "properties": {"novo_data_hora_agendamento": {"type": "string", "description": "Nova data/hora no formato AAAA-MM-DD HH:MM."}}, "required": ["novo_data_hora_agendamento"]}}}]

# ───────────────── 5. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp com OpenRouter e Whisper.", version="3.0.1-fix-audio")
@app.on_event("startup")
async def startup_event(): await asyncio.to_thread(criar_tabelas); print("Tabelas verificadas/criadas.", flush=True)
@app.get("/")
def health_get(): return {"status": "ok"}
@app.head("/")
def health_head(): return Response(status_code=200)

# ───────────────── 6. MODELOS DE PAYLOAD (CORRIGIDO) ────────── #
class ZapiText(BaseModel):
    message: Optional[str] = None
class ZapiAudio(BaseModel): # [NOVO] Modelo para o objeto de áudio
    audioUrl: Optional[str] = None
class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None
    audio: Optional[ZapiAudio] = None # [CORRIGIDO] para espelhar o JSON real

# ───────────────── 7. UTIL ──────────────────────────────────── #
async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"; payload = {"phone": telefone, "message": mensagem}; headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as client:
        try: r = await client.post(url, json=payload, headers=headers); r.raise_for_status()
        except Exception as exc: print("Falha ao enviar para Z-API:", exc, flush=True)

# ───────────────── 8. WEBHOOK (CORRIGIDO) ───────────────────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    raw = await request.json()
    print(">>> PAYLOAD RECEBIDO:", raw, flush=True)
    try: payload = ZapiWebhookPayload(**raw)
    except Exception as e: print("Erro de validação Pydantic:", e, flush=True); raise HTTPException(422, "Formato de payload inválido")

    telefone = payload.phone
    mensagem_usuario = None

    # [CORRIGIDO] Lógica para acessar a URL do áudio corretamente
    if payload.audio and payload.audio.audioUrl:
        print(">>> Recebido áudio. Transcrevendo com Whisper...", flush=True)
        texto_transcrito = await transcrever_audio_whisper(payload.audio.audioUrl)
        if texto_transcrito:
            mensagem_usuario = texto_transcrito
            print(f">>> Texto transcrito: '{texto_transcrito}'", flush=True)
        else:
            await enviar_resposta_whatsapp(telefone, "Desculpe, não consegui entender o seu áudio. Pode tentar novamente ou digitar?")
            return {"status": "erro_transcricao"}
    elif payload.text and payload.text.message:
        mensagem_usuario = payload.text.message

    if not mensagem_usuario: return {"status": "ignorado", "motivo": "sem conteúdo processável"}

    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)); db.commit()
    historico_recente = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= datetime.utcnow() - timedelta(hours=24)).order_by(HistoricoConversa.timestamp).all()
    mensagens_para_ia: List[Dict[str, str]] = [{"role": "system", "content": f"Você é um atendente da 'Clínica Odonto Feliz'. Hoje é {datetime.now().strftime('%d/%m/%Y')}. Use o histórico para ter contexto e as ferramentas para realizar as ações solicitadas."}]
    for msg in historico_recente: mensagens_para_ia.append({"role": msg.role, "content": msg.content})

    try:
        modelo_chat = "google/gemini-2.5-flash-preview-05-20"
        resp = openrouter_chat_completion(model=modelo_chat, messages=mensagens_para_ia, tools=tools, tool_choice="auto")
        ai_msg = resp.choices[0].message
        while ai_msg.tool_calls:
            msgs_com_ferramentas = mensagens_para_ia + [ai_msg]
            for call in ai_msg.tool_calls:
                fname, f_args = call.function.name, json.loads(call.function.arguments)
                func = available_functions.get(fname)
                if not func: raise HTTPException(500, f"Função desconhecida: {fname}")
                result = func(db=db, telefone_paciente=telefone, **f_args)
                msgs_com_ferramentas.append({"tool_call_id": call.id, "role": "tool", "name": fname, "content": result})
            resp = openrouter_chat_completion(model=modelo_chat, messages=msgs_com_ferramentas)
            ai_msg = resp.choices[0].message
        resposta_final = ai_msg.content
    except Exception as e:
        print("Erro na interação IA:", e, flush=True); resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."
    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final)); db.commit()
    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
