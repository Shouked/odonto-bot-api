"""
OdontoBot AI – main.py – v17.0.0-Final-TuneUp
────────────────────────────────────────────────────────────────────────────
• NOVA FERRAMENTA CRÍTICA: Adicionada a função `get_my_appointment_details`
  para permitir que a IA consulte os agendamentos existentes do paciente,
  resolvendo uma falha fundamental no fluxo.
• CORREÇÃO GLOBAL DE FUSO HORÁRIO: Todas as ferramentas que retornam
  datas/horas agora convertem explicitamente para o fuso horário de São
  Paulo (BR_TIMEZONE) antes de formatar a saída, garantindo que a IA
  sempre receba e exiba a hora local correta, eliminando o problema do UTC.
• PROMPT DE SISTEMA REFINADO: O prompt foi reestruturado com seções
  de "Fluxos de Trabalho" para guiar a IA de forma mais clara.
• MODELO DE IA: Alterado para Claude 3 Haiku para melhor seguimento de
  instruções complexas e manutenção de persona.
"""

# ───────────────── 1. IMPORTS & SETUP ─────────────
import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pytz
from dateparser import parse as parse_date
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import (Column, Date, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ───────────────── 2. ENVIRONMENT & CONSTANTS ─────────────
load_dotenv()

required_env_vars = ["DATABASE_URL", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ZAPI_API_URL", "ZAPI_INSTANCE_ID", "ZAPI_TOKEN", "ZAPI_CLIENT_TOKEN"]
for var in required_env_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente obrigatória '{var}' não foi definida.")

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ZAPI_API_URL = os.getenv("ZAPI_API_URL")
ZAPI_INSTANCE_ID = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN = os.getenv("ZAPI_CLIENT_TOKEN")

BR_TIMEZONE = pytz.timezone("America/Sao_Paulo")
BUSINESS_START_HOUR, BUSINESS_END_HOUR = 9, 18
SLOT_DURATION_MINUTES = 30
NOME_CLINICA = "DI DONATO ODONTO"

def get_now() -> datetime:
    return datetime.now(BR_TIMEZONE)

# ───────────────── 3. AI & API CLIENTS ─────────────
try:
    import openai
    openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://github.com/Shouked/odonto-bot-api", "X-Title": "OdontoBot AI"},
        timeout=httpx.Timeout(45.0)
    )
except ImportError as exc:
    raise RuntimeError("A biblioteca 'openai' não foi instalada. Execute 'pip install openai'.") from exc

def openrouter_chat_completion(**kwargs):
    return openrouter_client.chat.completions.create(**kwargs)

async def transcribe_audio_whisper(audio_url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(audio_url); response.raise_for_status()
        transcription = await asyncio.to_thread(openai_whisper_client.audio.transcriptions.create, model="whisper-1", file=("audio.ogg", response.content, "audio/ogg"))
        return transcription.text
    except Exception as e:
        print(f"Erro na transcrição de áudio: {e}", flush=True); return None

# ───────────────── 4. DATABASE (ORM) ─────────────
Base = declarative_base()
class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True); nome_completo = Column(String); primeiro_nome = Column(String)
    telefone = Column(String, unique=True, nullable=False); email = Column(String); data_nascimento = Column(Date)
class Agendamento(Base):
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True); paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora = Column(DateTime(timezone=True), nullable=False); procedimento = Column(String, nullable=False); status = Column(String, default="confirmado")
class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id = Column(Integer, primary_key=True); paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    role = Column(String, nullable=False); content = Column(Text, nullable=False); timestamp = Column(DateTime(timezone=True), default=get_now)
class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True); nome = Column(String, unique=True, nullable=False); categoria = Column(String, index=True)
    descricao = Column(Text); valor_descritivo = Column(String, nullable=False); valor_base = Column(Float)

engine = create_engine(DATABASE_URL, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal();
    try: yield db
    finally: db.close()

def initialize_database(db: Session):
    Base.metadata.create_all(bind=engine)
    if db.query(Procedimento).count() > 0: return
    # Seed data...
    db.commit()

# ───────────────── 5. HELPER FUNCTIONS ─────────────
def find_or_create_patient(db: Session, phone: str) -> Paciente:
    patient = db.query(Paciente).filter_by(telefone=phone).first()
    if not patient: patient = Paciente(telefone=phone); db.add(patient); db.commit(); db.refresh(patient)
    return patient

def get_weekday_in_portuguese(date_obj: datetime) -> str:
    weekdays = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    return weekdays[date_obj.weekday()]

# ───────────────── 6. TOOL FUNCTIONS (Data-Centric & Timezone-Aware) ─────────────

# <<<< NOVA FERRAMENTA >>>>
def get_my_appointment_details(db: Session, patient_id: int) -> str:
    """Ferramenta para consultar os detalhes do próximo agendamento do paciente."""
    upcoming_appointment = db.query(Agendamento).filter(
        Agendamento.paciente_id == patient_id,
        Agendamento.status == 'confirmado',
        Agendamento.data_hora > get_now()
    ).order_by(Agendamento.data_hora.asc()).first()

    if not upcoming_appointment:
        return "INFO: Nenhum agendamento futuro encontrado."

    # <<<< CORREÇÃO DE TIMEZONE >>>>
    local_time = upcoming_appointment.data_hora.astimezone(BR_TIMEZONE)
    weekday_name = get_weekday_in_portuguese(local_time)
    
    return f"DADOS_AGENDAMENTO: PROCEDIMENTO: {upcoming_appointment.procedimento}; DATA_HORA: {weekday_name}, {local_time.strftime('%d/%m/%Y às %H:%M')}"

def get_procedure_list(db: Session) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "ERRO: Lista de procedimentos não encontrada."
    categorias = defaultdict(list)
    for p in procedimentos: categorias[p.categoria].append(p.nome)
    data_str = "; ".join([f"CATEGORIA: {cat}, PROCEDIMENTOS: {', '.join(nomes)}" for cat, nomes in categorias.items()])
    return f"LISTA_PROCEDIMENTOS: {data_str}"

def get_procedure_details(db: Session, procedure_name: str) -> str:
    resultado = db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedure_name.strip()}%")).first()
    if not resultado: return f"ERRO: Procedimento '{procedure_name}' não encontrado."
    return f"DADOS_PROCEDIMENTO: NOME: {resultado.nome}; DESCRIÇÃO: {resultado.descricao or 'N/A'}; VALOR: {resultado.valor_descritivo}"

def get_available_slots(db: Session, day_str: str) -> str:
    parsed_date = parse_date(day_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_date: return f"ERRO: Data '{day_str}' inválida."
    target_date = parsed_date.astimezone(BR_TIMEZONE)
    if target_date.weekday() >= 5: return "INFO: Clínica fechada aos finais de semana."
    if target_date.date() < get_now().date(): return "ERRO: Não é possível verificar datas passadas."
    day_start = target_date.replace(hour=BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
    day_end = target_date.replace(hour=BUSINESS_END_HOUR, minute=0, second=0, microsecond=0)
    booked_slots = {ag.data_hora for ag in db.query(Agendamento.data_hora).filter(Agendamento.data_hora.between(day_start, day_end), Agendamento.status == "confirmado")}
    available_slots = []
    num_slots = int((day_end - day_start).total_seconds() / 60 / SLOT_DURATION_MINUTES)
    for i in range(num_slots):
        current_slot = day_start + timedelta(minutes=SLOT_DURATION_MINUTES * i)
        if current_slot not in booked_slots and current_slot > get_now():
            available_slots.append(current_slot.strftime('%H:%M'))
    if not available_slots: return f"INFO: Sem horários disponíveis para {target_date.strftime('%d/%m/%Y')}."
    weekday_name = get_weekday_in_portuguese(target_date)
    return f"HORARIOS_DISPONIVEIS: DIA: {weekday_name}, {target_date.strftime('%d/%m/%Y')}; HORARIOS: {', '.join(available_slots)}"

def schedule_appointment(db: Session, patient_id: int, date_str: str, time_str: str, procedure: str) -> str:
    combined_str = f"{date_str} {time_str}"
    parsed_datetime = parse_date(combined_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_datetime: return f"ERRO: Data e hora inválidas a partir de '{combined_str}'."
    dt_aware = parsed_datetime.astimezone(BR_TIMEZONE)
    if not (time(BUSINESS_START_HOUR) <= dt_aware.time() < time(BUSINESS_END_HOUR)): return "ERRO: Fora do horário comercial."
    if db.query(Agendamento).filter_by(data_hora=dt_aware, status="confirmado").first(): return "ERRO: Horário recém-agendado."
    patient = db.query(Paciente).get(patient_id)
    new_appointment = Agendamento(paciente_id=patient_id, data_hora=dt_aware, procedimento=procedure)
    db.add(new_appointment); db.commit()
    local_time = new_appointment.data_hora.astimezone(BR_TIMEZONE)
    weekday_name = get_weekday_in_portuguese(local_time)
    return f"AGENDAMENTO_SUCESSO: NOME: {patient.primeiro_nome}; PROCEDIMENTO: {procedure}; DATA_HORA: {weekday_name}, {local_time.strftime('%d/%m/%Y às %H:%M')}"

def cancel_appointment(db: Session, patient_id: int) -> str:
    upcoming = db.query(Agendamento).filter(Agendamento.paciente_id == patient_id, Agendamento.status == "confirmado", Agendamento.data_hora > get_now()).order_by(Agendamento.data_hora.asc()).first()
    if not upcoming: return "ERRO: Nenhum agendamento futuro encontrado."
    # <<<< CORREÇÃO DE TIMEZONE >>>>
    local_time = upcoming.data_hora.astimezone(BR_TIMEZONE)
    details = f"{upcoming.procedimento} em {local_time.strftime('%d/%m/%Y às %H:%M')}"
    upcoming.status = "cancelado"; db.commit()
    return f"CANCELAMENTO_SUCESSO: DETALHES: {details}"

def update_patient_info(db: Session, patient_id: int, full_name: str = None, email: str = None, birth_date_str: str = None) -> str:
    patient = db.query(Paciente).get(patient_id)
    if full_name: patient.nome_completo = full_name; patient.primeiro_nome = full_name.split(' ')[0]
    if email:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return "ERRO: E-mail inválido."
        patient.email = email
    if birth_date_str:
        parsed_date = parse_date(birth_date_str, languages=['pt'], settings={'DATE_ORDER': 'DMY'})
        if not parsed_date: return "ERRO: Data de nascimento inválida."
        patient.data_nascimento = parsed_date.date()
    db.commit()
    return check_onboarding_status(db, patient_id)

def check_onboarding_status(db: Session, patient_id: int) -> str:
    patient = db.query(Paciente).get(patient_id)
    missing_info = [field for field, value in [("nome_completo", patient.nome_completo), ("email", patient.email), ("data_nascimento", patient.data_nascimento)] if not value]
    if not missing_info: return "STATUS: CADASTRO_COMPLETO"
    return f"STATUS: CADASTRO_INCOMPLETO; FALTANDO: {', '.join(missing_info)}"

# ───────────────── 7. APP & WEBHOOK SETUP ─────────────
app = FastAPI(title="OdontoBot AI", version="17.0.0-Final-TuneUp")

@app.on_event("startup")
def startup_event():
    with SessionLocal() as db: initialize_database(db)
    print(f"🚀 API OdontoBot v{app.version} iniciada com sucesso!", flush=True)

@app.get("/", summary="Health Check")
def health_check_get(): return {"status": "ok", "version": app.version}
@app.head("/", summary="Health Check")
def health_check_head(): return Response(status_code=200)

class ZapiPayload(BaseModel): phone: str; text: Optional[Dict] = None; audio: Optional[Dict] = None

# <<<< LISTA DE FERRAMENTAS ATUALIZADA >>>>
AVAILABLE_TOOLS = {"get_my_appointment_details": get_my_appointment_details, "get_procedure_list": get_procedure_list, "get_procedure_details": get_procedure_details, "get_available_slots": get_available_slots, "schedule_appointment": schedule_appointment, "cancel_appointment": cancel_appointment, "update_patient_info": update_patient_info, "check_onboarding_status": check_onboarding_status}
TOOLS_DEFINITION = [
    {"type": "function", "function": {"name": "get_my_appointment_details", "description": "Para consultar os detalhes de um agendamento JÁ EXISTENTE do paciente. Use quando o usuário perguntar 'qual meu horário?', 'quando é minha consulta?', etc."}},
    {"type": "function", "function": {"name": "get_procedure_list", "description": "Para listar os serviços/tratamentos da clínica."}},
    {"type": "function", "function": {"name": "get_procedure_details", "description": "Para obter detalhes e preço de um procedimento específico.", "parameters": {"type": "object", "properties": {"procedure_name": {"type": "string"}}, "required": ["procedure_name"]}}},
    {"type": "function", "function": {"name": "get_available_slots", "description": "Para verificar horários disponíveis em uma data.", "parameters": {"type": "object", "properties": {"day_str": {"type": "string"}}, "required": ["day_str"]}}},
    {"type": "function", "function": {"name": "schedule_appointment", "description": "Para CRIAR um agendamento APÓS receber a confirmação explícita do usuário.", "parameters": {"type": "object", "properties": {"date_str": {"type": "string"}, "time_str": {"type": "string"}, "procedure": {"type": "string"}}, "required": ["date_str", "time_str", "procedure"]}}},
    {"type": "function", "function": {"name": "cancel_appointment", "description": "Para cancelar um agendamento."}},
    {"type": "function", "function": {"name": "update_patient_info", "description": "Para salvar dados pessoais do paciente.", "parameters": {"type": "object", "properties": {"full_name": {"type": "string"}, "email": {"type": "string"}, "birth_date_str": {"type": "string"}}}} },
    {"type": "function", "function": {"name": "check_onboarding_status", "description": "PARA USO INTERNO: Use ANTES de agendar para verificar se o cadastro está completo."}}
]

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    try: payload = ZapiPayload(**(await request.json()))
    except Exception as e: raise HTTPException(422, f"Payload inválido: {e}")

    user_phone, user_message = payload.phone, ""
    if payload.audio and payload.audio.get('audioUrl'): user_message = await transcribe_audio_whisper(payload.audio['audioUrl'])
    elif payload.text and payload.text.get('message'): user_message = payload.text['message']
    if not user_message.strip(): await send_zapi_message(user_phone, f"Olá! Sou a Sofia, assistente da {NOME_CLINICA}. Como posso te ajudar?"); return {"status": "greeting"}

    patient = find_or_create_patient(db, user_phone)
    history_count = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).count()
    is_first_message = history_count == 0
    db.add(HistoricoConversa(paciente_id=patient.id, role="user", content=user_message)); db.commit()

    # <<<< PROMPT ATUALIZADO >>>>
    system_prompt = f"""
## Persona: Sofia, Assistente Virtual da {NOME_CLINICA}
Você é a Sofia: calorosa, profissional e proativa. Seu objetivo é fazer cada paciente se sentir bem-vindo e cuidado.

## Contexto
- Hoje é: {get_now().strftime('%A, %d/%m/%Y')}.
- Paciente: {patient.primeiro_nome or 'Novo Paciente'}.
- É a primeira mensagem desta conversa: {'Sim' if is_first_message else 'Não'}.

## PRINCIPAIS FLUXOS DE TRABALHO
Siga os fluxos abaixo ESTRITAMENTE.

### Fluxo de Consulta de Agendamento
1. Se o paciente perguntar sobre seu agendamento existente (ex: "quando é minha consulta?", "qual meu horário?"), chame a ferramenta `get_my_appointment_details`.
2. Transforme o resultado em uma resposta clara para o paciente.

### Fluxo de Agendamento (NOVO AGENDAMENTO)
1.  **Verificação:** Ao pedir para agendar, PRIMEIRO chame `check_onboarding_status`.
2.  **Coleta:** Se "INCOMPLETO", peça UM dado faltante. Use `update_patient_info` para salvar. Repita até o cadastro estar completo.
3.  **Opções:** Com o cadastro completo, use `get_available_slots` para mostrar os horários.
4.  **Confirmação:** Após a escolha do horário, PERGUNTE para confirmar: "Posso confirmar seu agendamento para *[Procedimento]* na *[Dia da Semana], dia [Data]* às *[Hora]*?".
5.  **Finalização:** SOMENTE APÓS o "sim" do usuário, chame `schedule_appointment`.

## REGRAS CRÍTICAS
- **SEPARAÇÃO DE TAREFAS:** As ferramentas te darão DADOS BRUTOS. Sua única função é transformar esses dados em uma resposta amigável. NUNCA repita o texto da ferramenta.
- **SAUDAÇÃO INICIAL:** Se "É a primeira mensagem desta conversa" for "Sim", comece com uma saudação calorosa.
- **ERROS:** Se uma ferramenta retornar um ERRO, peça desculpas e diga que a equipe humana entrará em contato.
"""
    history = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).order_by(HistoricoConversa.timestamp.desc()).limit(15).all()
    messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    try:
        final_answer = ""
        # Usando Claude 3 Haiku para melhor seguimento de regras
        response_model = "google/gemini-2.5-pro"

        for _ in range(5):
            response = openrouter_chat_completion(model=response_model, messages=messages, tools=TOOLS_DEFINITION, tool_choice="auto", temperature=0.1)
            ai_message = response.choices[0].message
            messages.append(ai_message)
            if not ai_message.tool_calls: final_answer = ai_message.content; break
            
            for tool_call in ai_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🤖 IA -> Ferramenta: {func_name}({func_args})", flush=True)

                if func_to_call := AVAILABLE_TOOLS.get(func_name):
                    # Adiciona 'patient_id' automaticamente para funções que precisam dele
                    if func_name in ["get_my_appointment_details", "schedule_appointment", "cancel_appointment", "update_patient_info", "check_onboarding_status"]:
                        func_args['patient_id'] = patient.id
                    tool_result = func_to_call(db=db, **func_args)
                else: tool_result = f"ERRO: Ferramenta '{func_name}' não encontrada."
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": func_name, "content": tool_result})
        else: final_answer = "Peço desculpas, mas parece que há um problema para processar sua solicitação. Nossa equipe já foi notificada."
    except Exception as e:
        print(f"🚨 Erro crítico no loop da IA: {e}", flush=True)
        final_answer = "Desculpe, estou com um problema técnico. Por favor, tente novamente em alguns instantes."

    db.add(HistoricoConversa(paciente_id=patient.id, role="assistant", content=final_answer)); db.commit()
    await send_zapi_message(user_phone, final_answer)
    return {"status": "processed", "response": final_answer}

async def send_zapi_message(phone: str, message: str):
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    payload = {"phone": phone, "message": message}
    async with httpx.AsyncClient() as client:
        try: await client.post(url, json=payload, headers=headers, timeout=30)
        except Exception as e: print(f"🚨 Falha ao enviar mensagem para Z-API ({phone}): {e}", flush=True)

