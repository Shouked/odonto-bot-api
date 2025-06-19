"""
OdontoBot AI – main.py – v16.2.0-Clarity-Update
────────────────────────────────────────────────────────────────────────────
• CORREÇÃO DE DATAS: A ferramenta `get_available_slots` agora inclui o dia
  da semana em português na sua resposta, eliminando alucinações da IA.
• BLOQUEIO DE FINAIS DE SEMANA: Adicionada lógica para impedir a verificação
  de horários aos sábados e domingos, informando o usuário sobre o horário
  de funcionamento.
• FLUXO DE CONFIRMAÇÃO OBRIGATÓRIO: O prompt de sistema foi reforçado para
  FORÇAR a IA a pedir a confirmação do usuário (com todos os detalhes do
  agendamento) antes de efetivamente criar o agendamento.
• SAUDAÇÃO APRIMORADA: Adicionada regra no prompt para garantir uma
  apresentação calorosa e adequada a novos pacientes.
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
    if db.query(Procedimento).first(): return
    # Seed data...
    db.commit()

# ───────────────── 5. HELPER FUNCTIONS ─────────────
def find_or_create_patient(db: Session, phone: str) -> Paciente:
    patient = db.query(Paciente).filter_by(telefone=phone).first()
    if not patient: patient = Paciente(telefone=phone); db.add(patient); db.commit(); db.refresh(patient)
    return patient

def get_weekday_in_portuguese(date_obj: datetime) -> str:
    """Traduz o dia da semana para português."""
    weekdays = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    return weekdays[date_obj.weekday()]

# ───────────────── 6. TOOL FUNCTIONS (Polished Architecture) ─────────────
def get_procedure_list(db: Session) -> str:
    # ... (código da função inalterado)
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui carregar nossa lista de procedimentos. Nossa equipe pode te ajudar."
    categorias = defaultdict(list)
    for p in procedimentos: categorias[p.categoria].append(p.nome)
    resposta = "Claro! Nós oferecemos uma variedade de serviços para cuidar do seu sorriso. Eles incluem:\n\n"
    for cat, nomes in categorias.items(): resposta += f"*{cat}*\n" + "\n".join(f"- {n}" for n in nomes) + "\n\n"
    return resposta.strip() + "Se quiser saber mais detalhes ou o valor de algum deles, é só me perguntar!"

def get_procedure_details(db: Session, procedure_name: str) -> str:
    # ... (código da função inalterado)
    resultado = db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedure_name.strip()}%")).first()
    if not resultado: return f"Não encontrei um procedimento chamado '{procedure_name}'. Quer que eu liste todos os nossos serviços?"
    resposta = f"Sobre o procedimento *{resultado.nome}*:\n\n";
    if resultado.descricao: resposta += f"_{resultado.descricao}_\n\n"
    return resposta + f"O valor é: *{resultado.valor_descritivo}*."

def get_available_slots(db: Session, day_str: str) -> str:
    """Verifica e retorna os horários de agendamento livres em uma data, com validação de final de semana."""
    parsed_date = parse_date(day_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_date: return f"Não consegui entender a data '{day_str}'. Pode tentar 'amanhã' ou '25 de dezembro'?"
    
    target_date = parsed_date.astimezone(BR_TIMEZONE)

    # <<<< NOVA VALIDAÇÃO DE FINAL DE SEMANA >>>>
    if target_date.weekday() >= 5: # 5 é Sábado, 6 é Domingo
        return "Nossa clínica não abre aos sábados e domingos. Por favor, escolha um dia de segunda a sexta-feira."

    if target_date.date() < get_now().date(): return "Não podemos verificar horários em datas passadas."
    
    day_start = target_date.replace(hour=BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
    day_end = target_date.replace(hour=BUSINESS_END_HOUR, minute=0, second=0, microsecond=0)
    booked_slots = {ag.data_hora for ag in db.query(Agendamento.data_hora).filter(Agendamento.data_hora.between(day_start, day_end), Agendamento.status == "confirmado")}
    
    available_slots = []
    current_slot = day_start
    while current_slot < day_end:
        if current_slot not in booked_slots and current_slot > get_now(): available_slots.append(current_slot)
        current_slot += timedelta(minutes=SLOT_DURATION_MINUTES)
    
    if not available_slots: return f"Puxa, parece que não temos mais horários para {target_date.strftime('%d/%m/%Y')}. Gostaria de tentar outra data?"
    
    # <<<< NOVA RESPOSTA COM DIA DA SEMANA INCLUÍDO >>>>
    weekday_name = get_weekday_in_portuguese(target_date)
    return f"Para *{weekday_name}, dia {target_date.strftime('%d/%m/%Y')}*, tenho estes horários disponíveis: *{', '.join(s.strftime('%H:%M') for s in available_slots)}*."

def schedule_appointment(db: Session, patient_id: int, datetime_str: str, procedure: str) -> str:
    # ... (código da função inalterado)
    parsed_datetime = parse_date(datetime_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_datetime: return "Não consegui entender a data e hora. Por favor, seja mais específico, como 'amanhã às 10:30'."
    dt_aware = parsed_datetime.astimezone(BR_TIMEZONE)
    if not (time(BUSINESS_START_HOUR) <= dt_aware.time() < time(BUSINESS_END_HOUR)): return f"O horário {dt_aware.strftime('%H:%M')} está fora do nosso horário de funcionamento."
    if db.query(Agendamento).filter_by(data_hora=dt_aware, status="confirmado").first(): return f"Que pena, o horário de {dt_aware.strftime('%d/%m/%Y às %H:%M')} acabou de ser agendado. Poderia escolher outro?"
    patient = db.query(Paciente).get(patient_id)
    new_appointment = Agendamento(paciente_id=patient_id, data_hora=dt_aware, procedimento=procedure, status="confirmado")
    db.add(new_appointment); db.commit()
    weekday_name = get_weekday_in_portuguese(dt_aware)
    return f"Perfeito, {patient.primeiro_nome}! Seu agendamento para *{procedure}* foi confirmado com sucesso para *{weekday_name}, dia {dt_aware.strftime('%d/%m/%Y às %H:%M')}*. Você receberá um lembrete. Até lá!"

def cancel_appointment(db: Session, patient_id: int) -> str:
    # ... (código da função inalterado)
    upcoming = db.query(Agendamento).filter(Agendamento.paciente_id == patient_id, Agendamento.status == "confirmado", Agendamento.data_hora > get_now()).order_by(Agendamento.data_hora.asc()).first()
    if not upcoming: return "Verifiquei aqui e não encontrei nenhum agendamento futuro em seu nome."
    details = f"{upcoming.procedimento} no dia {upcoming.data_hora.strftime('%d/%m/%Y às %H:%M')}"
    upcoming.status = "cancelado"; db.commit()
    return f"Entendido. Seu agendamento de *{details}* foi cancelado. Se precisar remarcar, é só me chamar!"

def update_patient_info(db: Session, patient_id: int, full_name: str = None, email: str = None, birth_date_str: str = None) -> str:
    # ... (código da função inalterado)
    patient = db.query(Paciente).get(patient_id)
    if full_name: patient.nome_completo = full_name; patient.primeiro_nome = full_name.split(' ')[0]
    if email:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return "Este e-mail não parece válido. Pode verificar e me informar novamente?"
        patient.email = email
    if birth_date_str:
        parsed_date = parse_date(birth_date_str, languages=['pt'], settings={'DATE_ORDER': 'DMY'})
        if not parsed_date: return "Não entendi essa data. Poderia me informar no formato DD/MM/AAAA?"
        patient.data_nascimento = parsed_date.date()
    db.commit()
    return check_onboarding_status(db, patient_id)

def check_onboarding_status(db: Session, patient_id: int) -> str:
    # ... (código da função inalterado)
    patient = db.query(Paciente).get(patient_id)
    missing_info = []
    if not patient.nome_completo: missing_info.append("nome_completo")
    if not patient.email: missing_info.append("email")
    if not patient.data_nascimento: missing_info.append("data_nascimento")
    if not missing_info: return "CADASTRO_COMPLETO"
    return f"CADASTRO_INCOMPLETO. Faltando: {', '.join(missing_info)}."

# ───────────────── 7. APP & WEBHOOK SETUP ─────────────
app = FastAPI(title="OdontoBot AI", version="16.2.0-Clarity-Update")

@app.on_event("startup")
def startup_event():
    with SessionLocal() as db: initialize_database(db)
    print(f"🚀 API OdontoBot v{app.version} iniciada com sucesso!", flush=True)

@app.get("/", summary="Health Check")
def health_check_get(): return {"status": "ok", "version": app.version}
@app.head("/", summary="Health Check")
def health_check_head(): return Response(status_code=200)

class ZapiText(BaseModel): message: Optional[str] = None
class ZapiAudio(BaseModel): audioUrl: Optional[str] = None
class ZapiPayload(BaseModel): phone: str; text: Optional[ZapiText] = None; audio: Optional[ZapiAudio] = None

AVAILABLE_TOOLS = {"get_procedure_list": get_procedure_list, "get_procedure_details": get_procedure_details, "get_available_slots": get_available_slots, "schedule_appointment": schedule_appointment, "cancel_appointment": cancel_appointment, "update_patient_info": update_patient_info, "check_onboarding_status": check_onboarding_status}
TOOLS_DEFINITION = [
    # ... (Definições de ferramentas inalteradas)
    {"type": "function", "function": {"name": "get_procedure_list", "description": "Quando o usuário perguntar sobre os serviços/tratamentos da clínica."}},
    {"type": "function", "function": {"name": "get_procedure_details", "description": "Para obter detalhes e preço de um procedimento específico.", "parameters": {"type": "object", "properties": {"procedure_name": {"type": "string", "description": "Nome do procedimento. Ex: 'clareamento'."}}, "required": ["procedure_name"]}}},
    {"type": "function", "function": {"name": "get_available_slots", "description": "Para verificar horários/vagas disponíveis em uma data.", "parameters": {"type": "object", "properties": {"day_str": {"type": "string", "description": "A data mencionada. Ex: 'hoje', 'amanhã'."}}, "required": ["day_str"]}}},
    {"type": "function", "function": {"name": "schedule_appointment", "description": "Use para CRIAR o agendamento. Chame esta função SOMENTE APÓS ter apresentado todos os detalhes ao usuário e recebido sua confirmação explícita.", "parameters": {"type": "object", "properties": {"datetime_str": {"type": "string", "description": "Data e hora exatas confirmadas. Ex: 'amanhã às 15:30'."}, "procedure": {"type": "string", "description": "Procedimento a ser agendado."}}, "required": ["datetime_str", "procedure"]}}},
    {"type": "function", "function": {"name": "cancel_appointment", "description": "Quando o usuário quiser cancelar um agendamento."}},
    {"type": "function", "function": {"name": "update_patient_info", "description": "Quando o usuário fornecer dados pessoais (nome, email, data de nascimento) para o cadastro.", "parameters": {"type": "object", "properties": {"full_name": {"type": "string"}, "email": {"type": "string"}, "birth_date_str": {"type": "string"}}}} },
    {"type": "function", "function": {"name": "check_onboarding_status", "description": "PARA USO INTERNO: Use ANTES de tentar agendar para verificar se o cadastro do paciente está completo."}}
]

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    # ... (lógica do webhook inalterada, o prompt a seguir é a mudança principal)
    try: payload = ZapiPayload(**(await request.json()))
    except Exception as e: raise HTTPException(422, f"Payload inválido: {e}")

    user_phone, user_message = payload.phone, ""
    if payload.audio and payload.audio.audioUrl:
        user_message = await transcribe_audio_whisper(payload.audio.audioUrl)
        if not user_message: await send_zapi_message(user_phone, "Não consegui processar seu áudio. Pode tentar de novo ou mandar por texto?"); return {"status": "audio_error"}
    elif payload.text and payload.text.message: user_message = payload.text.message
    if not user_message.strip(): await send_zapi_message(user_phone, f"Olá! Sou a Sofia, da {NOME_CLINICA}. Como posso te ajudar?"); return {"status": "greeting"}

    patient = find_or_create_patient(db, user_phone)
    db.add(HistoricoConversa(paciente_id=patient.id, role="user", content=user_message)); db.commit()

    system_prompt = f"""
    ## Persona: Sofia, Assistente Virtual da {NOME_CLINICA}
    Você é a Sofia. Sua personalidade é calorosa, profissional, empática e proativa. Seu objetivo é fazer cada paciente se sentir bem-vindo e cuidado.

    ## Contexto
    - Hoje é: {get_now().strftime('%A, %d/%m/%Y')}.
    - Atendimento: Segunda a Sexta, das {BUSINESS_START_HOUR}h às {BUSINESS_END_HOUR}h.
    - Paciente: {patient.primeiro_nome or 'Novo Paciente'} (Tel: {user_phone}).

    ## Diretrizes de Conversação
    1.  **Saudação para Novos Pacientes:** Se o campo `Paciente:` indica 'Novo Paciente', sua primeira resposta DEVE incluir uma saudação calorosa de boas-vindas antes de qualquer outra coisa. Ex: "Olá! Bem-vindo(a) à {NOME_CLINICA}. Sou a Sofia, sua assistente virtual. Vi que você gostaria de..."
    2.  **Conduza a Conversa:** Seja proativa. Se um usuário diz "quero agendar", responda "Ótimo! Para qual procedimento e que dia fica bom para você?".
    3.  **Proatividade e Empatia:** Se não houver horários, sugira o próximo dia. Se o paciente cancelar, seja compreensiva ("Sem problemas, imprevistos acontecem. Já cancelei para você.").

    ### FLUXO DE AGENDAMENTO OBRIGATÓRIO
    Este fluxo é mandatório. Siga os passos na ordem correta.
    1.  **Verificar Cadastro:** Quando o usuário demonstrar interesse em agendar, sua PRIMEIRA ação é usar a ferramenta `check_onboarding_status`.
    2.  **Coletar Dados (se necessário):** Se o status for "CADASTRO_INCOMPLETO", você DEVE pedir a informação que falta de forma natural. Ex: "Para agendarmos, preciso só que me informe seu nome completo, por favor.". Use `update_patient_info` para salvar cada dado. Peça um dado por vez.
    3.  **Apresentar Opções:** Com o cadastro completo, use `get_available_slots` para mostrar os horários.
    4.  **CONFIRMAR ANTES DE AGENDAR:** Após o usuário escolher um horário, você é OBRIGADA a fazer uma pergunta de confirmação, repetindo todos os detalhes. Ex: "Perfeito! Posso confirmar seu agendamento para *[Procedimento]* na *[Dia da Semana], dia [Data]* às *[Hora]*?".
    5.  **Agendamento Final:** SOMENTE APÓS a resposta afirmativa do usuário ("sim", "ok", "pode confirmar", etc.), você deve chamar a ferramenta `schedule_appointment`.

    ## Regra de Ouro
    **NUNCA INVENTE INFORMAÇÕES.** Se não souber ou uma ferramenta falhar, diga: "Peço desculpas, não consegui verificar essa informação. Nossa equipe entrará em contato para te ajudar."
    """
    history = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).order_by(HistoricoConversa.timestamp.desc()).limit(15).all()
    messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    try:
        final_answer = ""
        for _ in range(5):
            response = openrouter_chat_completion(model="google/gemini-2.5-pro", messages=messages, tools=TOOLS_DEFINITION, tool_choice="auto")
            ai_message = response.choices[0].message
            messages.append(ai_message)
            if not ai_message.tool_calls: final_answer = ai_message.content; break
            
            for tool_call in ai_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🤖 IA -> Ferramenta: {func_name}({func_args})", flush=True)

                if func_to_call := AVAILABLE_TOOLS.get(func_name):
                    if func_name in ["schedule_appointment", "cancel_appointment", "update_patient_info", "check_onboarding_status"]:
                        func_args['patient_id'] = patient.id
                    tool_result = func_to_call(db=db, **func_args)
                else:
                    tool_result = f"Erro: Ferramenta '{func_name}' não encontrada."
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
