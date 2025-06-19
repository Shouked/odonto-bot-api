"""
OdontoBot AI – main.py – v16.1.0-Polished
────────────────────────────────────────────────────────────────────────────
• REFINAMENTO CRÍTICO: Foco na robustez e no fluxo de raciocínio da IA.
• NOVA FERRAMENTA "check_onboarding_status": Adicionada uma ferramenta para
  a IA verificar proativamente se o cadastro do paciente está completo
  ANTES de tentar agendar, tornando o fluxo mais lógico e humano.
• PROMPT DE SISTEMA APRIMORADO: As diretrizes agora instruem a IA a seguir
  uma sequência de verificação (checar cadastro -> pedir dados se necessário
  -> agendar), emulando um raciocínio mais humano.
• ROBUSTEZ NO AGENDAMENTO: A ferramenta 'schedule_appointment' foi
  simplificada para focar apenas em sua tarefa principal.
• ADICIONADO SEEDING DE PROCEDIMENTOS: Incluído o código para popular
  o banco de dados, tornando o script totalmente autocontido.
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
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import (Column, Date, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ───────────────── 2. ENVIRONMENT & CONSTANTS ─────────────
load_dotenv()

# Validação de variáveis de ambiente
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

# Constantes de negócio
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
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
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
            response = await client.get(audio_url)
            response.raise_for_status()
        transcription = await asyncio.to_thread(
            openai_whisper_client.audio.transcriptions.create,
            model="whisper-1", file=("audio.ogg", response.content, "audio/ogg")
        )
        return transcription.text
    except Exception as e:
        print(f"Erro na transcrição de áudio: {e}", flush=True)
        return None

# ───────────────── 4. DATABASE (ORM) ─────────────
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True)
    nome_completo = Column(String); primeiro_nome = Column(String)
    telefone = Column(String, unique=True, nullable=False)
    email = Column(String); data_nascimento = Column(Date)

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora = Column(DateTime(timezone=True), nullable=False)
    procedimento = Column(String, nullable=False)
    status = Column(String, default="confirmado")

class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=get_now)

class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
    categoria = Column(String, index=True)
    descricao = Column(Text)
    valor_descritivo = Column(String, nullable=False)
    valor_base = Column(Float)

engine = create_engine(DATABASE_URL, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def initialize_database(db: Session):
    Base.metadata.create_all(bind=engine)
    if db.query(Procedimento).first(): return
    print("🌱 Populando o banco com procedimentos iniciais...", flush=True)
    procedimentos_data = [
        {"categoria": "Diagnóstico e Prevenção", "nome": "Consulta de Avaliação", "valor_descritivo": "A partir de R$150", "descricao": "Avaliação completa da saúde bucal, diagnóstico e plano de tratamento."},
        {"categoria": "Diagnóstico e Prevenção", "nome": "Limpeza (Profilaxia)", "valor_descritivo": "A partir de R$200", "descricao": "Remoção de placa bacteriana e tártaro para manter a saúde e prevenir doenças."},
        {"categoria": "Radiografias", "nome": "Raio-X Panorâmica", "valor_descritivo": "R$120", "descricao": "Exame de imagem completo que fornece uma visão geral de todos os dentes, maxilares e estruturas adjacentes."},
        {"categoria": "Restaurações", "nome": "Restauração em Resina", "valor_descritivo": "A partir de R$250", "descricao": "Reparo de dentes danificados por cáries ou fraturas, utilizando resina da cor do dente para um resultado estético e funcional."},
        {"categoria": "Endodontia", "nome": "Tratamento de Canal", "valor_descritivo": "Consulte-nos", "descricao": "Tratamento da polpa dentária (nervo) para salvar dentes que de outra forma seriam perdidos."},
        {"categoria": "Estética", "nome": "Clareamento Dental a Laser", "valor_descritivo": "A partir de R$800", "descricao": "Tratamento rápido e eficaz realizado em consultório para dentes mais brancos e um sorriso radiante."}
    ]
    for p_data in procedimentos_data:
        numeros = re.findall(r'\d+', p_data["valor_descritivo"])
        valor_base = float(numeros[0]) if numeros else None
        db.add(Procedimento(nome=p_data["nome"], categoria=p_data["categoria"], descricao=p_data.get("descricao"), valor_descritivo=p_data["valor_descritivo"], valor_base=valor_base))
    db.commit()
    print("✅ Banco de dados populado.", flush=True)

# ───────────────── 5. HELPER FUNCTIONS ─────────────
def find_or_create_patient(db: Session, phone: str) -> Paciente:
    patient = db.query(Paciente).filter_by(telefone=phone).first()
    if not patient:
        patient = Paciente(telefone=phone)
        db.add(patient); db.commit(); db.refresh(patient)
    return patient

# ───────────────── 6. TOOL FUNCTIONS (Polished Architecture) ─────────────
def get_procedure_list(db: Session) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui carregar nossa lista de procedimentos. Nossa equipe pode te ajudar com isso."
    categorias = defaultdict(list)
    for p in procedimentos: categorias[p.categoria].append(p.nome)
    resposta = "Claro! Nós oferecemos uma variedade de serviços para cuidar do seu sorriso. Eles incluem:\n\n"
    for cat, nomes in categorias.items(): resposta += f"*{cat}*\n" + "\n".join(f"- {n}" for n in nomes) + "\n\n"
    return resposta.strip() + "Se quiser saber mais detalhes ou o valor de algum deles, é só me perguntar!"

def get_procedure_details(db: Session, procedure_name: str) -> str:
    resultado = db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedure_name.strip()}%")).first()
    if not resultado: return f"Não encontrei um procedimento chamado '{procedure_name}'. Quer que eu liste todos os nossos serviços?"
    resposta = f"Sobre o procedimento *{resultado.nome}*:\n\n"
    if resultado.descricao: resposta += f"_{resultado.descricao}_\n\n"
    return resposta + f"O valor é: *{resultado.valor_descritivo}*."

def get_available_slots(db: Session, day_str: str) -> str:
    parsed_date = parse_date(day_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_date: return f"Não entendi a data '{day_str}'. Pode tentar 'amanhã' ou '25 de dezembro'?"
    target_date = parsed_date.astimezone(BR_TIMEZONE)
    if target_date.date() < get_now().date(): return "Não podemos verificar horários em datas passadas."
    day_start = target_date.replace(hour=BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
    day_end = target_date.replace(hour=BUSINESS_END_HOUR, minute=0, second=0, microsecond=0)
    booked_slots = {ag.data_hora for ag in db.query(Agendamento.data_hora).filter(Agendamento.data_hora.between(day_start, day_end), Agendamento.status == "confirmado")}
    available_slots = []
    current_slot = day_start
    while current_slot < day_end:
        if current_slot not in booked_slots and current_slot > get_now(): available_slots.append(current_slot)
        current_slot += timedelta(minutes=SLOT_DURATION_MINUTES)
    if not available_slots: return f"Puxa, parece que não temos horários para {target_date.strftime('%d/%m/%Y')}. Gostaria de tentar outra data?"
    return f"Para o dia {target_date.strftime('%d/%m/%Y')}, tenho estes horários disponíveis: *{', '.join(s.strftime('%H:%M') for s in available_slots)}*."

def schedule_appointment(db: Session, patient_id: int, datetime_str: str, procedure: str) -> str:
    parsed_datetime = parse_date(datetime_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
    if not parsed_datetime: return "Não consegui entender a data e hora. Por favor, seja mais específico, como 'amanhã às 10:30'."
    dt_aware = parsed_datetime.astimezone(BR_TIMEZONE)
    if not (time(BUSINESS_START_HOUR) <= dt_aware.time() < time(BUSINESS_END_HOUR)): return f"O horário {dt_aware.strftime('%H:%M')} está fora do nosso horário de funcionamento."
    if db.query(Agendamento).filter_by(data_hora=dt_aware, status="confirmado").first(): return f"Que pena, o horário de {dt_aware.strftime('%d/%m/%Y às %H:%M')} acabou de ser agendado. Poderia escolher outro?"
    patient = db.query(Paciente).get(patient_id)
    new_appointment = Agendamento(paciente_id=patient_id, data_hora=dt_aware, procedimento=procedure, status="confirmado")
    db.add(new_appointment); db.commit()
    return f"Perfeito, {patient.primeiro_nome}! Seu agendamento para *{procedure}* foi confirmado com sucesso para o dia *{dt_aware.strftime('%d/%m/%Y às %H:%M')}*. Você receberá um lembrete um dia antes. Estamos ansiosos para te ver!"

def cancel_appointment(db: Session, patient_id: int) -> str:
    upcoming = db.query(Agendamento).filter(Agendamento.paciente_id == patient_id, Agendamento.status == "confirmado", Agendamento.data_hora > get_now()).order_by(Agendamento.data_hora.asc()).first()
    if not upcoming: return "Verifiquei aqui e não encontrei nenhum agendamento futuro em seu nome."
    details = f"{upcoming.procedimento} no dia {upcoming.data_hora.strftime('%d/%m/%Y às %H:%M')}"
    upcoming.status = "cancelado"; db.commit()
    return f"Entendido. Seu agendamento de *{details}* foi cancelado. Se precisar remarcar, é só me chamar!"

def update_patient_info(db: Session, patient_id: int, full_name: str = None, email: str = None, birth_date_str: str = None) -> str:
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
    return check_onboarding_status(db, patient_id) # Retorna o status atualizado

def check_onboarding_status(db: Session, patient_id: int) -> str:
    patient = db.query(Paciente).get(patient_id)
    missing_info = []
    if not patient.nome_completo: missing_info.append("nome_completo")
    if not patient.email: missing_info.append("email")
    if not patient.data_nascimento: missing_info.append("data_nascimento")
    if not missing_info: return "CADASTRO_COMPLETO"
    return f"CADASTRO_INCOMPLETO. Faltando: {', '.join(missing_info)}."

# ───────────────── 7. APP & WEBHOOK SETUP ─────────────
app = FastAPI(title="OdontoBot AI", version="16.1.0-Polished")

@app.on_event("startup")
def startup_event():
    with SessionLocal() as db: initialize_database(db)
    print(f"🚀 API OdontoBot v{app.version} iniciada com sucesso!", flush=True)

class ZapiText(BaseModel): message: Optional[str] = None
class ZapiAudio(BaseModel): audioUrl: Optional[str] = None
class ZapiPayload(BaseModel): phone: str; text: Optional[ZapiText] = None; audio: Optional[ZapiAudio] = None

AVAILABLE_TOOLS = {"get_procedure_list": get_procedure_list, "get_procedure_details": get_procedure_details, "get_available_slots": get_available_slots, "schedule_appointment": schedule_appointment, "cancel_appointment": cancel_appointment, "update_patient_info": update_patient_info, "check_onboarding_status": check_onboarding_status}
TOOLS_DEFINITION = [
    {"type": "function", "function": {"name": "get_procedure_list", "description": "Quando o usuário perguntar sobre os serviços/tratamentos da clínica."}},
    {"type": "function", "function": {"name": "get_procedure_details", "description": "Para obter detalhes e preço de um procedimento específico.", "parameters": {"type": "object", "properties": {"procedure_name": {"type": "string", "description": "Nome do procedimento. Ex: 'clareamento'."}}, "required": ["procedure_name"]}}},
    {"type": "function", "function": {"name": "get_available_slots", "description": "Para verificar horários/vagas disponíveis em uma data.", "parameters": {"type": "object", "properties": {"day_str": {"type": "string", "description": "A data mencionada. Ex: 'hoje', 'amanhã'."}}, "required": ["day_str"]}}},
    {"type": "function", "function": {"name": "schedule_appointment", "description": "Para CRIAR um agendamento. Use SOMENTE APÓS o usuário ter escolhido data/hora e o cadastro estar completo.", "parameters": {"type": "object", "properties": {"datetime_str": {"type": "string", "description": "Data e hora exatas. Ex: 'amanhã às 15:30'."}, "procedure": {"type": "string", "description": "Procedimento a ser agendado."}}, "required": ["datetime_str", "procedure"]}}},
    {"type": "function", "function": {"name": "cancel_appointment", "description": "Quando o usuário quiser cancelar um agendamento."}},
    {"type": "function", "function": {"name": "update_patient_info", "description": "Quando o usuário fornecer dados pessoais (nome, email, data de nascimento) para o cadastro.", "parameters": {"type": "object", "properties": {"full_name": {"type": "string"}, "email": {"type": "string"}, "birth_date_str": {"type": "string"}}}} },
    {"type": "function", "function": {"name": "check_onboarding_status", "description": "PARA USO INTERNO: Use ANTES de tentar agendar para verificar se o cadastro do paciente está completo."}}
]

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
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

    ## Diretrizes de Raciocínio e Conversação
    1.  **Conduza a Conversa:** Seja proativa. Se um usuário diz "quero agendar", responda "Ótimo! Para qual procedimento e que dia fica bom para você?".
    2.  **Fluxo de Agendamento Inteligente:**
        -   **Passo A (Verificação):** ANTES de mais nada, use a ferramenta `check_onboarding_status` para ver se o cadastro está completo.
        -   **Passo B (Coleta de Dados):** Se o resultado for "CADASTRO_INCOMPLETO", peça a informação que falta de forma natural. Ex: "Para agilizar seu atendimento, preciso só confirmar seu nome completo, por favor.". Use `update_patient_info` para salvar os dados. Peça um dado por vez.
        -   **Passo C (Agendamento):** Se o resultado for "CADASTRO_COMPLETO", prossiga normalmente: verifique horários com `get_available_slots`, e, após a escolha do paciente, confirme com ele ("Posso confirmar *Limpeza* para amanhã às 10h?") e use `schedule_appointment`.
    3.  **Proatividade e Empatia:** Se não houver horários, sugira o próximo dia. Se o paciente cancelar, seja compreensiva ("Sem problemas, imprevistos acontecem. Já cancelei para você.").
    4.  **Regra de Ouro:** **NUNCA INVENTE INFORMAÇÕES.** Se não souber ou uma ferramenta falhar, diga: "Peço desculpas, não consegui verificar essa informação. Nossa equipe entrará em contato para te ajudar."
    """
    history = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).order_by(HistoricoConversa.timestamp.desc()).limit(15).all()
    messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    try:
        final_answer = ""
        for _ in range(5): # Aumentado para 5 iterações para acomodar o fluxo de onboarding
            response = openrouter_chat_completion(model="google/gemini-2.5-flash", messages=messages, tools=TOOLS_DEFINITION, tool_choice="auto")
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

