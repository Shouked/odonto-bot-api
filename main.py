# OdontoBot AI – main.py – v16.3.3-FlowControl
────────────────────────────────────────────────────────────────────────────
• CONTROLE DE FLUXO (CRÍTICO): O prompt do sistema foi completamente reescrito para 
  garantir que a IA siga estritamente o fluxo de agendamento e nunca pule etapas.
• EXEMPLOS CONCRETOS: Adicionados exemplos de diálogo completo mostrando como 
  a IA deve interagir em cada etapa do fluxo de agendamento.
• PARÂMETROS DE MODELO: Ajustada a temperatura do modelo para 0.2 para aumentar
  a probabilidade de seguir as regras definidas.
• DESCRIÇÕES MAIS CLARAS: Melhoradas as descrições das ferramentas para enfatizar
  a necessidade de confirmação explícita antes de fazer agendamentos.

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

# ───────────────── 6. TOOL FUNCTIONS (Data-Centric Architecture) ─────────────
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
    """Ferramenta para criar agendamento a partir de data e hora separadas. Retorna confirmação ou erro."""
    combined_str = f"{date_str} {time_str}"
    # Tenta um parsing mais estrito primeiro
    parsed_datetime = parse_date(combined_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future", "STRICT_PARSING": True})
    if not parsed_datetime:
        # Fallback para parsing mais flexível
        parsed_datetime = parse_date(combined_str, languages=['pt'], settings={"PREFER_DATES_FROM": "future"})
        if not parsed_datetime:
            return f"ERRO: Data e hora inválidas a partir de '{combined_str}'."

    dt_aware = parsed_datetime.astimezone(BR_TIMEZONE)
    if not (time(BUSINESS_START_HOUR) <= dt_aware.time() < time(BUSINESS_END_HOUR)): return "ERRO: Fora do horário comercial."
    if db.query(Agendamento).filter_by(data_hora=dt_aware, status="confirmado").first(): return "ERRO: Horário recém-agendado."
    
    patient = db.query(Paciente).get(patient_id)
    new_appointment = Agendamento(paciente_id=patient_id, data_hora=dt_aware, procedimento=procedure)
    db.add(new_appointment); db.commit()
    
    weekday_name = get_weekday_in_portuguese(dt_aware)
    return f"AGENDAMENTO_SUCESSO: NOME: {patient.primeiro_nome}; PROCEDIMENTO: {procedure}; DATA_HORA: {weekday_name}, {dt_aware.strftime('%d/%m/%Y às %H:%M')}"

def cancel_appointment(db: Session, patient_id: int) -> str:
    upcoming = db.query(Agendamento).filter(Agendamento.paciente_id == patient_id, Agendamento.status == "confirmado", Agendamento.data_hora > get_now()).order_by(Agendamento.data_hora.asc()).first()
    if not upcoming: return "ERRO: Nenhum agendamento futuro encontrado."
    details = f"{upcoming.procedimento} em {upcoming.data_hora.strftime('%d/%m/%Y às %H:%M')}"
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
app = FastAPI(title="OdontoBot AI", version="16.3.3-FlowControl")

@app.on_event("startup")
def startup_event():
    with SessionLocal() as db: initialize_database(db)
    print(f"🚀 API OdontoBot v{app.version} iniciada com sucesso!", flush=True)

@app.get("/", summary="Health Check")
def health_check_get(): return {"status": "ok", "version": app.version}
@app.head("/", summary="Health Check")
def health_check_head(): return Response(status_code=200)

class ZapiPayload(BaseModel): phone: str; text: Optional[Dict] = None; audio: Optional[Dict] = None

AVAILABLE_TOOLS = {"get_procedure_list": get_procedure_list, "get_procedure_details": get_procedure_details, "get_available_slots": get_available_slots, "schedule_appointment": schedule_appointment, "cancel_appointment": cancel_appointment, "update_patient_info": update_patient_info, "check_onboarding_status": check_onboarding_status}

# <<<< DEFINIÇÃO DE FERRAMENTAS ATUALIZADA >>>>
TOOLS_DEFINITION = [
    {"type": "function", "function": {"name": "get_procedure_list", "description": "Para listar os serviços/tratamentos da clínica."}},
    {"type": "function", "function": {"name": "get_procedure_details", "description": "Para obter detalhes e preço de um procedimento específico.", "parameters": {"type": "object", "properties": {"procedure_name": {"type": "string"}}, "required": ["procedure_name"]}}},
    {"type": "function", "function": {"name": "get_available_slots", "description": "Para verificar horários disponíveis em uma data.", "parameters": {"type": "object", "properties": {"day_str": {"type": "string"}}, "required": ["day_str"]}}},
    {"type": "function", "function": {
        "name": "schedule_appointment",
        "description": "SOMENTE use esta ferramenta APÓS receber confirmação EXPLÍCITA (sim/confirme) do usuário para criar o agendamento. NUNCA chame esta função sem ter perguntado e recebido confirmação do horário escolhido.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_str": {"type": "string", "description": "A DATA do agendamento EXATAMENTE como discutida na conversa, como 'próxima segunda-feira' ou '23 de junho'."},
                "time_str": {"type": "string", "description": "O HORÁRIO exato do agendamento no formato HH:MM, como '09:00' ou '15:30'."},
                "procedure": {"type": "string", "description": "O nome EXATO do procedimento como retornado pelas ferramentas anteriores, sem modificações."}
            },
            "required": ["date_str", "time_str", "procedure"]
        }
    }},
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
    if not user_message: await send_zapi_message(user_phone, f"Olá! Sou a Sofia, da {NOME_CLINICA}. Como posso te ajudar?"); return {"status": "greeting"}

    patient = find_or_create_patient(db, user_phone)
    history_count = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).count()
    is_first_message = history_count == 0
    db.add(HistoricoConversa(paciente_id=patient.id, role="user", content=user_message)); db.commit()

    # <<<< PROMPT ATUALIZADO >>>>
    system_prompt = f'''
## Persona: Sofia, Assistente Virtual da {NOME_CLINICA}
Você é a Sofia: calorosa, profissional e proativa. Seu objetivo é fazer cada paciente se sentir bem-vindo e cuidado.

## Contexto
- Hoje é: {get_now().strftime('%A, %d/%m/%Y')}.
- Paciente: {patient.primeiro_nome or 'Novo Paciente'}.
- É a primeira mensagem desta conversa: {'Sim' if is_first_message else 'Não'}.

## FLUXO DE AGENDAMENTO (OBRIGATÓRIO)
Você DEVE seguir este fluxo ESTRITAMENTE na ordem abaixo para fazer agendamentos:

1️⃣ VERIFICAÇÃO: 
   - Quando o paciente pedir para agendar, PRIMEIRO chame a ferramenta `check_onboarding_status`.
   - NUNCA pule esta etapa.

2️⃣ COLETA DE DADOS (SE NECESSÁRIO): 
   - Se o status for "CADASTRO_INCOMPLETO", você DEVE:
     - Perguntar por UM dos dados faltantes mencionados na resposta (nome_completo, email ou data_nascimento)
     - Chamar `update_patient_info` para salvar SOMENTE após receber a resposta
     - Repetir até que todos os dados estejam completos
   - NUNCA pergunte dois dados ao mesmo tempo.

3️⃣ OFERECER PROCEDIMENTOS (SE CADASTRO COMPLETO):
   - PRIMEIRO chame `get_procedure_list` para mostrar opções disponíveis
   - Peça ao usuário que escolha um procedimento
   - Após a escolha, use `get_procedure_details` para confirmar detalhes

4️⃣ OFERECER HORÁRIOS:
   - SOMENTE após escolher o procedimento, chame `get_available_slots` para uma data
   - Se não houver horários, ofereça outras datas
   - Aguarde até o paciente escolher um horário específico

5️⃣ CONFIRMAÇÃO EXPLÍCITA:
   - Após a escolha do horário, você DEVE perguntar EXATAMENTE:
     "Posso confirmar seu agendamento para [Procedimento] na [Dia da Semana], dia [Data] às [Hora]?"
   - AGUARDE a confirmação explícita do usuário ("sim", "confirme", "pode agendar", etc.)
   - NUNCA prossiga sem essa confirmação!

6️⃣ FINALIZAÇÃO:
   - SOMENTE APÓS o "sim" explícito do usuário, chame `schedule_appointment`
   - Use os parâmetros exatos:
     - `date_str`: data escolhida (ex: "próxima segunda-feira", "23 de junho")
     - `time_str`: horário no formato HH:MM exato (ex: "09:00", "15:30")
     - `procedure`: nome exato do procedimento como retornado pelas ferramentas

## REGRAS CRÍTICAS
- SEPARAÇÃO: As ferramentas te darão DADOS BRUTOS. Transforme-os em resposta amigável e natural. NUNCA repita o texto da ferramenta.
- SAUDAÇÃO: Se "É a primeira mensagem desta conversa" for "Sim", comece com uma saudação calorosa.
- NOMES EXATOS: Use SEMPRE o nome exato do procedimento como retornado pelas ferramentas. Não modifique ou combine nomes.
- ERROS: Se uma ferramenta retornar um ERRO, peça desculpas e diga que a equipe humana entrará em contato.
- SEQUÊNCIA: NUNCA pule etapas do fluxo de agendamento. Isso é CRÍTICO!

## EXEMPLOS DO FLUXO DE AGENDAMENTO
### Exemplo 1 - Verificação e coleta
Paciente: "Quero agendar uma consulta"
Assistente: [chama check_onboarding_status]
Sistema: "STATUS: CADASTRO_INCOMPLETO; FALTANDO: nome_completo, email"
Assistente: "Claro! Precisamos completar seu cadastro. Qual é o seu nome completo?"
Paciente: "Maria Silva Santos"
Assistente: [chama update_patient_info com full_name="Maria Silva Santos"]
...

### Exemplo 2 - Confirmação e agendamento
Paciente: "Quero o horário de 14:30"
Assistente: "Posso confirmar seu agendamento para Limpeza na Quarta-feira, dia 23/06 às 14:30?"
Paciente: "Sim, pode agendar"
Assistente: [chama schedule_appointment com date_str="23 de junho", time_str="14:30", procedure="Limpeza"]
Sistema: "AGENDAMENTO_SUCESSO: NOME: Maria; PROCEDIMENTO: Limpeza; DATA_HORA: Quarta-feira, 23/06/2025 às 14:30"
Assistente: "Ótimo, Maria! Seu agendamento para Limpeza foi confirmado para quarta-feira, 23 de junho, às 14:30. Esperamos você em nossa clínica!"

## VERIFICAÇÃO FINAL ANTES DE CHAMAR FUNÇÕES:
Antes de chamar qualquer função, verifique:
1. Você está seguindo a ordem correta do fluxo?
2. Você completou a etapa anterior completamente?
3. Para `schedule_appointment`: o usuário deu confirmação EXPLÍCITA?
'''
    history = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).order_by(HistoricoConversa.timestamp.desc()).limit(15).all()
    messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    try:
        final_answer = ""
        for _ in range(5):
            response = openrouter_chat_completion(
                model="google/gemini-2.5-flash", 
                messages=messages, 
                tools=TOOLS_DEFINITION, 
                tool_choice="auto",
                temperature=0.2  # Valor mais baixo para seguir regras com mais precisão
            )
            ai_message = response.choices[0].message
            messages.append(ai_message)
            if not ai_message.tool_calls: final_answer = ai_message.content; break
            
            for tool_call in ai_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🤖 IA -> Ferramenta: {func_name}({func_args})", flush=True)

                if func_to_call := AVAILABLE_TOOLS.get(func_name):
                    if func_name in ["schedule_appointment", "cancel_appointment", "update_patient_info", "check_onboarding_status"]: func_args['patient_id'] = patient.id
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
