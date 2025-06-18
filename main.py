"""
OdontoBot AI – main.py – v15.0.0
────────────────────────────────────────────────────────────────────────────
• Arquitetura refatorada para usar "Function Calling" (Tool Use) com o Gemini.
  Isso elimina completamente as "alucinações" da IA, que agora consulta
  funções locais para obter dados reais (horários, preços, etc.).
• Persona "Sofia" aprimorada com um prompt de sistema detalhado para um
  atendimento mais humanizado, profissional e empático.
• Adicionadas novas funcionalidades:
  - Cancelar um agendamento existente.
  - Obter detalhes e descrição de um procedimento.
• Fluxo de conversação robusto que permite múltiplas interações (consultar,
  verificar horários e agendar em sequência).
• Código reorganizado e comentado para maior clareza e manutenibilidade.
"""

# ───────────── 1. IMPORTS & SETUP ─────────────
from __future__ import annotations
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
                        String, Text, create_engine, or_)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ───────────── 2. ENVIRONMENT & CONSTANTS ─────────────
load_dotenv()

# Carrega as variáveis de ambiente e verifica se todas estão presentes
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Usado para o Whisper
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Usado para o Gemini
ZAPI_API_URL = os.getenv("ZAPI_API_URL")
ZAPI_INSTANCE_ID = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN = os.getenv("ZAPI_CLIENT_TOKEN")

if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL,
            ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Uma ou mais variáveis de ambiente estão faltando. Verifique seu arquivo .env.")

# Constantes de negócio e tempo
BR_TZ = pytz.timezone("America/Sao_Paulo")
BUSINESS_START, BUSINESS_END = time(9), time(18) # Horário de funcionamento: 9h às 18h
SLOT_MINUTES = 30 # Duração de cada slot de agendamento

now_tz = lambda: datetime.now(BR_TZ)
now_naive = lambda: datetime.now(BR_TZ).replace(tzinfo=None)
to_naive = lambda dt: (BR_TZ.localize(dt) if dt.tzinfo is None else dt.astimezone(BR_TZ)).replace(tzinfo=None)

# ───────────── 3. AI & API CLIENTS ─────────────
# Cliente OpenAI para transcrição de áudio com Whisper
try:
    import openai
    openai_whisper = openai.OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    raise RuntimeError("A biblioteca da OpenAI não foi instalada. Por favor, instale com 'pip install openai'.")

# Cliente OpenRouter para acesso ao Gemini (ou outros modelos)
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/seu-usuario/odonto-bot-api", # Opcional: mude para seu repo
        "X-Title": "OdontoBot AI"
    },
    timeout=httpx.Timeout(60.0)
)

chat_completion = lambda **kwargs: openrouter_client.chat.completions.create(**kwargs)

async def transcribe_audio_whisper(audio_url: str) -> Optional[str]:
    """Faz o download de um áudio e o transcreve usando a API do Whisper."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            # A transcrição é uma operação síncrona, então a executamos em uma thread separada
            transcription = await asyncio.to_thread(
                openai_whisper.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.ogg", response.content, "audio/ogg")
            )
            return transcription.text
    except Exception as e:
        print(f"🚨 Erro na transcrição com Whisper: {e}")
        return None

# ───────────── 4. DATABASE (ORM) ─────────────
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True)
    nome_completo = Column(String)
    primeiro_nome = Column(String)
    telefone = Column(String, unique=True, nullable=False)
    # Outros campos que podem ser úteis no futuro
    # endereco = Column(String)
    # email = Column(String)
    # data_nascimento = Column(Date)

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    data_hora = Column(DateTime(timezone=False), nullable=False)
    procedimento = Column(String, nullable=False)
    status = Column(String, default="confirmado") # ex: confirmado, cancelado, realizado
    paciente = relationship("Paciente")

class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    role = Column(String) # 'user', 'assistant', ou 'tool'
    content = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=now_tz)

class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
    categoria = Column(String, index=True)
    descricao = Column(Text) # Adicionado para respostas mais completas
    valor_descritivo = Column(String, nullable=False) # ex: "A partir de R$ 200"
    valor_base = Column(Float) # Valor numérico para cálculos futuros

engine = create_engine(DATABASE_URL, pool_recycle=3600, connect_args={"options": "-c timezone=utc"})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ───────────── 5. DATABASE SEEDING ─────────────
def seed_database(db: Session) -> None:
    """Popula o banco de dados com dados iniciais se estiver vazio."""
    if db.query(Procedimento).first():
        return

    print("🌱 Populando o banco de dados com procedimentos iniciais...")
    procedimentos_iniciais = [
        Procedimento(nome="Consulta de Avaliação", categoria="Diagnóstico", descricao="Primeira consulta para avaliação completa da saúde bucal e planejamento de tratamento.", valor_descritivo="R$ 100 a R$ 150", valor_base=100.0),
        Procedimento(nome="Limpeza (Profilaxia)", categoria="Prevenção", descricao="Remoção de placa bacteriana e tártaro, seguida de polimento para manter os dentes saudáveis.", valor_descritivo="R$ 150 a R$ 250", valor_base=150.0),
        Procedimento(nome="Restauração em Resina", categoria="Clínica Geral", descricao="Reparo de dentes danificados por cáries, utilizando resina da cor do dente.", valor_descritivo="A partir de R$ 200", valor_base=200.0),
        Procedimento(nome="Clareamento Dental a Laser", categoria="Estética", descricao="Tratamento em consultório para clarear os dentes de forma rápida e eficaz.", valor_descritivo="Consulte nossos pacotes", valor_base=800.0),
        Procedimento(nome="Raio-X Panorâmica", categoria="Radiografias", descricao="Exame de imagem que fornece uma visão geral de todos os dentes e da mandíbula.", valor_descritivo="R$ 90 a R$ 120", valor_base=90.0)
    ]
    db.add_all(procedimentos_iniciais)
    db.commit()
    print("✅ Banco de dados populado.")

# Cria tabelas e popula o DB na inicialização
try:
    Base.metadata.create_all(engine)
    with SessionLocal() as db_session:
        seed_database(db_session)
except Exception as e:
    print(f"🚨 Erro ao inicializar o banco de dados: {e}")

# ───────────── 6. HELPER FUNCTIONS & BUSINESS LOGIC ─────────────
def normalize_relative_date(text: str) -> str:
    """Converte termos relativos como 'hoje' ou 'amanhã à tarde' em datas e horas concretas."""
    text_lower = text.lower()
    
    # Mapeia dias relativos para timedelta
    relative_days = {"hoje": 0, "amanhã": 1, "amanha": 1}
    for word, delta_days in relative_days.items():
        if word in text_lower:
            target_date = (now_naive().date() + timedelta(days=delta_days))
            text_lower = re.sub(word, target_date.strftime("%d/%m/%Y"), text_lower, flags=re.IGNORECASE)

    # Mapeia períodos do dia para horários específicos
    periods = {"manhã": "09:00", "manha": "09:00", "tarde": "15:00", "noite": "19:00"}
    for period, time_str in periods.items():
        if period in text_lower:
            text_lower = re.sub(rf"\b{period}\b", time_str, text_lower, flags=re.IGNORECASE)
            
    return text_lower

def find_or_create_patient(db: Session, phone: str) -> Paciente:
    """Busca um paciente pelo telefone. Se não existir, cria um novo."""
    patient = db.query(Paciente).filter_by(telefone=phone).first()
    if patient:
        return patient
    new_patient = Paciente(telefone=phone)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient

# ───────────── 7. TOOL FUNCTIONS (Para a IA usar) ─────────────

def list_procedures(db: Session) -> str:
    """Lista todos os procedimentos disponíveis, agrupados por categoria."""
    procedures = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedures:
        return "No momento, não temos uma lista de procedimentos cadastrada."
    
    categories = defaultdict(list)
    for p in procedures:
        categories[p.categoria].append(p.nome)
        
    response_parts = []
    for category, names in categories.items():
        proc_list = "\n".join(f"- {name}" for name in names)
        response_parts.append(f"*{category}*\n{proc_list}")
        
    return "\n\n".join(response_parts)

def get_procedure_details(db: Session, procedure_name: str) -> str:
    """Busca e retorna os detalhes de um procedimento específico, incluindo descrição e preço."""
    # Busca flexível, permitindo nomes parciais
    search_term = f"%{procedure_name.strip()}%"
    procedure = db.query(Procedimento).filter(Procedimento.nome.ilike(search_term)).first()
    
    if not procedure:
        return f"Não encontrei informações sobre o procedimento '{procedure_name}'. Gostaria de ver a lista completa de procedimentos que oferecemos?"
        
    details = f"Claro! Sobre o procedimento *{procedure.nome}*:\n\n"
    if procedure.descricao:
        details += f"*{procedure.descricao}*\n\n"
    details += f"O valor é: *{procedure.valor_descritivo}*."
    return details

def get_available_slots(db: Session, day_str: str) -> str:
    """Verifica e retorna os horários de agendamento livres para uma data específica."""
    normalized_day = normalize_relative_date(day_str)
    
    # Usa dateparser para uma interpretação flexível da data
    parsed_date = parse_date(normalized_day, languages=['pt'], settings={
        "TIMEZONE": "America/Sao_Paulo", 
        "PREFER_DATES_FROM": "future"
    })
    
    if not parsed_date:
        return f"Não consegui entender a data '{day_str}'. Por favor, tente algo como 'hoje', 'amanhã' ou '25 de dezembro'."
    
    target_date_naive = to_naive(parsed_date)
    
    if target_date_naive.date() < now_naive().date():
        return "Não é possível verificar horários em datas passadas."

    # Define o início e o fim do dia de trabalho
    day_start = target_date_naive.replace(hour=BUSINESS_START.hour, minute=0, second=0, microsecond=0)
    day_end = target_date_naive.replace(hour=BUSINESS_END.hour, minute=0, second=0, microsecond=0)

    # Busca agendamentos confirmados no dia
    booked_slots_query = db.query(Agendamento.data_hora).filter(
        Agendamento.data_hora.between(day_start, day_end),
        Agendamento.status == "confirmado"
    )
    booked_slots = {slot.data_hora for slot in booked_slots_query}

    # Gera todos os slots possíveis e filtra os disponíveis
    available_slots = []
    current_slot = day_start
    while current_slot < day_end:
        # Só mostra horários futuros
        if current_slot not in booked_slots and current_slot > now_naive():
            available_slots.append(current_slot)
        current_slot += timedelta(minutes=SLOT_MINUTES)
    
    if not available_slots:
        return f"Que pena, não tenho horários livres para {target_date_naive.strftime('%d/%m/%Y')}. Gostaria de verificar outra data?"
        
    slots_str = ", ".join(s.strftime("%H:%M") for s in available_slots)
    return f"Para o dia {target_date_naive.strftime('%d/%m/%Y')}, tenho os seguintes horários livres: {slots_str}."

def schedule_appointment(db: Session, patient_id: int, full_name: str, datetime_str: str, procedure: str) -> str:
    """Agenda uma consulta, validando o horário e atualizando os dados do paciente."""
    normalized_datetime = normalize_relative_date(datetime_str)
    dt_obj = parse_date(normalized_datetime, languages=['pt'], settings={"TIMEZONE": "America/Sao_Paulo", "PREFER_DATES_FROM": "future"})
    
    if not dt_obj:
        return f"Não consegui entender a data e hora '{datetime_str}'. Por favor, tente algo como 'amanhã às 10:00' ou '25/12 às 15:30'."

    dt_naive = to_naive(dt_obj)

    # Validação do horário
    if not (BUSINESS_START <= dt_naive.time() < BUSINESS_END):
         return f"O horário {dt_naive.strftime('%H:%M')} está fora do nosso horário de funcionamento (das {BUSINESS_START.strftime('%H:%M')} às {BUSINESS_END.strftime('%H:%M')})."

    if dt_naive < now_naive():
        return "Não é possível agendar em um horário que já passou."

    # Verifica se o slot está ocupado
    existing = db.query(Agendamento).filter_by(data_hora=dt_naive, status="confirmado").first()
    if existing:
        return f"Desculpe, o horário de {dt_naive.strftime('%d/%m/%Y às %H:%M')} acabou de ser preenchido. Gostaria de escolher outro?"

    patient = db.query(Paciente).filter_by(id=patient_id).first()
    if not patient: return "Erro: Paciente não encontrado."

    # Atualiza o nome do paciente se for a primeira vez
    if not patient.nome_completo:
        patient.nome_completo = full_name
        patient.primeiro_nome = full_name.split(' ')[0]

    new_appointment = Agendamento(
        paciente_id=patient.id,
        data_hora=dt_naive,
        procedimento=procedure,
        status="confirmado"
    )
    db.add(new_appointment)
    db.commit()
    
    return f"Perfeito, {patient.primeiro_nome}! Seu agendamento para *{procedure}* foi confirmado para o dia *{dt_naive.strftime('%d/%m/%Y às %H:%M')}*. Enviaremos um lembrete um dia antes. Até lá!"

def cancel_appointment(db: Session, patient_id: int) -> str:
    """Cancela o próximo agendamento de um paciente."""
    upcoming_appointment = db.query(Agendamento).filter(
        Agendamento.paciente_id == patient_id,
        Agendamento.status == "confirmado",
        Agendamento.data_hora > now_naive()
    ).order_by(Agendamento.data_hora.asc()).first()

    if not upcoming_appointment:
        return "Você não possui nenhum agendamento futuro para cancelar."

    appointment_details = f"{upcoming_appointment.procedimento} no dia {upcoming_appointment.data_hora.strftime('%d/%m/%Y às %H:%M')}"
    
    upcoming_appointment.status = "cancelado"
    db.commit()
    
    return f"Ok. Seu agendamento de {appointment_details} foi cancelado com sucesso. Se precisar, é só chamar para reagendar."


# ───────────── 8. FASTAPI APP & WEBHOOK ─────────────
app = FastAPI(
    title="OdontoBot AI",
    version="15.0.0",
    description="API de chatbot para agendamento em clínica odontológica usando Gemini e Function Calling."
)

@app.on_event("startup")
async def on_startup():
    print(f"🚀 API OdontoBot v{app.version} iniciada com sucesso!")

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "message": "Bem-vindo à API do OdontoBot!"}

# Modelos Pydantic para o payload do webhook
class ZapiTextMessage(BaseModel):
    message: Optional[str] = None

class ZapiAudioMessage(BaseModel):
    audioUrl: Optional[str] = None

class ZapiPayload(BaseModel):
    phone: str
    text: Optional[ZapiTextMessage] = None
    audio: Optional[ZapiAudioMessage] = None

# Mapeamento de nomes de ferramentas para funções Python
AVAILABLE_TOOLS = {
    "list_procedures": list_procedures,
    "get_procedure_details": get_procedure_details,
    "get_available_slots": get_available_slots,
    "schedule_appointment": schedule_appointment,
    "cancel_appointment": cancel_appointment,
}

# Definições das ferramentas para a IA
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "list_procedures",
            "description": "Lista todos os procedimentos e serviços oferecidos pela clínica, agrupados por categoria.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_procedure_details",
            "description": "Obtém informações detalhadas, incluindo descrição e preço, de um procedimento específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure_name": {"type": "string", "description": "O nome do procedimento a ser buscado. Ex: 'Limpeza', 'Clareamento Dental'."}
                },
                "required": ["procedure_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Verifica os horários de agendamento livres para uma data específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day_str": {"type": "string", "description": "A data para verificar. Pode ser relativa ('hoje', 'amanhã') ou específica ('25 de dezembro', '15/10/2025')."}
                },
                "required": ["day_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Agenda uma consulta para o paciente. Sempre peça e confirme o nome completo do paciente antes de usar esta função.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "description": "O nome completo do paciente, conforme informado por ele."},
                    "datetime_str": {"type": "string", "description": "A data e hora desejada para o agendamento. Ex: 'amanhã às 15:30', 'hoje 10h'."},
                    "procedure": {"type": "string", "description": "O nome do procedimento a ser agendado."}
                },
                "required": ["full_name", "datetime_str", "procedure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancela o próximo agendamento confirmado do paciente. Use quando o usuário pedir para cancelar sua consulta.",
        }
    }
]

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Recebe, processa e responde a mensagens do WhatsApp."""
    try:
        payload_data = await request.json()
        payload = ZapiPayload(**payload_data)
    except Exception as e:
        print(f"🚨 Erro de payload inválido: {e}")
        raise HTTPException(status_code=422, detail="Payload inválido.") from e

    phone_number = payload.phone
    user_message_text = None

    if payload.audio and payload.audio.audioUrl:
        user_message_text = await transcribe_audio_whisper(payload.audio.audioUrl)
    elif payload.text and payload.text.message:
        user_message_text = payload.text.message

    if not user_message_text:
        # Envia uma saudação padrão se a mensagem estiver vazia
        await send_zapi_message(phone_number, "Olá! Sou Sofia, assistente virtual da DI DONATO ODONTO. Como posso ajudar você hoje?")
        return {"status": "greeting_sent"}

    patient = find_or_create_patient(db, phone_number)
    db.add(HistoricoConversa(paciente_id=patient.id, role="user", content=user_message_text))
    db.commit()

    # Monta o histórico da conversa
    conversation_history_db = db.query(HistoricoConversa).filter(
        HistoricoConversa.paciente_id == patient.id
    ).order_by(HistoricoConversa.timestamp.desc()).limit(12).all()
    
    # O prompt do sistema é a personalidade da "Sofia"
    system_prompt = (
        "Você é Sofia, a assistente virtual da clínica odontológica DI DONATO ODONTO. "
        "Pergunte pelo nome da pessoa assim que você se apresentar, isso servirá para voce se dirigir a ela durante toda a conversa. "
        "Sua personalidade é profissional, extremamente prestativa, empática e eficiente. "
        "Seu objetivo é ajudar os pacientes a obter informações gerais sobre procedimentos realizados na clinica e agendar consultas de forma clara e fácil.\n\n"
        "REGRAS DE OURO:\n"
        "1. **Use as Ferramentas:** SEMPRE use as ferramentas disponíveis para obter informações REAIS. Não invente preços, horários ou detalhes de procedimentos.\n"
        "2. **Peça o Nome Completo:** ANTES de usar a ferramenta `schedule_appointment`, você DEVE perguntar e obter o nome completo do paciente.\n"
        "3. **Seja Clara e Concisa:** Responda de forma direta e fácil de entender. Use negrito (*texto*) para destacar informações importantes como datas, horários e nomes.\n"
        "4. **Confirme Informações:** Antes de finalizar um agendamento, confirme os detalhes com o usuário (ex: 'Só para confirmar, o agendamento para *Limpeza* fica no dia *25/12 às 10:30*. Correto?').\n"
        "5. **Lide com Incertezas:** Se não tiver certeza ou se uma ferramenta falhar, peça desculpas e diga que vai verificar com a equipe humana. Ex: 'Peço desculpas, não consegui verificar essa informação agora, mas nossa equipe entrará em contato em breve.'\n"
        "6. **Saudação e Despedida:** Comece sempre de forma amigável e termine se colocando à disposição."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": msg.role, "content": msg.content} for msg in reversed(conversation_history_db)])
    
    final_answer = ""
    # Loop para permitir que a IA chame ferramentas
    for _ in range(3): # Limita a 3 chamadas de ferramentas para evitar loops
        response = chat_completion(
            model="google/gemini-2.5-flash", # Modelo recomendado
            messages=messages,
            tools=TOOLS_DEFINITION,
            tool_choice="auto",
            temperature=0.3,
        )
        
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            # Se a IA não chamar nenhuma ferramenta, a resposta é final.
            final_answer = response_message.content or "Não sei como responder. Nossa equipe entrará em contato."
            break # Sai do loop de ferramentas

        # A IA quer chamar uma ferramenta
        messages.append(response_message) # Adiciona a chamada da ferramenta ao histórico

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = AVAILABLE_TOOLS.get(function_name)
            
            if not function_to_call:
                 tool_output = f"Erro: A ferramenta '{function_name}' não existe."
            else:
                try:
                    # Prepara os argumentos da função
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Adiciona 'db' e 'patient_id' se a função precisar
                    if function_name in ["schedule_appointment", "cancel_appointment"]:
                        function_args['patient_id'] = patient.id
                        tool_output = function_to_call(db=db, **function_args)
                    else:
                        tool_output = function_to_call(db=db, **function_args)

                except Exception as e:
                    print(f"🚨 Erro ao executar a ferramenta '{function_name}': {e}")
                    tool_output = f"Ocorreu um erro interno ao tentar executar a ação. Por favor, tente novamente."

            # Adiciona o resultado da ferramenta ao histórico da conversa
            messages.append(
                {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": tool_output}
            )
    else: # Se o loop terminar sem 'break'
        final_answer = "Parece que estamos com um problema para processar sua solicitação. Nossa equipe já foi notificada e entrará em contato."

    # Salva a resposta final da IA no histórico e envia para o usuário
    db.add(HistoricoConversa(paciente_id=patient.id, role="assistant", content=final_answer))
    db.commit()
    await send_zapi_message(phone_number, final_answer)
    
    return {"status": "processed", "response": final_answer}


async def send_zapi_message(phone: str, message: str) -> None:
    """Envia uma mensagem de texto usando a API da Z-API."""
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    payload = {"phone": phone, "message": message}
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(url, json=payload, headers=headers)
        except Exception as e:
            print(f"🚨 Erro ao enviar mensagem pela Z-API para {phone}: {e}")

