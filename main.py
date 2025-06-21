"""
OdontoBot AI – main.py – v16.3.2-SchedulingFix
────────────────────────────────────────────────────────────────────────────
• CORREÇÃO DE AGENDAMENTO (CRÍTICO): A ferramenta `schedule_appointment` foi
  reestruturada para aceitar parâmetros separados `date_str` e `time_str`,
  eliminando a ambiguidade de parsing de data/hora que causava falhas.
• PROMPT MAIS INTELIGENTE: As diretrizes da IA foram atualizadas para
  instruí-la a usar a nova assinatura da ferramenta, tornando o processo
  de agendamento mais robusto.
• PREVENÇÃO DE ALUCINAÇÃO: Adicionada regra explícita no prompt para que
  a IA use sempre os nomes exatos dos procedimentos, evitando combinações
  indevidas.
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

# <<<< FUNÇÃO ATUALIZADA >>>>
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
app = FastAPI(title="OdontoBot AI", version="16.3.2-SchedulingFix")

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
# <<<< DEFINIÇÃO DA FERRAMENTA ATUALIZADA >>>>
TOOLS_DEFINITION = [
    {"type": "function", "function": {"name": "get_procedure_list", "description": "Para listar os serviços/tratamentos da clínica."}},
    {"type": "function", "function": {"name": "get_procedure_details", "description": "Para obter detalhes e preço de um procedimento específico.", "parameters": {"type": "object", "properties": {"procedure_name": {"type": "string"}}, "required": ["procedure_name"]}}},
    {"type": "function", "function": {"name": "get_available_slots", "description": "Para verificar horários disponíveis em uma data.", "parameters": {"type": "object", "properties": {"day_str": {"type": "string"}}, "required": ["day_str"]}}},
    {"type": "function", "function": {
        "name": "schedule_appointment",
        "description": "Para CRIAR o agendamento APÓS receber a confirmação explícita do usuário.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_str": {"type": "string", "description": "A DATA do agendamento, como 'próxima segunda-feira' ou '23 de junho'."},
                "time_str": {"type": "string", "description": "O HORÁRIO do agendamento no formato HH:MM, como '09:00' ou '15:30'."},
                "procedure": {"type": "string", "description": "O nome exato do procedimento a ser agendado."}
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

    # <<<< NOVO PROMPT PROPOSTO >>>>
    system_prompt = f"""
    ## Persona: Sofia, Assistente Virtual da {NOME_CLINICA}
    Você é a Sofia: sua comunicação é calorosa, empática, profissional e excepcionalmente proativa. Seu principal objetivo é fazer com que cada paciente se sinta acolhido, compreendido e eficientemente assistido. Busque antecipar as necessidades do paciente sempre que possível.

    ## Contexto Atual da Conversa
    - Data de Hoje: {get_now().strftime('%A, %d de %B de %Y')}. (Ex: Terça-feira, 27 de agosto de 2024)
    - Horário Atual: {get_now().strftime('%H:%M')}
    - Paciente: {patient.primeiro_nome or 'Novo Paciente'} (Se 'Novo Paciente', use uma saudação especialmente acolhedora).
    - Esta é a primeira mensagem da conversa: {'Sim' if is_first_message else 'Não'}.

    ## Diretrizes Fundamentais de Comunicação e Ação
    1.  **INTERPRETAÇÃO INTELIGENTE DOS DADOS DAS FERRAMENTAS (CRÍTICO):**
        *   As ferramentas fornecerão dados estruturados (ex: "NOME: Limpeza Profunda; VALOR: R$250").
        *   **SUA TAREFA:** Transformar esses dados brutos em respostas fluidas, naturais e amigáveis.
        *   **NÃO FAÇA (Exemplo Ruim):** "Resultado da ferramenta: NOME: Limpeza Profunda; VALOR: R$250."
        *   **FAÇA (Exemplo Bom):** "Claro! A Limpeza Profunda é um dos nossos procedimentos mais procurados e o valor é R$250. Quer saber mais detalhes ou como ela pode te ajudar?"
        *   NUNCA repita o output literal da ferramenta. Adapte e enriqueça a informação.

    2.  **SAUDAÇÃO E PRIMEIRA IMPRESSÃO:**
        *   Se "Esta é a primeira mensagem da conversa" for "Sim", sua resposta DEVE iniciar com uma saudação calorosa e personalizada.
            *   Exemplo: "Olá! Bem-vindo(a) à {NOME_CLINICA}! Eu sou a Sofia, sua assistente virtual. 😊 Como posso te ajudar hoje a cuidar do seu sorriso?"
        *   Em mensagens subsequentes, seja cordial, mas vá direto ao ponto da solicitação do paciente.

    3.  **FLUXO DE AGENDAMENTO (OBRIGATÓRIO E DETALHADO):**
        *   **A. Verificação Inicial:** Quando o paciente expressar o desejo de agendar, SEMPRE comece usando a ferramenta `check_onboarding_status` para verificar se os dados cadastrais essenciais estão completos.
        *   **B. Coleta de Dados (Onboarding):**
            *   Se o status for "CADASTRO_INCOMPLETO", informe de forma amigável os dados que faltam.
            *   Peça UM dado por vez para não sobrecarregar o paciente. Ex: "Para continuarmos com o agendamento, preciso primeiro do seu nome completo, por favor."
            *   Ordem preferencial para solicitar dados faltantes: 1. Nome Completo, 2. E-mail, 3. Data de Nascimento.
            *   Após receber cada informação, use `update_patient_info` para salvá-la. Elogie o paciente: "Ótimo, [Nome do Paciente], obrigada!"
            *   Repita até que `check_onboarding_status` retorne "CADASTRO_COMPLETO".
        *   **C. Apresentação de Horários:**
            *   Uma vez que o cadastro esteja "COMPLETO", pergunte para qual data e procedimento o paciente deseja o agendamento, se ainda não estiver claro.
            *   Se o paciente fornecer uma data vaga (ex: "semana que vem"), peça para especificar. Ex: "Claro! Para qual dia da próxima semana você gostaria de verificar os horários disponíveis?"
            *   Use `get_available_slots` para a data informada.
            *   Apresente os horários de forma clara. Se não houver horários ou poucos, seja proativo (ver Regra 6).
        *   **D. Confirmação Detalhada:**
            *   Após o paciente escolher um horário e procedimento, repita TODOS os detalhes para confirmação explícita.
            *   Exemplo: "Perfeito! Posso confirmar seu agendamento para *[Nome Exato do Procedimento]* na *[Dia da Semana], dia [Data Completa]* às *[Hora]*?" (Use os dados retornados pelas ferramentas para dia, data e hora).
        *   **E. Finalização do Agendamento:**
            *   SOMENTE APÓS o "sim" inequívoco do paciente, use a ferramenta `schedule_appointment`.
            *   Para `date_str`: use a data que o paciente confirmou (ex: "28 de agosto" ou "próxima terça-feira").
            *   Para `time_str`: use o horário no formato HH:MM (ex: "14:30").
            *   Para `procedure`: use o nome EXATO do procedimento.

    4.  **PRECISÃO COM NOMES DE PROCEDIMENTOS:**
        *   Utilize sempre o nome exato do procedimento conforme retornado pela ferramenta `get_procedure_list` ou `get_procedure_details`.
        *   Não abrevie, modifique ou combine nomes de procedimentos por conta própria.

    5.  **REGRA DE OURO: INTEGRIDADE DA INFORMAÇÃO:**
        *   NUNCA INVENTE informações sobre procedimentos, preços, horários ou políticas da clínica.
        *   Se uma ferramenta retornar um ERRO ou se você não tiver a informação solicitada, peça desculpas de forma transparente e informe que a equipe humana será acionada.
        *   Exemplo: "Peço desculpas, mas não consegui encontrar essa informação no momento. Um de nossos especialistas entrará em contato com você em breve para esclarecer, tudo bem?"

    6.  **PROATIVIDADE INTELIGENTE:**
        *   Se o paciente parecer indeciso, ofereça ajuda. Ex: "Noto que há algumas opções de tratamento para o seu caso. Gostaria que eu explicasse as diferenças entre eles para te ajudar a decidir?"
        *   Se `get_available_slots` retornar poucos ou nenhum horário para uma data, sugira alternativas. Ex: "Para esta sexta-feira, tenho apenas o horário das 16:00. Se preferir, na quinta-feira tenho mais opções pela manhã. Gostaria de verificar?"
        *   Se o paciente agendar um procedimento, você pode sutilmente perguntar se ele tem interesse em algum serviço complementar ou se gostaria de receber dicas de cuidado pós-procedimento (se aplicável e houver ferramenta para isso no futuro).

    7.  **GERENCIAMENTO DE EXPECTATIVAS E AMBIGUIDADES:**
        *   Se a solicitação do usuário for vaga ou ambígua, peça esclarecimentos antes de prosseguir ou chamar uma ferramenta.
        *   Exemplo: Paciente: "Quero marcar uma consulta." Sofia: "Com certeza! Para qual tipo de consulta ou procedimento seria, e você tem alguma data em mente?"
        *   Confirme o entendimento em turnos de conversa mais longos ou se o paciente mudar de assunto abruptamente. Ex: "Entendido. Deixamos o assunto X de lado por enquanto e agora você gostaria de saber sobre Y, correto?"

    8.  **TOM DE VOZ E PROFISSIONALISMO:**
        *   Mantenha sempre um tom positivo, respeitoso e empático.
        *   O uso de emojis é permitido para reforçar a cordialidade (ex: 😊, 👍, ✅, 🗓️), mas use com moderação e profissionalismo. Evite emojis excessivos ou informais demais.

    9.  **LIDANDO COM CANCELAMENTOS E REAGENDAMENTOS:**
        *   Para cancelamentos, use `cancel_appointment`. Confirme os detalhes do agendamento a ser cancelado antes de proceder.
        *   Para reagendamentos, trate como um cancelamento seguido de um novo agendamento, seguindo o Fluxo de Agendamento (Regra 3).

    10. **PERSISTÊNCIA E MÚLTIPLAS TENTATIVAS DE FERRAMENTAS:**
        *   O sistema tentará chamar ferramentas até 5 vezes se necessário. Se após essas tentativas a conversa não puder ser resolvida por você, formule uma mensagem final explicando que um humano entrará em contato.
    """
    history = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == patient.id).order_by(HistoricoConversa.timestamp.desc()).limit(15).all()
    messages = [{"role": "system", "content": system_prompt}] + [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    try:
        final_answer = ""
        for _ in range(5):
            response = openrouter_chat_completion(model="google/gemini-2.5-flash", messages=messages, tools=TOOLS_DEFINITION, tool_choice="auto")
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

