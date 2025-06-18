# OdontoBot AI – main.py – v14.1.1
# Atualizações:
# - Novo system_prompt mais robusto
# - Fallback para resposta vazia da IA
# - Estrutura mantida com boas práticas

from __future__ import annotations

# ───────────── 1. IMPORTS ─────────────
import asyncio
import calendar
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional

import httpx
import pytz
from dateparser import parse as parse_date
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import (
    Column, Date, DateTime, Float, ForeignKey, Integer, String, Text,
    and_, or_, create_engine
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ───────────── 2. ENVIRONMENT ─────────────
load_dotenv()
(
    DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY,
    ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN
) = (
    os.getenv("DATABASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENROUTER_API_KEY"),
    os.getenv("ZAPI_API_URL"), os.getenv("ZAPI_INSTANCE_ID"), os.getenv("ZAPI_TOKEN"), os.getenv("ZAPI_CLIENT_TOKEN")
)

if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente obrigatória está vazia.")

BR_TZ = pytz.timezone("America/Sao_Paulo")
BUSINESS_START, BUSINESS_END, SLOT_MIN = time(9), time(17), 30

def now_tz() -> datetime:
    return datetime.now(BR_TZ)

def now_naive() -> datetime:
    return datetime.now(BR_TZ).replace(tzinfo=None)

def to_naive_sp(dt: datetime) -> datetime:
    return (
        BR_TZ.localize(dt) if dt.tzinfo is None else dt.astimezone(BR_TZ)
    ).replace(tzinfo=None)

# ───────────── 3. OPENAI / OPENROUTER ─────────────
try:
    import openai

    openai_whisper = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/Shouked/odonto-bot-api",
            "X-Title": "OdontoBot AI"
        },
        timeout=httpx.Timeout(45.0),
    )

    def chat_completion(**kw):
        return openrouter.chat.completions.create(**kw)

    async def transcrever_audio(url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(url)
                r.raise_for_status()
            tr = await asyncio.to_thread(
                openai_whisper.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.ogg", r.content, "audio/ogg"),
            )
            return tr.text
        except Exception as exc:
            print("Erro Whisper:", exc)
            return None

except ImportError as exc:
    raise RuntimeError("Biblioteca openai não instalada.") from exc
# ───────────── 4. SQLALCHEMY ORM ─────────────
Base = declarative_base()

class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True)
    nome_completo = Column(String)
    telefone = Column(String, unique=True, nullable=False)
    endereco = Column(String)
    email = Column(String)
    data_nascimento = Column(Date)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan")
    historico = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    data_hora = Column(DateTime(timezone=False), nullable=False)
    procedimento = Column(String, nullable=False)
    status = Column(String, default="confirmado")
    paciente = relationship("Paciente", back_populates="agendamentos")

class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=now_tz)
    paciente = relationship("Paciente", back_populates="historico")

class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
    categoria = Column(String, index=True)
    valor_descritivo = Column(String, nullable=False)
    valor_base = Column(Float)

engine = create_engine(DATABASE_URL, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables() -> None:
    Base.metadata.create_all(bind=engine)

def seed_procedimentos(db: Session) -> None:
    if db.query(Procedimento).first():
        return
    db.add(
        Procedimento(
            nome="Clareamento em consultório (por arcada)",
            categoria="Clareamento Dentário",
            valor_descritivo="R$316 a R$330",
            valor_base=316,
        )
    )
    db.commit()

# ───────────── 5. HELPERS ─────────────
def limpar_rotulos(texto: str) -> str:
    for padrao in (r"^\\s*A[cç]ão:\\s*", r"^\\s*Resumo:\\s*", r"^\\s*Action:\\s*", r"^\\s*Summary:\\s*"):
        texto = re.sub(padrao, "", texto, flags=re.I | re.M)
    return texto.strip()

def parse_data_nasc(s: str) -> Optional[datetime]:
    m = re.match(r"^(\\d{1,2})[/-](\\d{1,2})[/-](\\d{2,4})$", s.strip())
    if m:
        d, mth, y = map(int, m.groups())
        y += 1900 if y < 100 and y > 25 else (2000 if y < 100 else 0)
        if 1 <= mth <= 12 and 1 <= d <= calendar.monthrange(y, mth)[1]:
            return datetime(year=y, month=mth, day=d)
    return parse_date(s, languages=["pt"], settings={"DATE_ORDER": "DMY"})

REL_MAP = {"hoje": 0, "amanhã": 1, "amanha": 1, "depois de amanhã": 2, "depois de amanha": 2}

def normalizar_data_relativa(txt: str) -> str:
    low = txt.lower().strip()
    if low in REL_MAP:
        return (now_naive().date() + timedelta(days=REL_MAP[low])).strftime("%d/%m/%Y")
    prox = re.match(r"^pr[oó]xima\\s+(\\w+)$", low)
    if prox:
        dias = ["segunda", "terça", "terca", "quarta", "quinta", "sexta", "sábado", "sabado", "domingo"]
        idx = dias.index(prox.group(1)) if prox.group(1) in dias else -1
        if idx >= 0:
            delta = (idx - now_naive().weekday() + 7) % 7 or 7
            return (now_naive().date() + timedelta(days=delta)).strftime("%d/%m/%Y")
    return txt
NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'´` ]{6,}$")

def tentar_salvar_nome(db: Session, p: Paciente, msg: str) -> None:
    if p.nome_completo:
        return
    cand = " ".join(msg.strip().split())
    if NAME_RE.match(cand) and len(cand.split()) >= 2 and not re.search(r"\d", cand):
        p.nome_completo = cand
        db.commit()
        db.refresh(p)

def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    pac = db.query(Paciente).filter_by(telefone=tel).first()
    if not pac:
        pac = Paciente(telefone=tel)
        db.add(pac)
        db.commit()
        db.refresh(pac)
    return pac

def listar_todos_os_procedimentos(db: Session) -> str:
    procs = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procs:
        return "Lista de procedimentos indisponível."
    cats: dict[str, list[str]] = defaultdict(list)
    for p in procs:
        cats[p.categoria].append(p.nome)
    return "\n\n".join(f"*{cat}*\n" + "\n".join(f"- {n}" for n in nomes) for cat, nomes in cats.items())

def consultar_precos_procedimentos(db: Session, termo_busca: str | None = None, termo: str | None = None) -> str:
    query = termo_busca or termo or ""
    norm = re.sub(r"[-.,]", " ", query.lower())
    palavras = norm.split()
    if not palavras:
        return "Não entendi qual procedimento você procura."
    filtros = [Procedimento.nome.ilike(f"%{p}%") for p in palavras]
    res = db.query(Procedimento).filter(or_(*filtros)).all()
    if not res:
        return f"Não encontrei valores para '{query}'."
    res.sort(key=lambda proc: sum(p in proc.nome.lower() for p in palavras), reverse=True)
    linhas = []
    for r in res[:3]:
        if r.valor_base:
            linhas.append(f"O valor para {r.nome} é a partir de R$ {int(r.valor_base):,}.00".replace(",", "."))
        else:
            linhas.append(f"Para {r.nome}, o valor é {r.valor_descritivo}")
    return "\n".join(linhas)

def consultar_horarios_disponiveis(db: Session, dia: str) -> str:
    dia = normalizar_data_relativa(dia)
    d = parse_date(dia, languages=["pt"], settings={"TIMEZONE": "America/Sao_Paulo"})
    if not d:
        return "Data inválida."
    d_naive = to_naive_sp(d)
    ini = d_naive.replace(hour=BUSINESS_START.hour, minute=0, second=0, microsecond=0)
    fim = d_naive.replace(hour=BUSINESS_END.hour, minute=0, second=0, microsecond=0)
    ocupados = {
        a.data_hora.replace(second=0, microsecond=0)
        for a in db.query(Agendamento)
        .filter(Agendamento.data_hora.between(ini, fim), Agendamento.status == "confirmado")
    }
    slots = []
    cur = ini
    delta = timedelta(minutes=SLOT_MIN)
    while cur <= fim - delta:
        if cur not in ocupados and cur > now_naive():
            slots.append(cur)
        cur += delta
    if not slots:
        return "Sem horários livres nesse dia."
    return "Horários livres: " + ", ".join(s.strftime("%H:%M") for s in slots) + "."

def listar_agendamentos_ativos(db: Session, telefone_paciente: str) -> str:
    p = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(
        Agendamento.paciente_id == p.id,
        Agendamento.status == "confirmado",
        Agendamento.data_hora >= now_naive()
    ).all()
    if not ags:
        return "Você não possui agendamentos ativos."
    return "\\n".join(f"ID {a.id}: {a.procedimento} – {a.data_hora:%d/%m/%Y às %H:%M}" for a in ags)
# ───────────── 6. TOOLS (function-calling) ─────────────
available_functions: Dict[str, Any] = {
    "processar_solicitacao_agendamento": listar_agendamentos_ativos,  # corrigido
    "listar_todos_os_procedimentos": listar_todos_os_procedimentos,
    "consultar_precos_procedimentos": consultar_precos_procedimentos,
    "consultar_horarios_disponiveis": consultar_horarios_disponiveis,
    "listar_agendamentos_ativos": listar_agendamentos_ativos,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "listar_agendamentos_ativos",
            "description": "Mostra agendamentos confirmados",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_todos_os_procedimentos",
            "description": "Lista todos os procedimentos",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_precos_procedimentos",
            "description": "Informa preço de um procedimento",
            "parameters": {
                "type": "object",
                "properties": {"termo_busca": {"type": "string"}},
                "required": ["termo_busca"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_horarios_disponiveis",
            "description": "Exibe horários livres",
            "parameters": {
                "type": "object",
                "properties": {"dia": {"type": "string"}},
                "required": ["dia"],
            },
        },
    }
]

# ───────────── 7. FASTAPI APP ─────────────
app = FastAPI(title="OdontoBot AI", version="14.1.1")

@app.on_event("startup")
async def startup() -> None:
    await asyncio.to_thread(create_tables)
    with SessionLocal() as db:
        seed_procedimentos(db)
    print("🟢 Banco pronto")

@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.head("/")
def health_h() -> Response:
    return Response(status_code=200)

# ───────────── 8. WHATSAPP WEBHOOK ─────────────
class ZapiText(BaseModel):
    message: Optional[str] = None

class ZapiAudio(BaseModel):
    audioUrl: Optional[str] = None

class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None
    audio: Optional[ZapiAudio] = None

async def send_whatsapp(phone: str, message: str) -> None:
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": phone, "message": message}
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(url, json=payload, headers=headers)
        except Exception as exc:
            print("Erro Z-API:", exc, flush=True)

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    raw = await request.json()
    print("payload", raw)

    try:
        pay = ZapiWebhookPayload(**raw)
    except Exception as exc:
        raise HTTPException(422, "payload inválido") from exc

    tel = pay.phone
    user_msg: Optional[str] = None

    if pay.audio and pay.audio.audioUrl:
        user_msg = await transcrever_audio(pay.audio.audioUrl)
    elif pay.text and pay.text.message:
        user_msg = pay.text.message

    if not user_msg:
        await send_whatsapp(tel, "Olá! Sou Sofia, assistente da DI DONATO ODONTO. Como posso ajudar?")
        return {"status": "saudacao"}

    paciente = buscar_ou_criar_paciente(db, tel)
    tentar_salvar_nome(db, paciente, user_msg)
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=user_msg))
    db.commit()

    historico = (
        db.query(HistoricoConversa)
        .filter(
            HistoricoConversa.paciente_id == paciente.id,
            HistoricoConversa.timestamp >= now_tz() - timedelta(hours=24),
            HistoricoConversa.role != "system",
        )
        .order_by(HistoricoConversa.timestamp)
        .all()
    )

    system_prompt = (
        "Você é Sofia, assistente virtual da clínica odontológica DI DONATO ODONTO, da Dra. Valéria Cristina. "
        "Seu papel é conversar de forma amigável, clara e objetiva, ajudando o paciente a:\n"
        "1. Confirmar ou informar o nome completo.\n"
        "2. Dizer qual procedimento deseja realizar (como limpeza, clareamento, etc.).\n"
        "3. Informar a data e o horário preferidos.\n\n"
        "Sempre mantenha um tom acolhedor, evite repetições como 'para finalizar o cadastro', e substitua termos relativos como 'amanhã' por datas absolutas (ex: 20/06).\n"
        "Nunca diga que é uma IA.\n"
        "Se não tiver dados suficientes, peça as informações de forma natural, como se fosse uma atendente de verdade.\n\n"
        "Caso detecte a intenção de agendar, reagendar ou cancelar, use as ferramentas disponíveis para processar a solicitação.\n"
        "Seja objetiva e ajude o paciente a resolver tudo de forma prática e cordial."
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    msgs.extend({"role": m.role, "content": m.content} for m in historico[-10:])
    msgs.append({"role": "user", "content": user_msg})

    try:
        resp = chat_completion(
            model="google/gemini-2.5-flash-preview-05-20",
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            top_p=0.8,
            max_tokens=1024,
        )
        ai_msg = resp.choices[0].message  # type: ignore[index]

        while getattr(ai_msg, "tool_calls", None):
            chain = msgs + [ai_msg]
            for call in ai_msg.tool_calls:
                fname = call.function.name
                args = json.loads(call.function.arguments)
                kwargs = {"db": db}
                if "telefone_paciente" in available_functions[fname].__code__.co_varnames:
                    kwargs["telefone_paciente"] = tel
                result = available_functions[fname](**kwargs, **args)
                chain.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": fname,
                    "content": result,
                })
            resp = chat_completion(
                model="google/gemini-2.5-flash-preview-05-20",
                messages=chain,
                temperature=0.2,
                top_p=0.8,
                max_tokens=1024,
            )
            ai_msg = resp.choices[0].message  # type: ignore[index]

        resposta_final = limpar_rotulos(ai_msg.content or "")
        if not resposta_final:
            resposta_final = "Desculpe, não consegui entender. Poderia repetir?"

    except Exception as exc:
        print("Erro IA:", exc)
        resposta_final = "Desculpe, ocorreu um problema técnico."

    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final))
    db.commit()

    await send_whatsapp(tel, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
