"""
OdontoBot AI – main.py – v14.0.8
────────────────────────────────────────────────────────────────────────────
Changelog
─────────
14.0.5  • sem “para finalizar cadastro”, nome completo salvo
14.0.6  • remove SyntaxError em blocos try/with
14.0.7  • ajustes menores de formatação
14.0.8  • corrige parâmetro termo_busca em consultar_precos_procedimentos
"""

# ─────────── 0. FUTURE PROOF (Python >= 3.11 type-hints) ──────────
from __future__ import annotations

# ─────────── 1. STANDARD / THIRD-PARTY IMPORTS ──────────
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
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    and_,
    or_,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ─────────── 2. ENV VARS & CONSTS ──────────
load_dotenv()
(
    DATABASE_URL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    ZAPI_API_URL,
    ZAPI_INSTANCE_ID,
    ZAPI_TOKEN,
    ZAPI_CLIENT_TOKEN,
) = (
    os.getenv("DATABASE_URL"),
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENROUTER_API_KEY"),
    os.getenv("ZAPI_API_URL"),
    os.getenv("ZAPI_INSTANCE_ID"),
    os.getenv("ZAPI_TOKEN"),
    os.getenv("ZAPI_CLIENT_TOKEN"),
)

if not all(
    [
        DATABASE_URL,
        OPENAI_API_KEY,
        OPENROUTER_API_KEY,
        ZAPI_API_URL,
        ZAPI_INSTANCE_ID,
        ZAPI_TOKEN,
        ZAPI_CLIENT_TOKEN,
    ]
):
    raise RuntimeError("Alguma variável de ambiente obrigatória está vazia.")

BR_TZ = pytz.timezone("America/Sao_Paulo")

BUSINESS_START, BUSINESS_END = time(9), time(17)
SLOT_MIN = 30  # minutos

def now_tz()    -> datetime: return datetime.now(BR_TZ)
def now_naive() -> datetime: return datetime.now(BR_TZ).replace(tzinfo=None)
def to_naive_sp(dt: datetime) -> datetime:
    return (
        BR_TZ.localize(dt) if dt.tzinfo is None else dt.astimezone(BR_TZ)
    ).replace(tzinfo=None)

# ─────────── 3. OPENAI & OPENROUTER CLIENTS ──────────
try:
    import openai  # type: ignore
    openai_whisper = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/Shouked/odonto-bot-api",
            "X-Title": "OdontoBot AI",
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
            return tr.text  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            print("Erro Whisper:", exc)
            return None

except ImportError as exc:  # noqa: BLE001
    raise RuntimeError("Biblioteca openai não instalada.") from exc

# ─────────── 4. SQLALCHEMY ORM ──────────
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
    historico    = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")

class Agendamento(Base):
    __tablename__ = "agendamentos"
    id = Column(Integer, primary_key=True)
    paciente_id  = Column(Integer, ForeignKey("pacientes.id"))
    data_hora    = Column(DateTime(timezone=False), nullable=False)
    procedimento = Column(String, nullable=False)
    status       = Column(String, default="confirmado")
    paciente     = relationship("Paciente", back_populates="agendamentos")

class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"
    id = Column(Integer, primary_key=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"))
    role     = Column(String, nullable=False)
    content  = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=now_tz)
    paciente  = relationship("Paciente", back_populates="historico")

class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True)
    nome             = Column(String, unique=True, nullable=False)
    categoria        = Column(String, index=True)
    valor_descritivo = Column(String, nullable=False)
    valor_base       = Column(Float)

# Engine & Session
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

# Preenche procedimentos básicos (lista resumida para exemplo)
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

# ─────────── 5. HELPERS ──────────
def limpar_rotulos(texto: str) -> str:
    for padrao in (
        r"^\s*A[cç]ão:\s*",
        r"^\s*Resumo:\s*",
        r"^\s*Action:\s*",
        r"^\s*Summary:\s*",
    ):
        texto = re.sub(padrao, "", texto, flags=re.I | re.M)
    return texto.strip()

def parse_data_nasc(s: str) -> Optional[datetime]:
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s.strip())
    if m:
        d, mth, y = map(int, m.groups())
        y += 1900 if y < 100 and y > 25 else (2000 if y < 100 else 0)
        if 1 <= mth <= 12 and 1 <= d <= calendar.monthrange(y, mth)[1]:
            return datetime(y, mth, d)
    return parse_date(s, languages=["pt"], settings={"DATE_ORDER": "DMY"})

REL_MAP = {"hoje":0, "amanhã":1, "amanha":1, "depois de amanhã":2, "depois de amanha":2}
def normalizar_data_relativa(txt: str) -> str:
    t = txt.lower().strip()
    if t in REL_MAP:
        return (now_naive().date()+timedelta(days=REL_MAP[t])).strftime("%d/%m/%Y")
    prox = re.match(r"^pr[oó]xima\s+(\w+)$", t)
    if prox:
        dias=["segunda","terça","terca","quarta","quinta","sexta","sábado","sabado","domingo"]
        idx=dias.index(prox.group(1)) if prox.group(1) in dias else -1
        if idx>=0:
            delta=(idx-now_naive().weekday()+7)%7 or 7
            return (now_naive().date()+timedelta(days=delta)).strftime("%d/%m/%Y")
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
    return "\n\n".join(f"*{cat}*\n"+"\n".join(f"- {n}" for n in nomes) for cat,nomes in cats.items())

# -------- correção termo_busca --------
def consultar_precos_procedimentos(
    db: Session,
    termo_busca: str | None = None,
    termo: str | None = None,
) -> str:
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
        for a in db.query(Agendamento).filter(
            Agendamento.data_hora.between(ini, fim),
            Agendamento.status == "confirmado",
        )
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
        Agendamento.data_hora >= now_naive(),
    ).all()
    if not ags:
        return "Você não possui agendamentos ativos."
    return "\n".join(f"ID {a.id}: {a.procedimento} – {a.data_hora:%d/%m/%Y às %H:%M}" for a in ags)

# ─────────── 6. FUNÇÃO CENTRAL – processar_solicitacao_agendamento ──────────
# (idêntica à v14.0.7 — sem alterações; mantida aqui integral)

def processar_solicitacao_agendamento(
    db: Session,
    telefone_paciente: str,
    intencao: str,
    procedimento: Optional[str] = None,
    data_hora_texto: Optional[str] = None,
    confirmacao_usuario: bool = False,
    dados_paciente: Optional[Dict[str, str]] = None,
) -> str:
    paciente = buscar_ou_criar_paciente(db, telefone_paciente)

    # Atualiza cadastro
    if dados_paciente:
        if nome := dados_paciente.get("nome_completo"):
            paciente.nome_completo = nome.strip()
        if email := dados_paciente.get("email"):
            paciente.email = email
        if endereco := dados_paciente.get("endereco"):
            paciente.endereco = endereco
        if dn_str := dados_paciente.get("data_nascimento"):
            dn = parse_data_nasc(dn_str)
            if dn:
                paciente.data_nascimento = dn.date()
            else:
                return "Formato de data inválido. Use DD/MM/AAAA."
        db.commit()
        db.refresh(paciente)

    pendentes = [
        campo
        for campo, valor in [
            ("nome completo", paciente.nome_completo),
            ("data de nascimento", paciente.data_nascimento),
            ("e-mail", paciente.email),
            ("endereço", paciente.endereco),
        ]
        if not valor
    ]
    if pendentes:
        return (
            f"Ação: Solicite {pendentes[0]} de forma natural, sem repetir 'para finalizar o cadastro'."
        )

    ags = (
        db.query(Agendamento)
        .filter(
            Agendamento.paciente_id == paciente.id,
            Agendamento.status == "confirmado",
            Agendamento.data_hora >= now_naive(),
        )
        .order_by(Agendamento.data_hora)
        .all()
    )

    if intencao == "cancelar":
        if not ags:
            return "Você não possui agendamentos para cancelar."
        if len(ags) > 1 and not confirmacao_usuario:
            ids = ", ".join(str(a.id) for a in ags)
            return f"Ação: Peça quais IDs cancelar. IDs: {ids}"
        for ag in ags:
            ag.status = "cancelado"
        db.commit()
        return "Agendamento cancelado."

    if intencao in {"agendar", "reagendar"}:
        if not procedimento:
            return "Ação: Pergunte qual procedimento deseja."
        proc_obj = (
            db.query(Procedimento)
            .filter(Procedimento.nome.ilike(f"%{procedimento}%"))
            .first()
        )
        proc_real = proc_obj.nome if proc_obj else procedimento

        if not data_hora_texto:
            return f"Ação: Pergunte data e horário para '{proc_real}'."

        if re.search(r"[A-Za-z]", data_hora_texto):
            return "Por favor informe apenas a data e o horário (ex.: 20/06 às 10:00)."

        data_hora_texto = normalizar_data_relativa(data_hora_texto)
        dt = parse_date(
            data_hora_texto,
            languages=["pt"],
            settings={"PREFER_DATES_FROM": "future", "TIMEZONE": "America/Sao_Paulo"},
        )
        if not dt:
            return "Data/hora inválida."
        dt_naive = to_naive_sp(dt)

        if not BUSINESS_START <= dt_naive.time() <= BUSINESS_END:
            return "Atendemos apenas das 09:00 às 17:00."
        if db.query(Agendamento).filter_by(data_hora=dt_naive, status="confirmado").first():
            return "Horário ocupado. Use a função `consultar_horarios_disponiveis`."

        if not confirmacao_usuario:
            return (
                "Ação: Peça confirmação final. Resumo: "
                f"{proc_real} em {dt_naive:%d/%m/%Y às %H:%M}. Está correto?"
            )

        if intencao == "agendar":
            db.add(
                Agendamento(
                    paciente_id=paciente.id,
                    data_hora=dt_naive,
                    procedimento=proc_real,
                )
            )
        else:
            if not ags:
                return "Nenhum agendamento para reagendar."
            ags[0].data_hora = dt_naive
        db.commit()
        return f"Sucesso! Agendado para {dt_naive:%d/%m/%Y às %H:%M}."

    return "Desculpe, não entendi a solicitação."

# ─────────── 7. TOOLS ESPECIFICAÇÃO ──────────
available_functions: Dict[str, Any] = {
    "processar_solicitacao_agendamento": processar_solicitacao_agendamento,
    "listar_todos_os_procedimentos": listar_todos_os_procedimentos,
    "consultar_precos_procedimentos": consultar_precos_procedimentos,
    "consultar_horarios_disponiveis": consultar_horarios_disponiveis,
    "listar_agendamentos_ativos": listar_agendamentos_ativos,
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "processar_solicitacao_agendamento",
            "description": "Fluxo de agendar/reagendar/cancelar e coletar dados.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intencao": {
                        "type": "string",
                        "enum": ["agendar", "reagendar", "cancelar", "coletar_dados"],
                    },
                    "procedimento": {"type": "string"},
                    "data_hora_texto": {"type": "string"},
                    "confirmacao_usuario": {"type": "boolean"},
                    "dados_paciente": {
                        "type": "object",
                        "properties": {
                            "nome_completo": {"type": "string"},
                            "email": {"type": "string"},
                            "data_nascimento": {"type": "string"},
                            "endereco": {"type": "string"},
                        },
                    },
                },
                "required": ["intencao"],
            },
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
    },
    {
        "type": "function",
        "function": {
            "name": "listar_agendamentos_ativos",
            "description": "Mostra agendamentos confirmados",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ─────────── 8. FASTAPI ──────────
app = FastAPI(title="OdontoBot AI", version="14.0.8")

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
def health_head() -> Response:
    return Response(status_code=200)

# ─────────── 9. WEBHOOK ENDPOINT ──────────
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
        except Exception as exc:  # noqa: BLE001
            print("Erro ao enviar via Z-API:", exc, flush=True)

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    raw = await request.json()
    print(">>> payload recebido", raw)
    try:
        data = ZapiWebhookPayload(**raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="Payload inválido") from exc

    tel = data.phone
    user_msg: Optional[str] = None

    if data.audio and data.audio.audioUrl:
        user_msg = await transcrever_audio(data.audio.audioUrl)
    elif data.text and data.text.message:
        user_msg = data.text.message

    if not user_msg:
        await send_whatsapp(tel, "Olá! Sou Sofia, assistente da DI DONATO ODONTO. Em que posso ajudar?")
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
        "Você é Sofia, assistente virtual da clínica DI DONATO ODONTO (Dra. Valéria Cristina). "
        "Responda em português BR e utilize as ferramentas somente quando necessário. "
        "Peça dados de forma natural, sem repetir 'para finalizar o cadastro'. "
        "Converta expressões relativas de tempo (amanhã, próxima sexta) em datas absolutas antes de chamar funções."
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
            msgs_tools = msgs + [ai_msg]
            for call in ai_msg.tool_calls:  # type: ignore[attr-defined]
                fname = call.function.name
                kwargs = {"db": db}
                if fname == "consultar_precos_procedimentos":
                    args = json.loads(call.function.arguments)
                    # Nome do parâmetro já compatível (termo_busca)
                else:
                    args = json.loads(call.function.arguments)
                if "telefone_paciente" in available_functions[fname].__code__.co_varnames:  # type: ignore[attr-defined]
                    kwargs["telefone_paciente"] = tel
                result = available_functions[fname](**kwargs, **args)
                msgs_tools.append(
                    {
                        "tool_call_id": call.id,
                        "role": "tool",
                        "name": fname,
                        "content": result,
                    }
                )
            resp = chat_completion(
                model="google/gemini-2.5-flash-preview-05-20",
                messages=msgs_tools,
                temperature=0.2,
                top_p=0.8,
                max_tokens=1024,
            )
            ai_msg = resp.choices[0].message  # type: ignore[index]

        resposta_final = limpar_rotulos(ai_msg.content or "")

    except Exception as exc:  # noqa: BLE001
        print("Erro IA:", exc)
        resposta_final = "Desculpe, ocorreu um problema técnico."

    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final))
    db.commit()

    await send_whatsapp(tel, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
