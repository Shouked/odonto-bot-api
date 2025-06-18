"""
API principal do OdontoBot AI — versão 14.0.3
----------------------------------------------------------------
▪ Grava nome completo automaticamente, mesmo se a IA enviar só o primeiro.
▪ `consultar_horarios_disponiveis` lista todos os slots livres.
▪ Prompt reforçado para a IA devolver `nome_completo` exatamente como digitado.
▪ Mantém todas as correções anteriores (rótulos, datas naïve, etc.).
"""

from __future__ import annotations

import asyncio
import calendar
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, time, date as DateObject
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
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ─────────────────────────────── 1. ENV/VARS ──────────────────────────────── #
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
    raise RuntimeError("Alguma variável de ambiente obrigatória não foi definida.")

BR_TIMEZONE = pytz.timezone("America/Sao_Paulo")


def get_now() -> datetime:
    return datetime.now(BR_TIMEZONE)


# Utilidades naïve (sem tzinfo) ──────────────────────────────────────────── #
def get_now_naive() -> datetime:
    return datetime.now(BR_TIMEZONE).replace(tzinfo=None)


def to_naive_sp(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = BR_TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(BR_TIMEZONE)
    return dt.replace(tzinfo=None)


BUSINESS_START = time(hour=9)
BUSINESS_END = time(hour=17)
SLOT_MINUTES = 30

# ─────────────────────────────── 2. IA CLIENTS ────────────────────────────── #
try:
    import openai  # type: ignore

    openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/Shouked/odonto-bot-api",
            "X-Title": "OdontoBot AI",
        },
        timeout=httpx.Timeout(45.0),
    )

    def openrouter_chat_completion(**kw):
        return openrouter_client.chat.completions.create(**kw)

    async def transcrever_audio_whisper(audio_url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(audio_url, timeout=30)
                resp.raise_for_status()
                audio_bytes = resp.content
            transcription = await asyncio.to_thread(
                openai_whisper_client.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.ogg", audio_bytes, "audio/ogg"),
            )
            return transcription.text  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            print(f"Erro ao transcrever áudio: {exc}", flush=True)
            return None

except ImportError as exc:  # noqa: BLE001
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ─────────────────────────────── 3. DATABASE ─────────────────────────────── #
Base = declarative_base()


class Paciente(Base):
    __tablename__ = "pacientes"

    id: int = Column(Integer, primary_key=True)
    nome_completo: str | None = Column(String)
    telefone: str = Column(String, unique=True, nullable=False)
    endereco: str | None = Column(String)
    email: str | None = Column(String)
    data_nascimento: Date | None = Column(Date)

    agendamentos = relationship(
        "Agendamento", back_populates="paciente", cascade="all, delete-orphan"
    )
    historico = relationship(
        "HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan"
    )


class Agendamento(Base):
    __tablename__ = "agendamentos"

    id: int = Column(Integer, primary_key=True)
    paciente_id: int = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    data_hora: datetime = Column(DateTime(timezone=False), nullable=False)
    procedimento: str = Column(String, nullable=False)
    status: str = Column(String, default="confirmado")

    paciente = relationship("Paciente", back_populates="agendamentos")


class HistoricoConversa(Base):
    __tablename__ = "historico_conversas"

    id: int = Column(Integer, primary_key=True)
    paciente_id: int = Column(Integer, ForeignKey("pacientes.id"), nullable=False)
    role: str = Column(String, nullable=False)
    content: str = Column(Text, nullable=False)
    timestamp: datetime = Column(DateTime(timezone=True), default=get_now)

    paciente = relationship("Paciente", back_populates="historico")


class Procedimento(Base):
    __tablename__ = "procedimentos"

    id: int = Column(Integer, primary_key=True)
    nome: str = Column(String, unique=True, nullable=False)
    categoria: str = Column(String, index=True)
    valor_descritivo: str = Column(String, nullable=False)
    valor_base: float | None = Column(Float)


engine = create_engine(DATABASE_URL, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def criar_tabelas() -> None:
    Base.metadata.create_all(bind=engine)


# ───────────────────── 3.1 POPULATE PROCEDIMENTOS ────────────────────────── #
def popular_procedimentos_iniciais(db: Session) -> None:
    if db.query(Procedimento).first():
        return

    procedimentos_data = [
        {"categoria": "Procedimentos Básicos", "nome": "Consulta diagnóstica", "valor": "R$100 a R$162"},
        {"categoria": "Radiografias", "nome": "Raio-X periapical ou bite-wing", "valor": "R$15 a R$34"},
        {"categoria": "Radiografias", "nome": "Raio-X Panorâmica", "valor": "R$57 a R$115"},
        {"categoria": "Procedimentos Básicos", "nome": "Limpeza simples (Profilaxia)", "valor": "R$100 a R$400"},
        {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (1 face)", "valor": "a partir de R$100"},
        {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (2 faces)", "valor": "a partir de R$192"},
        {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Incisivo/Canino)", "valor": "R$517 a R$630"},
        {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Pré-molar/Molar)", "valor": "R$432 a R$876"},
        {"categoria": "Exodontia (Cirúrgicos)", "nome": "Extração simples de dente permanente", "valor": "R$150 a R$172"},
        {"categoria": "Exodontia (Cirúrgicos)", "nome": "Extração de dente de leite", "valor": "R$96 a R$102"},
        {"categoria": "Exodontia (Cirúrgicos)", "nome": "Extração de dente incluso/impactado", "valor": "R$364 a R$390"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa provisória", "valor": "R$150 a R$268"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa metalo-cerâmica", "valor": "R$576 a R$600"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa cerâmica pura", "valor": "R$576 a R$605"},
        {"categoria": "Clareamento Dentário", "nome": "Clareamento caseiro (por arcada)", "valor": "R$316 a R$330"},
        {"categoria": "Clareamento Dentário", "nome": "Clareamento em consultório (por arcada)", "valor": "R$316 a R$330"},
        {"categoria": "Implantes / Cirurgias", "nome": "Implante dentário unitário", "valor": "a partir de R$576"},
        {"categoria": "Implantes / Cirurgias", "nome": "Enxertos ósseos", "valor": "R$200 a R$800"},
        {"categoria": "Implantes / Cirurgias", "nome": "Levantamento de seio maxilar", "valor": "R$576 a R$800"},
    ]

    for p in procedimentos_data:
        nums = re.findall(r"\d+", p["valor"])
        valor_base = float(nums[0]) if nums else None
        db.add(
            Procedimento(
                nome=p["nome"],
                categoria=p["categoria"],
                valor_descritivo=p["valor"],
                valor_base=valor_base,
            )
        )
    db.commit()


# ─────────────────────────────── 4. HELPERS ──────────────────────────────── #
def limpar_rotulos(texto: str) -> str:
    for padrao in (
        r"^\s*A[cç]ão:\s*",
        r"^\s*Resumo:\s*",
        r"^\s*Action:\s*",
        r"^\s*Summary:\s*",
    ):
        texto = re.sub(padrao, "", texto, flags=re.I | re.M)
    return texto.strip()


def parse_data_nascimento(s: str) -> Optional[datetime]:
    """
    Converte datas DD/MM/AAAA (ou D/M/AA). Se falhar, usa dateparser (ordem DMY).
    """
    s = s.strip()
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s)
    if m:
        d, mth, y = map(int, m.groups())
        if y < 100:
            y += 1900 if y > 25 else 2000
        if 1 <= mth <= 12 and 1 <= d <= calendar.monthrange(y, mth)[1]:
            return datetime(y, mth, d)
    return parse_date(s, languages=["pt"], settings={"DATE_ORDER": "DMY"})


NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'´` ]{6,}$")  # ≥ 6 chars alfabéticos


def tentar_salvar_nome(db: Session, paciente: Paciente, texto: str) -> None:
    if paciente.nome_completo:
        return
    if NAME_RE.match(texto) and len(texto.split()) >= 2 and not re.search(r"\d", texto):
        paciente.nome_completo = texto.strip()
        db.commit()


def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente:
        paciente = Paciente(telefone=tel)
        db.add(paciente)
        db.commit()
        db.refresh(paciente)
    return paciente


def listar_todos_os_procedimentos(db: Session) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos:
        return "Não consegui encontrar a lista de procedimentos no momento."
    categorias: dict[str, list[str]] = defaultdict(list)
    for p in procedimentos:
        categorias[p.categoria].append(p.nome)
    resposta = "Oferecemos uma ampla gama de serviços! Nossos procedimentos incluem:\n\n"
    for cat, nomes in categorias.items():
        resposta += f"*{cat}*\n" + "\n".join(f"- {n}" for n in nomes) + "\n\n"
    return resposta.strip()


def consultar_precos_procedimentos(db: Session, termo_busca: str) -> str:
    termo = re.sub(r"[-.,]", " ", termo_busca.lower())
    filtros = [Procedimento.nome.ilike(f"%{p}%") for p in termo.split()]
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    if not resultados:
        return f"Não encontrei preços para '{termo_busca}'."
    linhas = []
    for r in resultados:
        linhas.append(
            f"O valor para {r.nome} é "
            + (
                f"a partir de R$ {int(r.valor_base):,}.00".replace(",", ".")
                if r.valor_base
                else r.valor_descritivo
            )
        )
    return "\n".join(linhas)


def consultar_horarios_disponiveis(db: Session, dia: str) -> str:
    dia_obj = parse_date(dia, languages=["pt"], settings={"TIMEZONE": "America/Sao_Paulo"})
    if not dia_obj:
        return "Não entendi a data."
    dia_naive = to_naive_sp(dia_obj)
    inicio = dia_naive.replace(hour=BUSINESS_START.hour, minute=0, second=0, microsecond=0)
    fim = dia_naive.replace(hour=BUSINESS_END.hour, minute=0, second=0, microsecond=0)
    ags = (
        db.query(Agendamento)
        .filter(Agendamento.data_hora.between(inicio, fim), Agendamento.status == "confirmado")
        .all()
    )
    ocupados = {a.data_hora.replace(second=0, microsecond=0) for a in ags}
    slots = []
    cur = inicio
    delta = timedelta(minutes=SLOT_MINUTES)
    while cur <= fim - delta:
        if cur not in ocupados and cur > get_now_naive():
            slots.append(cur)
        cur += delta
    if not slots:
        return "Infelizmente não há horários disponíveis nesse dia."
    return "Horários livres: " + ", ".join(s.strftime("%H:%M") for s in slots) + ". Qual prefere?"


# ───────────── 5. FUNÇÃO CENTRAL DE AGENDAMENTO ─────────────────────────── #
def processar_solicitacao_agendamento(
    db: Session,
    telefone_paciente: str,
    intencao: str,
    procedimento: Optional[str] = None,
    data_hora_texto: Optional[str] = None,
    confirmacao_usuario: bool = False,
    dados_paciente: Optional[Dict[str, str]] = None,
) -> str:
    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)

    # coleta / atualização de dados
    if dados_paciente:
        if nome := dados_paciente.get("nome_completo"):
            paciente.nome_completo = nome
        if email := dados_paciente.get("email"):
            paciente.email = email
        if endereco := dados_paciente.get("endereco"):
            paciente.endereco = endereco
        if dn := dados_paciente.get("data_nascimento"):
            dn_obj = parse_data_nascimento(dn)
            if dn_obj:
                paciente.data_nascimento = dn_obj.date()
            else:
                return "Formato de data inválido. Use DD/MM/AAAA."
        db.commit()

    faltando = [
        campo
        for campo, val in [
            ("nome completo", paciente.nome_completo),
            ("data de nascimento", paciente.data_nascimento),
            ("e-mail", paciente.email),
            ("endereço", paciente.endereco),
        ]
        if not val
    ]
    if faltando:
        return f"Ação: Continue o cadastro. Solicite {faltando[0]}."

    ags_ativos = (
        db.query(Agendamento)
        .filter(
            Agendamento.paciente_id == paciente.id,
            Agendamento.status == "confirmado",
            Agendamento.data_hora >= get_now_naive(),
        )
        .order_by(Agendamento.data_hora)
        .all()
    )

    # cancelar
    if intencao == "cancelar":
        if not ags_ativos:
            return "Você não possui agendamentos para cancelar."
        if len(ags_ativos) > 1 and not confirmacao_usuario:
            ids = ", ".join(str(a.id) for a in ags_ativos)
            return f"Ação: Liste e peça confirmação para cancelar. IDs: {ids}"
        for a in ags_ativos:
            a.status = "cancelado"
        db.commit()
        return "Sucesso! Agendamento cancelado."

    # agendar / reagendar
    if intencao in {"agendar", "reagendar"}:
        if not procedimento:
            return "Ação: Pergunte qual procedimento deseja agendar."
        proc_obj = db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedimento}%")).first()
        proc_real = proc_obj.nome if proc_obj else procedimento
        if not data_hora_texto:
            return f"Ação: Pergunte dia e horário para '{proc_real}'."
        dt_obj = parse_date(
            data_hora_texto,
            languages=["pt"],
            settings={"PREFER_DATES_FROM": "future", "TIMEZONE": "America/Sao_Paulo"},
        )
        if not dt_obj:
            return "Não entendi data/hora. Peça novamente."
        dt_naive = to_naive_sp(dt_obj)
        if not (BUSINESS_START <= dt_naive.time() <= BUSINESS_END):
            return "Atendemos apenas das 09:00 às 17:00."
        if db.query(Agendamento).filter_by(data_hora=dt_naive, status="confirmado").first():
            return "Horário ocupado. Use `consultar_horarios_disponiveis`."
        if not confirmacao_usuario:
            return (
                "Ação: Peça confirmação final. Resumo: Agendamento de "
                f"{proc_real} para {dt_naive.strftime('%d/%m/%Y às %H:%M')} com a Dra. "
                "Valéria Cristina Di Donato. Está correto?"
            )
        # confirmar
        if intencao == "agendar":
            db.add(Agendamento(paciente_id=paciente.id, data_hora=dt_naive, procedimento=proc_real))
        else:
            if not ags_ativos:
                return "Não há agendamento para reagendar."
            ags_ativos[0].data_hora = dt_naive
        db.commit()
        return f"Sucesso! Agendamento para {dt_naive.strftime('%d/%m/%Y às %H:%M')}."

    return "Não entendi a solicitação. Pode reformular?"


# ─────────────────────── 6. FUNÇÕES EXPONÍVEIS ───────────────────────────── #
available_functions: Dict[str, Any] = {
    "processar_solicitacao_agendamento": processar_solicitacao_agendamento,
    "listar_todos_os_procedimentos": listar_todos_os_procedimentos,
    "consultar_precos_procedimentos": consultar_precos_procedimentos,
    "consultar_horarios_disponiveis": consultar_horarios_disponiveis,
}

tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "processar_solicitacao_agendamento",
            "description": (
                "Ferramenta central para agendar, reagendar, cancelar e coletar dados."
            ),
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
            "description": "Lista todos os serviços oferecidos pela clínica.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_precos_procedimentos",
            "description": "Retorna preço de um procedimento específico.",
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
            "description": "Mostra horários livres em um dia informado.",
            "parameters": {
                "type": "object",
                "properties": {"dia": {"type": "string"}},
                "required": ["dia"],
            },
        },
    },
]

# ─────────────────────────────── 7. FASTAPI ──────────────────────────────── #
app = FastAPI(title="OdontoBot AI", version="14.0.3")


@app.on_event("startup")
async def on_startup() -> None:
    await asyncio.to_thread(criar_tabelas)
    with SessionLocal() as db:
        popular_procedimentos_iniciais(db)
    print("Tabelas OK.", flush=True)


@app.get("/")
def health_get() -> dict[str, str]:
    return {"status": "ok"}


@app.head("/")
def health_head() -> Response:
    return Response(status_code=200)


# ───────────────────────────── 8. WEBHOOK ────────────────────────────────── #
class ZapiText(BaseModel):
    message: Optional[str] = None


class ZapiAudio(BaseModel):
    audioUrl: Optional[str] = None


class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None
    audio: Optional[ZapiAudio] = None


async def enviar_resposta_whatsapp(telefone: str, mensagem: str) -> None:
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": telefone, "message": mensagem}
    headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print("Falha ao enviar via Z-API:", exc, flush=True)


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    raw = await request.json()
    print(">>> PAYLOAD:", raw, flush=True)

    try:
        payload = ZapiWebhookPayload(**raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, "Formato inválido") from exc

    tel = payload.phone
    user_msg: Optional[str] = None

    if payload.audio and payload.audio.audioUrl:
        texto = await transcrever_audio_whisper(payload.audio.audioUrl)
        if texto:
            user_msg = texto
        else:
            await enviar_resposta_whatsapp(tel, "Desculpe, não entendi seu áudio.")
            return {"status": "erro_transcricao"}
    elif payload.text and payload.text.message:
        user_msg = payload.text.message

    if not user_msg:
        await enviar_resposta_whatsapp(
            tel, "Olá! Sou Sofia, assistente da DI DONATO ODONTO. Como posso ajudar?"
        )
        return {"status": "saudacao"}

    paciente = buscar_ou_criar_paciente(db, tel)
    tentar_salvar_nome(db, paciente, user_msg)  # salvamento automático de nome
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=user_msg))
    db.commit()

    historico = (
        db.query(HistoricoConversa)
        .filter(
            HistoricoConversa.paciente_id == paciente.id,
            HistoricoConversa.timestamp >= get_now() - timedelta(hours=24),
            HistoricoConversa.role != "system",
        )
        .order_by(HistoricoConversa.timestamp)
        .all()
    )

    system_prompt = (
        "Você é Sofia, assistente virtual da clínica DI DONATO ODONTO (Dra. Valéria Cristina). "
        "Responda em português BR, nunca invente informações nem forneça diagnósticos. "
        "Para horários, use apenas a função `consultar_horarios_disponiveis`. "
        "Quando enviar nome completo, mantenha exatamente como o usuário digitou."
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    msgs.extend({"role": m.role, "content": m.content} for m in historico[-10:])
    msgs.append({"role": "user", "content": user_msg})

    try:
        model = "google/gemini-2.5-flash-preview-05-20"
        resp = openrouter_chat_completion(
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            top_p=0.8,
            max_tokens=1024,
        )
        ai_msg = resp.choices[0].message  # type: ignore[index]

        while getattr(ai_msg, "tool_calls", None):
            msgs_tool = msgs + [ai_msg]
            for call in ai_msg.tool_calls:  # type: ignore[attr-defined]
                fname = call.function.name
                fargs = json.loads(call.function.arguments)
                fn = available_functions.get(fname)
                if not fn:
                    raise HTTPException(500, f"Função desconhecida: {fname}")
                kwargs: Dict[str, Any] = {"db": db}
                if "telefone_paciente" in fn.__code__.co_varnames:  # type: ignore[attr-defined]
                    kwargs["telefone_paciente"] = tel
                result = fn(**kwargs, **fargs)
                msgs_tool.append(
                    {"tool_call_id": call.id, "role": "tool", "name": fname, "content": result}
                )
            resp = openrouter_chat_completion(
                model=model,
                messages=msgs_tool,
                temperature=0.2,
                top_p=0.8,
                max_tokens=1024,
            )
            ai_msg = resp.choices[0].message  # type: ignore[index]

        final = limpar_rotulos(ai_msg.content or "")

    except Exception as exc:  # noqa: BLE001
        print("Erro IA:", exc, flush=True)
        final = "Desculpe, houve um problema técnico. Tente de novo."

    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=final))
    db.commit()

    await enviar_resposta_whatsapp(tel, final)
    return {"status": "ok", "resposta": final}
