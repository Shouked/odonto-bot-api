"""
API principal do OdontoBot AI — versão 14.0.1 (hot-fix).
Correções:
• remove rótulos “Ação/Resumo” das respostas
• grava horários como naïve (sem fuso) e compara corretamente
• elimina duplo return no webhook
"""

from __future__ import annotations

import asyncio
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
    func as sql_func,
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
    """Retorna timezone-aware (legado)."""
    return datetime.now(BR_TIMEZONE)


# NOVO ▸ utilidades naïve (sem tzinfo)
def get_now_naive() -> datetime:
    return datetime.now(BR_TIMEZONE).replace(tzinfo=None)


def to_naive_sp(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = BR_TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(BR_TIMEZONE)
    return dt.replace(tzinfo=None)


def get_today_br() -> DateObject:
    return get_now().date()


def get_tomorrow_br() -> DateObject:
    return get_today_br() + timedelta(days=1)


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
                response = await client.get(audio_url, timeout=30)
                response.raise_for_status()
                audio_bytes = response.content
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
    raise RuntimeError("Pacote 'openai' ou dependências não instaladas.") from exc

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

    # ALTERADO ▸ timezone=False  -> grava naïve
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


# ─────────────────── 3.1 POPULA PROCEDIMENTOS INICIAIS ────────────────────── #
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
        {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração simples de dente permanente", "valor": "R$150 a R$172"},
        {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente de leite", "valor": "R$96 a R$102"},
        {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente incluso/impactado", "valor": "R$364 a R$390"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa provisória", "valor": "R$150 a R$268"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa metalo-cerâmica", "valor": "R$576 a R$600"},
        {"categoria": "Próteses e Coroas", "nome": "Coroa cerâmica pura", "valor": "R$576 a R$605"},
        {"categoria": "Clareamento Dentário", "nome": "Clareamento caseiro (por arcada)", "valor": "R$316 a R$330"},
        {"categoria": "Clareamento Dentário", "nome": "Clareamento em consultório (por arcada)", "valor": "R$316 a R$330"},
        {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Implante dentário unitário", "valor": "a partir de R$576"},
        {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Enxertos ósseos", "valor": "R$200 a R$800"},
        {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Levantamento de seio maxilar", "valor": "R$576 a R$800"},
    ]

    for p_data in procedimentos_data:
        numeros = re.findall(r"\d+", p_data["valor"])
        valor_base = float(numeros[0]) if numeros else None
        db.add(
            Procedimento(
                nome=p_data["nome"],
                categoria=p_data["categoria"],
                valor_descritivo=p_data["valor"],
                valor_base=valor_base,
            )
        )
    db.commit()


# ─────────────────────────────── 4. TOOLS ─────────────────────────────────── #
def limpar_rotulos(texto: str) -> str:
    """Remove rótulos internos 'Ação:' / 'Resumo:' etc. da resposta."""
    padroes = [
        r"^\s*A[cç]ão:\s*",
        r"^\s*Resumo:\s*",
        r"^\s*Action:\s*",
        r"^\s*Summary:\s*",
    ]
    for p in padroes:
        texto = re.sub(p, "", texto, flags=re.IGNORECASE | re.MULTILINE)
    return texto.strip()


def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:  # noqa: D401
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
    for categoria, nomes in categorias.items():
        resposta += f"*{categoria}*\n"
        for nome in nomes:
            resposta += f"- {nome}\n"
        resposta += "\n"
    return resposta.strip()


def consultar_precos_procedimentos(db: Session, termo_busca: str) -> str:
    termo_normalizado = re.sub(r"[-.,]", " ", termo_busca.lower())
    palavras_chave = termo_normalizado.split()
    filtros = [Procedimento.nome.ilike(f"%{palavra}%") for palavra in palavras_chave]
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    if not resultados:
        return f"Não encontrei informações de valores para '{termo_busca}'."

    respostas = []
    for r in resultados:
        if r.valor_base:
            respostas.append(f"O valor para {r.nome} é a partir de R$ {int(r.valor_base):,}.00".replace(",", "."))
        else:
            respostas.append(f"Para {r.nome}, o valor é {r.valor_descritivo}")
    return "\n".join(respostas)


# NOVO: disponibilidade de horários
def consultar_horarios_disponiveis(db: Session, dia: str) -> str:
    """Retorna horários livres em slots de 30 min, das 09:00 às 17:00."""
    dia_obj = parse_date(
        dia,
        languages=["pt"],
        settings={
            "TO_TIMEZONE": "America/Sao_Paulo",
            "TIMEZONE": "America/Sao_Paulo",
        },
    )
    if not dia_obj:
        return "Não entendi a data para verificar horários."

    dia_naive = to_naive_sp(dia_obj)

    dia_inicio = dia_naive.replace(hour=BUSINESS_START.hour, minute=0, second=0, microsecond=0)
    dia_fim = dia_naive.replace(hour=BUSINESS_END.hour, minute=0, second=0, microsecond=0)

    ags = (
        db.query(Agendamento)
        .filter(
            Agendamento.data_hora >= dia_inicio,
            Agendamento.data_hora < dia_fim,
            Agendamento.status == "confirmado",
        )
        .all()
    )

    ocupados: set[datetime] = {a.data_hora.replace(second=0, microsecond=0) for a in ags}
    slots: list[datetime] = []
    cursor = dia_inicio
    delta = timedelta(minutes=SLOT_MINUTES)
    while cursor <= dia_fim - delta:
        if cursor not in ocupados and cursor > get_now_naive():
            slots.append(cursor)
        cursor += delta

    if not slots:
        return "Infelizmente não há horários disponíveis nesse dia."

    formatted = ", ".join(s.strftime("%H:%M") for s in slots[:16])
    return f"Horários livres: {formatted}. Qual deles você prefere?"


# ─────────── ferramenta central de agendamento & onboarding ─────────────── #
def processar_solicitacao_agendamento(
    db: Session,
    telefone_paciente: str,
    intencao: str,
    procedimento: Optional[str] = None,
    data_hora_texto: Optional[str] = None,
    confirmacao_usuario: bool = False,
    dados_paciente: Optional[Dict[str, str]] = None,
) -> str:
    """Gerencia fluxo completo de agendamento."""

    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)

    # 1. coleta de dados do paciente
    if dados_paciente:
        if nome := dados_paciente.get("nome_completo"):
            paciente.nome_completo = nome
        if email := dados_paciente.get("email"):
            paciente.email = email
        if endereco := dados_paciente.get("endereco"):
            paciente.endereco = endereco
        if data_nasc_str := dados_paciente.get("data_nascimento"):
            data_nasc_obj = parse_date(data_nasc_str, languages=["pt"])
            if data_nasc_obj:
                paciente.data_nascimento = data_nasc_obj.date()
            else:
                return "Formato de data de nascimento inválido. Peça novamente no formato DD/MM/AAAA."
        db.commit()

    dados_faltantes = [
        campo
        for campo, valor in [
            ("nome completo", paciente.nome_completo),
            ("data de nascimento", paciente.data_nascimento),
            ("e-mail", paciente.email),
            ("endereço", paciente.endereco),
        ]
        if not valor
    ]
    if dados_faltantes:
        return f"Ação: Continue o cadastro. O próximo dado a ser solicitado é: {dados_faltantes[0]}. Peça de forma natural."

    # 2. intenção do usuário
    agendamentos_ativos = (
        db.query(Agendamento)
        .filter(
            Agendamento.paciente_id == paciente.id,
            Agendamento.status == "confirmado",
            Agendamento.data_hora >= get_now_naive(),
        )
        .order_by(Agendamento.data_hora)
        .all()
    )

    # Cancelar agendamento
    if intencao == "cancelar":
        if not agendamentos_ativos:
            return "Você não possui agendamentos para cancelar."
        if len(agendamentos_ativos) > 1 and not confirmacao_usuario:
            return (
                "Ação: Liste os agendamentos e peça para o usuário confirmar qual(is) deseja cancelar. IDs: "
                + ", ".join(str(ag.id) for ag in agendamentos_ativos)
            )
        for ag in agendamentos_ativos:
            ag.status = "cancelado"
        db.commit()
        return "Sucesso! O(s) agendamento(s) foi(ram) cancelado(s)."

    # Agendar / Reagendar
    if intencao in {"agendar", "reagendar"}:
        if not procedimento:
            return "Ação: Pergunte qual procedimento o paciente deseja agendar."

        proc_obj = db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedimento}%")).first()
        procedimento_real = proc_obj.nome if proc_obj else procedimento

        if not data_hora_texto:
            return f"Ação: Pergunte o dia e hora para o agendamento de '{procedimento_real}'."

        dt_agendamento_obj = parse_date(
            data_hora_texto,
            languages=["pt"],
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": "America/Sao_Paulo",
                "TO_TIMEZONE": "America/Sao_Paulo",
            },
        )
        if not dt_agendamento_obj:
            return "Não consegui entender a data e hora. Peça para o usuário tentar novamente de outra forma."

        dt_agendamento_naive = to_naive_sp(dt_agendamento_obj)

        if not (BUSINESS_START <= dt_agendamento_naive.time() <= BUSINESS_END):
            return "Atendemos apenas das 09:00 às 17:00. Por favor, escolha um horário nesse intervalo."

        conflito = (
            db.query(Agendamento)
            .filter_by(data_hora=dt_agendamento_naive, status="confirmado")
            .first()
        )
        if conflito:
            return (
                f"Desculpe, o horário de {dt_agendamento_naive.strftime('%H:%M')} já está ocupado. "
                "Use a ferramenta `consultar_horarios_disponiveis` para ver outras opções."
            )

        if not confirmacao_usuario:
            resumo = (
                "Ação: Peça a confirmação final ao usuário. Resumo: Agendamento de "
                f"{procedimento_real} para {dt_agendamento_naive.strftime('%d/%m/%Y às %H:%M')} com a Dra. "
                "Valéria Cristina Di Donato. Está correto?"
            )
            return resumo

        # confirmação positiva → executar
        if intencao == "agendar":
            db.add(
                Agendamento(
                    paciente_id=paciente.id,
                    data_hora=dt_agendamento_naive,
                    procedimento=procedimento_real,
                )
            )
            db.commit()
            return (
                f"Sucesso! Agendamento para '{procedimento_real}' criado para "
                f"{dt_agendamento_naive.strftime('%d/%m/%Y às %H:%M')}."
            )
        elif intencao == "reagendar":
            if not agendamentos_ativos:
                return "Não há agendamento para reagendar."
            agendamentos_ativos[0].data_hora = dt_agendamento_naive
            db.commit()
            return (
                f"Sucesso! Agendamento reagendado para {dt_agendamento_naive.strftime('%d/%m/%Y às %H:%M')}."
            )

    return "Não entendi a solicitação. Por favor, pergunte ao usuário o que ele deseja fazer."


# ───────────────────────── FUNÇÕES DISPONÍVEIS ───────────────────────────── #
available_functions: dict[str, Any] = {
    "processar_solicitacao_agendamento": processar_solicitacao_agendamento,
    "listar_todos_os_procedimentos": listar_todos_os_procedimentos,
    "consultar_precos_procedimentos": consultar_precos_procedimentos,
    "consultar_horarios_disponiveis": consultar_horarios_disponiveis,
}

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "processar_solicitacao_agendamento",
            "description": (
                "Ferramenta central para agendar, reagendar, cancelar e coletar dados. "
                "Passe a intenção do usuário e os dados coletados."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intencao": {
                        "type": "string",
                        "enum": ["agendar", "reagendar", "cancelar", "coletar_dados"],
                    },
                    "procedimento": {"type": "string", "description": "Procedimento desejado."},
                    "data_hora_texto": {"type": "string", "description": "Data e hora solicitadas."},
                    "confirmacao_usuario": {
                        "type": "boolean",
                        "description": "True se o usuário confirmou o resumo.",
                    },
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
            "description": "Use quando o usuário perguntar sobre serviços da clínica.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_precos_procedimentos",
            "description": "Use para informar preços ou valores de algum procedimento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "termo_busca": {"type": "string", "description": "Procedimento desejado."}
                },
                "required": ["termo_busca"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_horarios_disponiveis",
            "description": "Use para exibir horários disponíveis em um dia informado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dia": {"type": "string", "description": "Data (ex.: 'amanhã', '20/06/2025')."}
                },
                "required": ["dia"],
            },
        },
    },
]

# ─────────────────────────────── 5. FASTAPI ──────────────────────────────── #
app = FastAPI(title="OdontoBot AI", version="14.0.1")


@app.on_event("startup")
async def on_startup() -> None:
    await asyncio.to_thread(criar_tabelas)
    with SessionLocal() as db:
        popular_procedimentos_iniciais(db)
    print("Tabelas verificadas e procedimentos prontos.", flush=True)


@app.get("/")
def health_get() -> dict[str, str]:
    return {"status": "ok"}


@app.head("/")
def health_head() -> Response:
    return Response(status_code=200)


# ─────────────────────────────── 6. MODELS ──────────────────────────────── #
class ZapiText(BaseModel):
    message: Optional[str] = None


class ZapiAudio(BaseModel):
    audioUrl: Optional[str] = None


class ZapiWebhookPayload(BaseModel):
    phone: str
    text: Optional[ZapiText] = None
    audio: Optional[ZapiAudio] = None


# ─────────────────────────────── 7. UTILS ────────────────────────────────── #
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


# ─────────────────────────────── 8. WEBHOOK ──────────────────────────────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    raw = await request.json()
    print(">>> PAYLOAD RECEBIDO:", raw, flush=True)

    try:
        payload = ZapiWebhookPayload(**raw)
    except Exception as exc:  # noqa: BLE001
        print("Erro de validação Pydantic:", exc, flush=True)
        raise HTTPException(422, "Formato de payload inválido")

    telefone = payload.phone
    mensagem_usuario: Optional[str] = None

    if payload.audio and payload.audio.audioUrl:
        texto_transcrito = await transcrever_audio_whisper(payload.audio.audioUrl)
        if texto_transcrito:
            mensagem_usuario = texto_transcrito
            print(f">>> Texto transcrito: '{texto_transcrito}'", flush=True)
        else:
            await enviar_resposta_whatsapp(
                telefone, "Desculpe, não consegui entender o seu áudio."
            )
            return {"status": "erro_transcricao"}
    elif payload.text and payload.text.message:
        mensagem_usuario = payload.text.message

    if not mensagem_usuario:
        await enviar_resposta_whatsapp(
            telefone,
            "Olá! Sou a Sofia, assistente virtual da DI DONATO ODONTO. Como posso te ajudar hoje?",
        )
        return {"status": "saudacao_enviada"}

    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    db.add(
        HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)
    )
    db.commit()

    historico_recente = (
        db.query(HistoricoConversa)
        .filter(
            HistoricoConversa.paciente_id == paciente.id,
            HistoricoConversa.timestamp >= get_now() - timedelta(hours=24),
            HistoricoConversa.role != "system",
        )
        .order_by(HistoricoConversa.timestamp)
        .all()
    )

    NOME_CLINICA, PROFISSIONAL = "DI DONATO ODONTO", "Dra. Valéria Cristina Di Donato"

    system_prompt = (
        f"Você é Sofia, assistente virtual da clínica {NOME_CLINICA}, onde a responsável é {PROFISSIONAL}. "
        "Responda SEMPRE em português do Brasil e nunca invente informações. "
        "Se não souber algo, informe que não possui essa informação. "
        "Nunca forneça diagnósticos ou recomendações médicas. Hoje é "
        f"{get_now().strftime('%d/%m/%Y')}.\n\n"
        "### Regras de Fluxo:\n"
        "1. Para perguntas sobre serviços/preços use as ferramentas apropriadas.\n"
        "2. Para quaisquer ações de agenda use somente `processar_solicitacao_agendamento`.\n"
        "3. Siga à risca a instrução retornada pelas ferramentas.\n"
        "4. Quando retornar um resumo de confirmação, aguarde a resposta do usuário e marque `confirmacao_usuario=true` na próxima chamada."
    )

    mensagens_para_ia: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    mensagens_para_ia.extend(
        {"role": msg.role, "content": msg.content} for msg in historico_recente[-10:]
    )
    mensagens_para_ia.append({"role": "user", "content": mensagem_usuario})

    try:
        modelo_chat = "google/gemini-2.5-flash-preview-05-20"
        resp = openrouter_chat_completion(
            model=modelo_chat,
            messages=mensagens_para_ia,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            top_p=0.8,
            max_tokens=1024,
        )

        ai_msg = resp.choices[0].message  # type: ignore[index]

        # Loop até a IA não chamar mais ferramentas
        while getattr(ai_msg, "tool_calls", None):
            msgs_com_ferramentas = mensagens_para_ia + [ai_msg]
            for call in ai_msg.tool_calls:  # type: ignore[attr-defined]
                fname = call.function.name
                f_args = json.loads(call.function.arguments)
                func = available_functions.get(fname)
                if not func:
                    raise HTTPException(500, f"Função desconhecida: {fname}")

                # Monta kwargs obrigatórios
                kwargs: dict[str, Any] = {"db": db}
                if "telefone_paciente" in func.__code__.co_varnames:  # type: ignore[attr-defined]
                    kwargs["telefone_paciente"] = telefone

                result = func(**kwargs, **f_args)
                msgs_com_ferramentas.append(
                    {
                        "tool_call_id": call.id,
                        "role": "tool",
                        "name": fname,
                        "content": result,
                    }
                )

            resp = openrouter_chat_completion(
                model=modelo_chat,
                messages=msgs_com_ferramentas,
                temperature=0.2,
                top_p=0.8,
                max_tokens=1024,
            )
            ai_msg = resp.choices[0].message  # type: ignore[index]

        resposta_final = limpar_rotulos(ai_msg.content or "")

    except Exception as exc:  # noqa: BLE001
        print("Erro na interação IA:", exc, flush=True)
        resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente mais tarde."

    db.add(
        HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final)
    )
    db.commit()

    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
