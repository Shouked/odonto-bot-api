"""
OdontoBot AI – main.py – v14.0.9
────────────────────────────────────────────────────────────────────────────
Novidades
─────────
14.0.9 • Fluxo reforçado: só pergunta dia/horário depois de ter NOME COMPLETO
         e PROCEDIMENTO. • `data_hora_texto` agora passa sempre primeiro por
         normalizar_data_relativa() — garante que “amanhã”, “próxima sexta”
         virem datas absolutas antes do parse.
14.0.8 • corrigiu parâmetro termo_busca em consultar_precos_procedimentos
14.0.7 • ajuste try/with, nome completo salvo, sem frases repetidas
"""

from __future__ import annotations

import asyncio, calendar, json, os, re
from collections import defaultdict
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional

import httpx, pytz
from dateparser import parse as parse_date
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import (
    Column, Date, DateTime, Float, ForeignKey, Integer, String, Text,
    and_, or_, create_engine
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ────────── 1. ENV / CONSTS ──────────
load_dotenv()
(
    DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY,
    ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN
) = (
    os.getenv("DATABASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENROUTER_API_KEY"),
    os.getenv("ZAPI_API_URL"), os.getenv("ZAPI_INSTANCE_ID"), os.getenv("ZAPI_TOKEN"), os.getenv("ZAPI_CLIENT_TOKEN")
)
if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL,
            ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente está vazia.")

BR_TZ = pytz.timezone("America/Sao_Paulo")
BUSINESS_START, BUSINESS_END, SLOT_MIN = time(9), time(17), 30

def now_tz()    -> datetime: return datetime.now(BR_TZ)
def now_naive() -> datetime: return datetime.now(BR_TZ).replace(tzinfo=None)
def to_naive_sp(dt: datetime) -> datetime:
    return (BR_TZ.localize(dt) if dt.tzinfo is None else dt.astimezone(BR_TZ)).replace(tzinfo=None)

# ────────── 2. OPENAI / OPENROUTER ──────────
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
    def chat_completion(**kw): return openrouter.chat.completions.create(**kw)
    async def transcrever_audio(url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(url); r.raise_for_status()
            tr = await asyncio.to_thread(
                openai_whisper.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.ogg", r.content, "audio/ogg"))
            return tr.text  # type: ignore[attr-defined]
        except Exception as exc:
            print("Erro Whisper:", exc); return None
except ImportError as exc:
    raise RuntimeError("Biblioteca openai não instalada.") from exc

# ────────── 3. SQLALCHEMY MODELS ──────────
Base = declarative_base()
class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True)
    nome_completo = Column(String)
    telefone = Column(String, unique=True, nullable=False)
    endereco = Column(String); email = Column(String)
    data_nascimento = Column(Date)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan")
    historico    = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")
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
    role = Column(String); content = Column(Text)
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
    try:    yield db
    finally: db.close()
def create_tables(): Base.metadata.create_all(bind=engine)

def seed_procedimentos(db: Session) -> None:
    if db.query(Procedimento).first(): return
    db.add(Procedimento(nome="Clareamento em consultório (por arcada)",
                        categoria="Clareamento Dentário",
                        valor_descritivo="R$316 a R$330",
                        valor_base=316))
    db.commit()

# ────────── 4. HELPERS ──────────
def limpar_rotulos(txt: str) -> str:
    for pat in (r"^\s*A[cç]ão:\s*", r"^\s*Resumo:\s*", r"^\s*Action:\s*", r"^\s*Summary:\s*"):
        txt = re.sub(pat, "", txt, flags=re.I|re.M)
    return txt.strip()

def parse_data_nasc(s: str) -> Optional[datetime]:
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s.strip())
    if m:
        d,mth,y = map(int, m.groups())
        y += 1900 if y<100 and y>25 else (2000 if y<100 else 0)
        if 1<=mth<=12 and 1<=d<=calendar.monthrange(y,mth)[1]:
            return datetime(y,mth,d)
    return parse_date(s, languages=["pt"], settings={"DATE_ORDER":"DMY"})

REL_MAP = {"hoje":0,"amanhã":1,"amanha":1,"depois de amanhã":2,"depois de amanha":2}
def normalizar_data_relativa(txt: str) -> str:
    low = txt.lower().strip()
    if low in REL_MAP:
        return (now_naive().date()+timedelta(days=REL_MAP[low])).strftime("%d/%m/%Y")
    prox = re.match(r"^pr[oó]xima\s+(\w+)$", low)
    if prox:
        dias=["segunda","terça","terca","quarta","quinta","sexta","sábado","sabado","domingo"]
        idx = dias.index(prox.group(1)) if prox.group(1) in dias else -1
        if idx>=0:
            delta=(idx-now_naive().weekday()+7)%7 or 7
            return (now_naive().date()+timedelta(days=delta)).strftime("%d/%m/%Y")
    return txt

NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'´` ]{6,}$")
def tentar_salvar_nome(db:Session, p:Paciente, msg:str)->None:
    if p.nome_completo: return
    cand=" ".join(msg.strip().split())
    if NAME_RE.match(cand) and len(cand.split())>=2 and not re.search(r"\d",cand):
        p.nome_completo=cand; db.commit(); db.refresh(p)

def buscar_ou_criar_paciente(db:Session,tel:str)->Paciente:
    p=db.query(Paciente).filter_by(telefone=tel).first()
    if not p:
        p=Paciente(telefone=tel)
        db.add(p); db.commit(); db.refresh(p)
    return p

def listar_todos_os_procedimentos(db:Session)->str:
    procs=db.query(Procedimento).order_by(Procedimento.categoria,Procedimento.nome).all()
    if not procs: return "Lista de procedimentos indisponível."
    cats:dict[str,list[str]]=defaultdict(list)
    for p in procs: cats[p.categoria].append(p.nome)
    return "\n\n".join(f"*{c}*\n"+"\n".join(f"- {n}" for n in ns) for c,ns in cats.items())

def consultar_precos_procedimentos(db:Session,termo_busca:str|None=None,termo:str|None=None)->str:
    query = termo_busca or termo or ""
    norm = re.sub(r"[-.,]", " ", query.lower())
    palavras = [w for w in norm.split() if w]
    if not palavras: return "Não entendi qual procedimento você procura."
    filtros=[Procedimento.nome.ilike(f"%{w}%") for w in palavras]
    res=db.query(Procedimento).filter(or_(*filtros)).all()
    if not res: return f"Não encontrei valores para '{query}'."
    res.sort(key=lambda p:sum(w in p.nome.lower() for w in palavras), reverse=True)
    out=[]
    for r in res[:3]:
        out.append(f"O valor para {r.nome} é "+
                   (f"a partir de R$ {int(r.valor_base):,}.00".replace(",", ".") if r.valor_base else r.valor_descritivo))
    return "\n".join(out)

def consultar_horarios_disponiveis(db:Session,dia:str)->str:
    dia=normalizar_data_relativa(dia)
    d=parse_date(dia,languages=["pt"],settings={"TIMEZONE":"America/Sao_Paulo"})
    if not d: return "Data inválida."
    d_naive=to_naive_sp(d)
    ini=d_naive.replace(hour=BUSINESS_START.hour,minute=0,second=0,microsecond=0)
    fim=d_naive.replace(hour=BUSINESS_END.hour,minute=0,second=0,microsecond=0)
    ocupados={a.data_hora.replace(second=0,microsecond=0) for a in
              db.query(Agendamento).filter(Agendamento.data_hora.between(ini,fim),
                                           Agendamento.status=="confirmado")}
    cur,slots=ini,[]
    delta=timedelta(minutes=SLOT_MIN)
    while cur<=fim-delta:
        if cur not in ocupados and cur>now_naive(): slots.append(cur)
        cur+=delta
    return ("Horários livres: "+", ".join(s.strftime("%H:%M") for s in slots)+"." if slots
            else "Sem horários livres nesse dia.")

def listar_agendamentos_ativos(db:Session,telefone_paciente:str)->str:
    p=buscar_ou_criar_paciente(db,telefone_paciente)
    ags=db.query(Agendamento).filter(Agendamento.paciente_id==p.id,
                                     Agendamento.status=="confirmado",
                                     Agendamento.data_hora>=now_naive()).all()
    return ("\n".join(f"ID {a.id}: {a.procedimento} – {a.data_hora:%d/%m/%Y às %H:%M}" for a in ags)
            if ags else "Você não possui agendamentos ativos.")

# ────────── 5. FUNÇÃO CENTRAL ──────────
def processar_solicitacao_agendamento(
    db:Session, telefone_paciente:str, intencao:str,
    procedimento:Optional[str]=None, data_hora_texto:Optional[str]=None,
    confirmacao_usuario:bool=False, dados_paciente:Optional[Dict[str,str]]=None,
)->str:
    p = buscar_ou_criar_paciente(db, telefone_paciente)

    # Atualiza paciente
    if dados_paciente:
        if nome:=dados_paciente.get("nome_completo"): p.nome_completo=nome.strip()
        if email:=dados_paciente.get("email"): p.email=email
        if end:=dados_paciente.get("endereco"): p.endereco=end
        if dn_str:=dados_paciente.get("data_nascimento"):
            dn=parse_data_nasc(dn_str)
            if not dn: return "Formato de data inválido. Use DD/MM/AAAA."
            p.data_nascimento=dn.date()
        db.commit(); db.refresh(p)

    pendentes=[c for c,v in [
        ("nome completo", p.nome_completo), ("data de nascimento", p.data_nascimento),
        ("e-mail", p.email), ("endereço", p.endereco)
    ] if not v]
    if pendentes:
        return f"Ação: Solicite {pendentes[0]} de forma natural, sem repetir 'para finalizar o cadastro'."

    ags=db.query(Agendamento).filter(Agendamento.paciente_id==p.id,
                                     Agendamento.status=="confirmado",
                                     Agendamento.data_hora>=now_naive()).order_by(Agendamento.data_hora).all()

    if intencao=="cancelar":
        if not ags: return "Você não possui agendamentos para cancelar."
        if len(ags)>1 and not confirmacao_usuario:
            return "Ação: Peça quais IDs cancelar. IDs: "+", ".join(str(a.id) for a in ags)
        for a in ags: a.status="cancelado"
        db.commit(); return "Agendamento cancelado."

    if intencao in {"agendar","reagendar"}:
        # 1️⃣ precisa ter nome completo antes de qualquer coisa
        if not p.nome_completo:
            return "Ação: Solicite nome completo primeiro."
        # 2️⃣ precisa do procedimento antes de dia/horário
        if not procedimento:
            return "Ação: Pergunte qual procedimento o paciente deseja."

        proc_obj=db.query(Procedimento).filter(Procedimento.nome.ilike(f"%{procedimento}%")).first()
        proc_real = proc_obj.nome if proc_obj else procedimento

        # 3️⃣ pergunta data/hora somente depois de ter procedimento
        if not data_hora_texto:
            return f"Ação: Pergunte data e horário para '{proc_real}'."

        # normaliza expressão relativa ANTES de validar
        data_hora_texto = normalizar_data_relativa(data_hora_texto)

        if re.search(r"[A-Za-z]", data_hora_texto):
            return "Informe apenas data e horário (ex.: 20/06 às 10:00)."

        dt=parse_date(data_hora_texto, languages=["pt"],
                      settings={"PREFER_DATES_FROM":"future","TIMEZONE":"America/Sao_Paulo"})
        if not dt: return "Data/hora inválida."
        dt_naive=to_naive_sp(dt)

        if not BUSINESS_START<=dt_naive.time()<=BUSINESS_END:
            return "Atendemos das 09:00 às 17:00."
        if db.query(Agendamento).filter_by(data_hora=dt_naive,status="confirmado").first():
            return "Horário ocupado. Use `consultar_horarios_disponiveis`."

        if not confirmacao_usuario:
            return ("Ação: Peça confirmação final. Resumo: "
                    f"{proc_real} em {dt_naive:%d/%m/%Y às %H:%M}. Está correto?")

        if intencao=="agendar":
            db.add(Agendamento(paciente_id=p.id,data_hora=dt_naive,procedimento=proc_real))
        else:
            if not ags: return "Nenhum agendamento para reagendar."
            ags[0].data_hora=dt_naive
        db.commit(); return f"Sucesso! Agendado para {dt_naive:%d/%m/%Y às %H:%M}."

    return "Desculpe, não entendi a solicitação."

# ────────── 6. TOOLS ──────────
available_functions:Dict[str,Any]={
    "processar_solicitacao_agendamento":processar_solicitacao_agendamento,
    "listar_todos_os_procedimentos":listar_todos_os_procedimentos,
    "consultar_precos_procedimentos":consultar_precos_procedimentos,
    "consultar_horarios_disponiveis":consultar_horarios_disponiveis,
    "listar_agendamentos_ativos":listar_agendamentos_ativos,
}
tools=[
    {
        "type":"function",
        "function":{
            "name":"processar_solicitacao_agendamento",
            "description":"Fluxo completo de agenda e cadastro.",
            "parameters":{
                "type":"object",
                "properties":{
                    "intencao":{"type":"string","enum":["agendar","reagendar","cancelar","coletar_dados"]},
                    "procedimento":{"type":"string"},
                    "data_hora_texto":{"type":"string"},
                    "confirmacao_usuario":{"type":"boolean"},
                    "dados_paciente":{
                        "type":"object",
                        "properties":{
                            "nome_completo":{"type":"string"},"email":{"type":"string"},
                            "data_nascimento":{"type":"string"},"endereco":{"type":"string"},
                        },
                    },
                },
                "required":["intencao"],
            },
        },
    },
    {"type":"function","function":{"name":"listar_todos_os_procedimentos",
        "description":"Lista procedimentos","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"consultar_precos_procedimentos",
        "description":"Preço de procedimento",
        "parameters":{"type":"object","properties":{"termo_busca":{"type":"string"}},"required":["termo_busca"]}}},
    {"type":"function","function":{"name":"consultar_horarios_disponiveis",
        "description":"Horários livres","parameters":{"type":"object","properties":{"dia":{"type":"string"}},"required":["dia"]}}},
    {"type":"function","function":{"name":"listar_agendamentos_ativos",
        "description":"Mostra agendamentos","parameters":{"type":"object","properties":{}}}},
]

# ────────── 7. FASTAPI ──────────
app = FastAPI(title="OdontoBot AI", version="14.0.9")
@app.on_event("startup")
async def startup()->None:
    await asyncio.to_thread(create_tables)
    with SessionLocal() as db: seed_procedimentos(db)
    print("🟢 Banco pronto")

@app.get("/")  def health()->dict[str,str]: return{"status":"ok"}
@app.head("/") def health_h()->Response:    return Response(status_code=200)

# ────────── 8. WEBHOOK ──────────
class ZapiText(BaseModel):  message:Optional[str]=None
class ZapiAudio(BaseModel): audioUrl:Optional[str]=None
class ZapiWebhookPayload(BaseModel):
    phone:str; text:Optional[ZapiText]=None; audio:Optional[ZapiAudio]=None

async def send_whatsapp(phone:str,msg:str)->None:
    url=f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload={"phone":phone,"message":msg}
    headers={"Content-Type":"application/json","Client-Token":ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as c:
        try: await c.post(url,json=payload,headers=headers)
        except Exception as exc: print("Erro Z-API:",exc,flush=True)

@app.post("/whatsapp/webhook")
async def webhook(req:Request,db:Session=Depends(get_db))->Dict[str,Any]:
    raw=await req.json(); print("payload",raw)
    try: pay=ZapiWebhookPayload(**raw)
    except Exception as exc: raise HTTPException(422,"payload inválido") from exc
    tel=pay.phone; user_msg:Optional[str]=None
    if pay.audio and pay.audio.audioUrl: user_msg=await transcrever_audio(pay.audio.audioUrl)
    elif pay.text and pay.text.message:  user_msg=pay.text.message
    if not user_msg:
        await send_whatsapp(tel,"Olá! Sou Sofia, assistente da DI DONATO ODONTO. Como posso ajudar?")
        return{"status":"saudacao"}

    pac=buscar_ou_criar_paciente(db,tel); tentar_salvar_nome(db,pac,user_msg)
    db.add(HistoricoConversa(paciente_id=pac.id,role="user",content=user_msg)); db.commit()

    hist=db.query(HistoricoConversa).filter(
        HistoricoConversa.paciente_id==pac.id,
        HistoricoConversa.timestamp>=now_tz()-timedelta(hours=24),
        HistoricoConversa.role!="system").order_by(HistoricoConversa.timestamp).all()

    sys_prompt=("Você é Sofia, assistente virtual da clínica DI DONATO ODONTO (Dra. Valéria Cristina). "
                "Responda em português BR. Colete dados na ordem: nome completo → procedimento → data/hora. "
                "Use as funções somente quando necessário e converta termos como 'amanhã' em datas absolutas.")

    msgs=[{"role":"system","content":sys_prompt}]
    msgs+= [{"role":m.role,"content":m.content} for m in hist[-10:]]
    msgs.append({"role":"user","content":user_msg})

    try:
        resp=chat_completion(model="google/gemini-2.5-flash-preview-05-20",
                             messages=msgs,tools=tools,tool_choice="auto",
                             temperature=0.2,top_p=0.8,max_tokens=1024)
        ai_msg=resp.choices[0].message  # type: ignore[index]
        while getattr(ai_msg,"tool_calls",None):
            chain=msgs+[ai_msg]
            for call in ai_msg.tool_calls:  # type: ignore[attr-defined]
                fname=call.function.name
                args=json.loads(call.function.arguments)
                kwargs={"db":db}
                if "telefone_paciente" in available_functions[fname].__code__.co_varnames:  # type: ignore[attr-defined]
                    kwargs["telefone_paciente"]=tel
                result=available_functions[fname](**kwargs,**args)
                chain.append({"tool_call_id":call.id,"role":"tool","name":fname,"content":result})
            resp=chat_completion(model="google/gemini-2.5-flash-preview-05-20",
                                 messages=chain,temperature=0.2,top_p=0.8,max_tokens=1024)
            ai_msg=resp.choices[0].message  # type: ignore[index]

        final=limpar_rotulos(ai_msg.content or "")
    except Exception as exc:
        print("Erro IA:",exc); final="Desculpe, ocorreu um problema técnico."

    db.add(HistoricoConversa(paciente_id=pac.id,role="assistant",content=final)); db.commit()
    await send_whatsapp(tel,final); return{"status":"ok","resposta":final}
