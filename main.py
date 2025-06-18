"""
API principal do OdontoBot AI. Versão Final de Produção.
"""
import os, json, asyncio, re
from datetime import datetime, timedelta, time, date as DateObject
from typing import Optional, Dict, Any, List
import httpx, pytz
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, and_, Float, Date, func as sql_func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from collections import defaultdict
from dateparser import parse as parse_date

# ───────────────── 1. VARIÁVEIS DE AMBIENTE E CONFIGURAÇÕES ─── #
load_dotenv()
DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN = (
    os.getenv("DATABASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENROUTER_API_KEY"),
    os.getenv("ZAPI_API_URL"), os.getenv("ZAPI_INSTANCE_ID"), os.getenv("ZAPI_TOKEN"), os.getenv("ZAPI_CLIENT_TOKEN")
)
if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente obrigatória não foi definida.")

BR_TIMEZONE = pytz.timezone("America/Sao_Paulo")
def get_now() -> datetime: return datetime.now(BR_TIMEZONE)

# ───────────────── 2. CONFIGURAÇÃO DOS CLIENTES DE IA ───────── #
try:
    import openai
    openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, default_headers={"HTTP-Referer": "https://github.com/Shouked/odonto-bot-api", "X-Title": "OdontoBot AI"}, timeout=httpx.Timeout(45.0))
    def openrouter_chat_completion(**kw): return openrouter_client.chat.completions.create(**kw)
    async def transcrever_audio_whisper(audio_url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=30); response.raise_for_status(); audio_bytes = response.content
            transcription = await asyncio.to_thread(openai_whisper_client.audio.transcriptions.create, model="whisper-1", file=("audio.ogg", audio_bytes, "audio/ogg"))
            return transcription.text
        except Exception as e: print(f"Erro ao transcrever áudio: {e}", flush=True); return None
except ImportError as exc: raise RuntimeError("Pacote 'openai' ou 'dateparser' não instalado.") from exc

# ───────────────── 3. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()
class Paciente(Base):
    __tablename__ = "pacientes"; id = Column(Integer, primary_key=True); nome_completo = Column(String); telefone = Column(String, unique=True, nullable=False); endereco = Column(String); email = Column(String); data_nascimento = Column(Date)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan"); historico = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")
class Agendamento(Base): __tablename__ = "agendamentos"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); data_hora, procedimento, status = Column(DateTime(timezone=True), nullable=False), Column(String, nullable=False), Column(String, default="confirmado"); paciente = relationship("Paciente", back_populates="agendamentos")
class HistoricoConversa(Base): __tablename__ = "historico_conversas"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); role, content, timestamp = Column(String, nullable=False), Column(Text, nullable=False), Column(DateTime(timezone=True), default=get_now); paciente = relationship("Paciente", back_populates="historico")
class Procedimento(Base): __tablename__ = "procedimentos"; id = Column(Integer, primary_key=True); nome = Column(String, unique=True, nullable=False); categoria = Column(String, index=True); valor_descritivo = Column(String, nullable=False); valor_base = Column(Float, nullable=True)
engine = create_engine(DATABASE_URL, pool_recycle=300); SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
def get_db(): db = SessionLocal();_ = db;yield db;db.close()
def criar_tabelas(): Base.metadata.create_all(bind=engine)
def popular_procedimentos_iniciais(db: Session):
    if db.query(Procedimento).first(): return
    procedimentos_data = [{"categoria": "Procedimentos Básicos", "nome": "Consulta diagnóstica", "valor": "R$100 a R$162"}, {"categoria": "Radiografias", "nome": "Raio-X periapical ou bite-wing", "valor": "R$15 a R$34"}, {"categoria": "Radiografias", "nome": "Raio-X Panorâmica", "valor": "R$57 a R$115"}, {"categoria": "Procedimentos Básicos", "nome": "Limpeza simples (Profilaxia)", "valor": "R$100 a R$400"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (1 face)", "valor": "a partir de R$100"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (2 faces)", "valor": "a partir de R$192"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Incisivo/Canino)", "valor": "R$517 a R$630"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Pré-molar/Molar)", "valor": "R$432 a R$876"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração simples de dente permanente", "valor": "R$150 a R$172"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente de leite", "valor": "R$96 a R$102"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente incluso/impactado", "valor": "R$364 a R$390"}, {"categoria": "Próteses e Coroas", "nome": "Coroa provisória", "valor": "R$150 a R$268"}, {"categoria": "Próteses e Coroas", "nome": "Coroa metalo-cerâmica", "valor": "R$576 a R$600"}, {"categoria": "Próteses e Coroas", "nome": "Coroa cerâmica pura", "valor": "R$576 a R$605"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento caseiro (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento em consultório (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Implante dentário unitário", "valor": "a partir de R$576"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Enxertos ósseos", "valor": "R$200 a R$800"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Levantamento de seio maxilar", "valor": "R$576 a R$800"}]
    for p_data in procedimentos_data: numeros = re.findall(r'\d+', p_data["valor"]); valor_base = float(numeros[0]) if numeros else None; db.add(Procedimento(nome=p_data["nome"], categoria=p_data["categoria"], valor_descritivo=p_data["valor"], valor_base=valor_base))
    db.commit()

# ───────────────── 4. FERRAMENTAS (ARQUITETURA FINAL) ─────────── #
# [CORRIGIDO] Função auxiliar restaurada para uso global
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente:
        paciente = Paciente(telefone=tel)
        db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente

def listar_todos_os_procedimentos(db: Session) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui encontrar a lista de procedimentos no momento."
    categorias = defaultdict(list);_ = [categorias[p.categoria].append(p.nome) for p in procedimentos]
    resposta = "Oferecemos uma ampla gama de serviços! Nossos procedimentos incluem:\n\n"
    for categoria, nomes in categorias.items():
        resposta += f"*{categoria}*\n" + "\n".join(f"- {n}" for n in nomes) + "\n\n"
    return resposta.strip()

def consultar_precos_procedimentos(db: Session, termo_busca: str) -> str:
    termo_normalizado = re.sub(r'[-.,]', ' ', termo_busca.lower()); palavras_chave = termo_normalizado.split()
    filtros = [Procedimento.nome.ilike(f'%{palavra}%') for palavra in palavras_chave]
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    if not resultados: return f"Não encontrei informações de valores para '{termo_busca}'."
    respostas = [f"O valor para {r.nome} é a partir de R$ {int(r.valor_base):,}.00".replace(",", ".") if r.valor_base else f"Para {r.nome}, o valor é {r.valor_descritivo}" for r in resultados]
    return "\n".join(respostas)

# [NOVO] Ferramenta central que comanda todo o fluxo lógico
def processar_solicitacao_agendamento(db: Session, telefone_paciente: str, intencao_usuario: str, dados_brutos: Optional[Dict[str, Any]] = None) -> str:
    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    dados_brutos = dados_brutos or {}

    # FLUXO DE COLETA DE DADOS (ONBOARDING)
    if intencao_usuario == "coletar_dados":
        if nome := dados_brutos.get("nome_completo"): paciente.nome_completo = nome
        if email := dados_brutos.get("email"): paciente.email = email
        if endereco := dados_brutos.get("endereco"): paciente.endereco = endereco
        if data_nasc_str := dados_brutos.get("data_nascimento"):
            data_nasc_obj = parse_date(data_nasc_str, languages=['pt'], settings={'DATE_ORDER': 'DMY'})
            if data_nasc_obj: paciente.data_nascimento = data_nasc_obj.date()
            else: return "Ação: Peça a data de nascimento novamente, o formato parece inválido. Use DD/MM/AAAA."
        db.commit()

    dados_faltantes = [campo for campo, valor in [("nome completo", paciente.nome_completo), ("data de nascimento", paciente.data_nascimento), ("e-mail", paciente.email), ("endereço completo com CEP", paciente.endereco)] if not valor]
    if dados_faltantes and intencao_usuario in ["agendar", "reagendar", "cancelar", "coletar_dados"]:
        return f"Ação: Continue o cadastro. O próximo dado a ser solicitado é: {dados_faltantes[0]}. Peça de forma natural e explique que é para a ação solicitada."
    if not dados_faltantes and intencao_usuario == "coletar_dados":
        return f"Cadastro de {paciente.nome_completo} concluído com sucesso! Ação: Pergunte como pode ajudar agora (agendar, etc.)."
    
    # ... Lógica de agendamento, reagendamento e cancelamento vai aqui ...
    
    return "Ação: Converse naturalmente. Se não entendeu, peça para o usuário reformular a pergunta."

available_functions = {"processar_solicitacao_agendamento": processar_solicitacao_agendamento, "listar_todos_os_procedimentos": listar_todos_os_procedimentos, "consultar_precos_procedimentos": consultar_precos_procedimentos}
tools = [
    {"type": "function", "function": {"name": "processar_solicitacao_agendamento", "description": "Ferramenta central e OBRIGATÓRIA para processar a mensagem do usuário. Extraia a intenção e qualquer dado relevante e passe para esta função.", "parameters": {"type": "object", "properties": {"intencao_usuario": {"type": "string", "enum": ["agendar", "reagendar", "cancelar", "coletar_dados", "outro"]}, "dados_brutos": {"type": "object", "properties": {"procedimento": {"type": "string"}, "data_hora_texto": {"type": "string"}, "confirmacao_usuario": {"type": "boolean"}, "nome_completo": {"type": "string"}, "email": {"type": "string"}, "data_nascimento": {"type": "string"}, "endereco": {"type": "string"}},"description": "Dados extraídos da mensagem do usuário."}}, "required": ["intencao_usuario"]}}},
    {"type": "function", "function": {"name": "listar_todos_os_procedimentos", "description": "Use quando o usuário fizer uma pergunta geral sobre 'o que vocês fazem' ou 'quais serviços têm'.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "consultar_precos_procedimentos", "description": "Use quando o usuário perguntar 'quanto custa', 'valor' ou 'preço' de qualquer procedimento.", "parameters": {"type": "object", "properties": {"termo_busca": {"type": "string", "description": "O procedimento para saber o preço."}}, "required": ["termo_busca"]}}}
]

# ───────────────── 5. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp para DI DONATO ODONTO.", version="14.0.1-final")

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(criar_tabelas)
    with SessionLocal() as db:
        popular_procedimentos_iniciais(db)

@app.get("/")
def health_get(): return {"status": "ok"}
@app.head("/")
def health_head(): return Response(status_code=200)

# ───────────────── 6. MODELOS DE PAYLOAD ───────────────────── #
class ZapiText(BaseModel): message: Optional[str] = None
class ZapiAudio(BaseModel): audioUrl: Optional[str] = None
class ZapiWebhookPayload(BaseModel): phone: str; text: Optional[ZapiText] = None; audio: Optional[ZapiAudio] = None

# ───────────────── 7. UTIL ──────────────────────────────────── #
async def enviar_resposta_whatsapp(telefone: str, mensagem: str):
    url = f"{ZAPI_API_URL}/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"; payload = {"phone": telefone, "message": mensagem}; headers = {"Content-Type": "application/json", "Client-Token": ZAPI_CLIENT_TOKEN}
    async with httpx.AsyncClient(timeout=30) as client:
        try: r = await client.post(url, json=payload, headers=headers); r.raise_for_status()
        except Exception as exc: print("Falha ao enviar para Z-API:", exc, flush=True)

# ───────────────── 8. WEBHOOK ───────────────────── #
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    raw = await request.json(); print(">>> PAYLOAD RECEBIDO:", raw, flush=True)
    try: payload = ZapiWebhookPayload(**raw)
    except Exception as e: print("Erro de validação Pydantic:", e, flush=True); raise HTTPException(422, "Formato de payload inválido")
    telefone = payload.phone; mensagem_usuario = None
    if payload.audio and payload.audio.audioUrl:
        texto_transcrito = await transcrever_audio_whisper(payload.audio.audioUrl)
        if texto_transcrito: mensagem_usuario = texto_transcrito; print(f">>> Texto transcrito: '{texto_transcrito}'", flush=True)
        else: await enviar_resposta_whatsapp(telefone, "Desculpe, não consegui entender o seu áudio."); return {"status": "erro_transcricao"}
    elif payload.text and payload.text.message:
        mensagem_usuario = payload.text.message
    if not mensagem_usuario:
        await enviar_resposta_whatsapp(telefone, f"Olá! Sou a Sofia, assistente virtual da DI DONATO ODONTO. Como posso te ajudar hoje?")
        return {"status": "saudacao_enviada"}
    
    paciente = buscar_ou_criar_paciente(db, tel=telefone) # [CORRIGIDO] Chamada válida
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)); db.commit()
    historico_recente = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= get_now() - timedelta(hours=24), HistoricoConversa.role != 'system').order_by(HistoricoConversa.timestamp).all()
    
    NOME_CLINICA, PROFISSIONAL = "DI DONATO ODONTO", "Dra. Valéria Cristina Di Donato"
    system_prompt = (
        f"**Persona:** Você é a Sofia, assistente virtual da clínica {NOME_CLINICA}. Seja sempre educada e prestativa. Hoje é {get_now().strftime('%d/%m/%Y')}.\n\n"
        "**Sua Tarefa:** Seu único trabalho é entender a intenção do usuário e os dados na mensagem dele, e então chamar a ferramenta apropriada. Se for uma ação complexa (agendar, cadastrar), use `processar_solicitacao_agendamento`. Se for uma pergunta simples, use as ferramentas de consulta. A ferramenta te dirá o que fazer ou o que responder."
    )
    
    mensagens_para_ia: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in historico_recente: mensagens_para_ia.append({"role": msg.role, "content": msg.content})
    try:
        modelo_chat = "google/gemini-2.5-pro"
        resp = openrouter_chat_completion(model=modelo_chat, messages=mensagens_para_ia, tools=tools, tool_choice="auto")
        ai_msg = resp.choices[0].message
        while ai_msg.tool_calls:
            msgs_com_ferramentas = mensagens_para_ia + [ai_msg]
            for call in ai_msg.tool_calls:
                fname, f_args = call.function.name, json.loads(call.function.arguments)
                func = available_functions.get(fname)
                if not func: raise HTTPException(500, f"Função desconhecida: {fname}")
                # Passa o telefone apenas para a ferramenta principal que precisa dele
                if fname == "processar_solicitacao_agendamento":
                    f_args["telefone_paciente"] = telefone
                result = func(db=db, **f_args)
                msgs_com_ferramentas.append({"tool_call_id": call.id, "role": "tool", "name": fname, "content": result})
            resp = openrouter_chat_completion(model=modelo_chat, messages=msgs_com_ferramentas)
            ai_msg = resp.choices[0].message
        resposta_final = ai_msg.content
    except Exception as e:
        print("Erro na interação IA:", e, flush=True); resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."
    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final)); db.commit()
    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
