"""
API principal do OdontoBot AI. Versão de Demonstração para Clínica.

FastAPI que serve de webhook para a Z-API, processa mensagens de WhatsApp via
OpenAI e interage com o banco para gerenciar pacientes e agendamentos.
"""

import os, json, asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, or_, and_
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from collections import defaultdict

# ───────────────── 1. VARIÁVEIS DE AMBIENTE ────────────────── #
load_dotenv()
DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN = (
    os.getenv("DATABASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENROUTER_API_KEY"),
    os.getenv("ZAPI_API_URL"), os.getenv("ZAPI_INSTANCE_ID"), os.getenv("ZAPI_TOKEN"), os.getenv("ZAPI_CLIENT_TOKEN")
)
if not all([DATABASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY, ZAPI_API_URL, ZAPI_INSTANCE_ID, ZAPI_TOKEN, ZAPI_CLIENT_TOKEN]):
    raise RuntimeError("Alguma variável de ambiente obrigatória não foi definida.")

# ───────────────── 2. CONFIGURAÇÃO DOS CLIENTES DE IA ───────── #
try:
    import openai
    openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, default_headers={"HTTP-Referer": "https://github.com/Shouked/odonto-bot-api", "X-Title": "OdontoBot AI"})
    def openrouter_chat_completion(**kw): return openrouter_client.chat.completions.create(**kw)
    async def transcrever_audio_whisper(audio_url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=30)
                response.raise_for_status()
                audio_bytes = response.content
            transcription = await asyncio.to_thread(openai_whisper_client.audio.transcriptions.create, model="whisper-1", file=("audio.ogg", audio_bytes, "audio/ogg"))
            return transcription.text
        except Exception as e: print(f"Erro ao transcrever áudio: {e}", flush=True); return None
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 3. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()
class Paciente(Base): __tablename__ = "pacientes"; id, nome, telefone = Column(Integer, primary_key=True), Column(String), Column(String, unique=True, nullable=False); agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan"); historico = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")
class Agendamento(Base): __tablename__ = "agendamentos"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); data_hora, procedimento, status = Column(DateTime, nullable=False), Column(String, nullable=False), Column(String, default="confirmado"); paciente = relationship("Paciente", back_populates="agendamentos")
class HistoricoConversa(Base): __tablename__ = "historico_conversas"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); role, content, timestamp = Column(String, nullable=False), Column(Text, nullable=False), Column(DateTime, default=datetime.utcnow); paciente = relationship("Paciente", back_populates="historico")
class Procedimento(Base): __tablename__ = "procedimentos"; id = Column(Integer, primary_key=True); nome = Column(String, unique=True, nullable=False); categoria = Column(String, index=True); valor_descritivo = Column(String, nullable=False)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
def get_db(): db = SessionLocal();_ = db;yield db;db.close()
def criar_tabelas(): Base.metadata.create_all(bind=engine)
def popular_procedimentos_iniciais(db: Session):
    if db.query(Procedimento).first(): return
    print("Populando tabela de procedimentos pela primeira vez...", flush=True)
    procedimentos_data = [{"categoria": "Procedimentos Básicos", "nome": "Consulta diagnóstica", "valor": "R$100 a R$162"}, {"categoria": "Radiografias", "nome": "Raio-X periapical ou bite-wing", "valor": "R$15 a R$34"}, {"categoria": "Radiografias", "nome": "Raio-X Panorâmica", "valor": "R$57 a R$115"}, {"categoria": "Procedimentos Básicos", "nome": "Limpeza simples (Profilaxia)", "valor": "R$100 a R$400 (média R$150–400)"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (1 face)", "valor": "a partir de R$100"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (2 faces)", "valor": "a partir de R$192"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Incisivo/Canino)", "valor": "R$517 a R$630"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Pré-molar/Molar)", "valor": "R$432 a R$876"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração simples de dente permanente", "valor": "R$150 a R$172"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente de leite", "valor": "R$96 a R$102"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente incluso/impactado", "valor": "R$364 a R$390"}, {"categoria": "Próteses e Coroas", "nome": "Coroa provisória", "valor": "R$150 a R$268"}, {"categoria": "Próteses e Coroas", "nome": "Coroa metalo-cerâmica", "valor": "R$576 a R$600"}, {"categoria": "Próteses e Coroas", "nome": "Coroa cerâmica pura", "valor": "R$576 a R$605"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento caseiro (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento em consultório (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Implante dentário unitário (coroa + pilar)", "valor": "a partir de R$576"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Enxertos ósseos", "valor": "R$200 a R$800"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Levantamento de seio maxilar (sinus lift)", "valor": "R$576 a R$800"}]
    for p_data in procedimentos_data: db.add(Procedimento(nome=p_data["nome"], categoria=p_data["categoria"], valor_descritivo=p_data["valor"]))
    db.commit()

# ───────────────── 4. FERRAMENTAS ──────────────────────────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente: paciente = Paciente(telefone=tel, nome=f"Paciente {tel}"); db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente
def agendar_consulta(db: Session, telefone_paciente: str, data_hora_agendamento: str, procedimento: str) -> str:
    try: dt = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    if dt < datetime.now(): return f"Não é possível agendar no passado."
    if dt.weekday() >= 5: return f"A clínica não funciona aos fins de semana."
    if not (9 <= dt.hour < 18): return f"O horário de funcionamento é das 09:00 às 18:00."
    pac = buscar_ou_criar_paciente(db, telefone_paciente); db.add(Agendamento(paciente_id=pac.id, data_hora=dt, procedimento=procedimento)); db.commit()
    return f"Perfeito! Agendamento para '{procedimento}' confirmado para {dt.strftime('%d/%m/%Y às %H:%M')}."
def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.data_hora >= datetime.now(), Agendamento.status == "confirmado").order_by(Agendamento.data_hora).all()
    if not ags: return "Você não possui agendamentos futuros."
    linhas = [f"- ID {a.id}: {a.procedimento} em {a.data_hora.strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos são:\n" + "\n".join(linhas)
def cancelar_agendamento(db: Session, telefone_paciente: str, id_agendamento: int) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente); ag = db.query(Agendamento).filter_by(id=id_agendamento, paciente_id=pac.id).first()
    if not ag: return f"Agendamento ID {id_agendamento} não encontrado."
    if ag.status == "cancelado": return "Esse agendamento já está cancelado."
    ag.status = "cancelado"; db.commit()
    return f"Agendamento {id_agendamento} cancelado com sucesso."
def consultar_e_reagendar_inteligente(db: Session, telefone_paciente: str, novo_data_hora_agendamento: str) -> str:
    pac = buscar_ou_criar_paciente(db, telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.data_hora >= datetime.now(), Agendamento.status == "confirmado").all()
    if not ags: return "Você não tem nenhum agendamento futuro para reagendar."
    if len(ags) > 1: return "Encontrei mais de um agendamento. Qual deles você gostaria de reagendar? Informe o ID.\n" + consultar_meus_agendamentos(db, telefone_paciente)
    ag_reagendar = ags[0]
    try: nova_dt = datetime.strptime(novo_data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "O formato da nova data e hora é inválido. Use AAAA-MM-DD HH:MM."
    if nova_dt < datetime.now(): return f"Não é possível reagendar para o passado."
    if nova_dt.weekday() >= 5: return f"A clínica não funciona aos fins de semana."
    if not (9 <= nova_dt.hour < 18): return f"O horário de funcionamento é das 09:00 às 18:00."
    id_antigo = ag_reagendar.id; ag_reagendar.data_hora = nova_dt; db.commit()
    return f"Pronto! Seu agendamento (ID {id_antigo}) foi reagendado para {nova_dt.strftime('%d/%m/%Y às %H:%M')}."

# [NOVO] Ferramenta para listar todos os procedimentos
def listar_todos_os_procedimentos(db: Session, telefone_paciente: str) -> str:
    """Lista todos os procedimentos e suas categorias oferecidos pela clínica."""
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui encontrar a lista de procedimentos no momento."
    
    categorias = defaultdict(list)
    for p in procedimentos: categorias[p.categoria].append(p.nome)
    
    resposta = "Oferecemos uma ampla gama de serviços para cuidar do seu sorriso! Nossos procedimentos incluem:\n\n"
    for categoria, nomes in categorias.items():
        resposta += f"**{categoria}**\n"
        for nome in nomes: resposta += f"- {nome}\n"
        resposta += "\n"
    return resposta

# [MELHORADO] Busca robusta por palavra-chave
def consultar_precos_procedimentos(db: Session, telefone_paciente: str, termo_busca: str) -> str:
    """Consulta o preço de um ou mais procedimentos com base em palavras-chave."""
    palavras_chave = termo_busca.split()
    filtros = [Procedimento.nome.ilike(f'%{palavra}%') for palavra in palavras_chave]
    
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    
    if not resultados:
        return f"Não encontrei informações de valores para '{termo_busca}'. Posso tentar buscar por outro termo?"
    
    linhas = [f"- {r.nome}: {r.valor_descritivo}" for r in resultados]
    return f"Encontrei os seguintes valores para '{termo_busca}':\n" + "\n".join(linhas)

available_functions = {"agendar_consulta": agendar_consulta, "consultar_meus_agendamentos": consultar_meus_agendamentos, "cancelar_agendamento": cancelar_agendamento, "consultar_e_reagendar_inteligente": consultar_e_reagendar_inteligente, "listar_todos_os_procedimentos": listar_todos_os_procedimentos, "consultar_precos_procedimentos": consultar_precos_procedimentos}
tools = [{"type": "function", "function": {"name": "agendar_consulta", "description": "Agenda uma nova consulta.", "parameters": {"type": "object", "properties": {"data_hora_agendamento": {"type": "string"}, "procedimento": {"type": "string"}}, "required": ["data_hora_agendamento", "procedimento"]}}},
         {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Lista agendamentos futuros do paciente, com IDs.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "cancelar_agendamento", "description": "Cancela um agendamento por ID.", "parameters": {"type": "object", "properties": {"id_agendamento": {"type": "integer"}}, "required": ["id_agendamento"]}}},
         {"type": "function", "function": {"name": "consultar_e_reagendar_inteligente", "description": "Ferramenta inteligente para reagendar uma consulta.", "parameters": {"type": "object", "properties": {"novo_data_hora_agendamento": {"type": "string", "description": "Nova data/hora no formato AAAA-MM-DD HH:MM."}}, "required": ["novo_data_hora_agendamento"]}}},
         {"type": "function", "function": {"name": "listar_todos_os_procedimentos", "description": "Lista todos os serviços e procedimentos oferecidos pela clínica. Use quando o usuário fizer uma pergunta geral sobre o que a clínica faz.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "consultar_precos_procedimentos", "description": "Consulta preços de procedimentos. Use quando o usuário perguntar 'quanto custa', 'valor', 'preço'.", "parameters": {"type": "object", "properties": {"termo_busca": {"type": "string", "description": "O procedimento que o usuário quer saber o preço, ex: 'limpeza' ou 'raio x'."}}, "required": ["termo_busca"]}}}]

# ───────────────── 5. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp com OpenRouter e Whisper.", version="4.1.0-demo")
@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(criar_tabelas)
    print("Tabelas verificadas/criadas.", flush=True)
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
    telefone = payload.phone
    mensagem_usuario = None
    if payload.audio and payload.audio.audioUrl:
        texto_transcrito = await transcrever_audio_whisper(payload.audio.audioUrl)
        if texto_transcrito: mensagem_usuario = texto_transcrito; print(f">>> Texto transcrito: '{texto_transcrito}'", flush=True)
        else: await enviar_resposta_whatsapp(telefone, "Desculpe, não consegui entender o seu áudio."); return {"status": "erro_transcricao"}
    elif payload.text and payload.text.message:
        mensagem_usuario = payload.text.message
    if not mensagem_usuario:
        await enviar_resposta_whatsapp(telefone, "Olá! Sou o assistente virtual da DI DONATO ODONTO. Como posso te ajudar hoje?")
        return {"status": "ignorado", "motivo": "sem conteúdo processável"}

    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)); db.commit()
    
    # [CORRIGIDO] Lógica de histórico para não salvar o prompt do sistema
    historico_recente = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= datetime.utcnow() - timedelta(hours=24), HistoricoConversa.role != 'system').order_by(HistoricoConversa.timestamp).all()
    
    NOME_CLINICA, PROFISSIONAL = "DI DONATO ODONTO", "Dra. Valéria Cristina Di Donato"
    system_prompt = (f"Você é OdontoBot, assistente virtual da {NOME_CLINICA}, onde os atendimentos são realizados pela {PROFISSIONAL}. "
                     f"Seja sempre educado, prestativo e conciso. Hoje é {datetime.now().strftime('%d/%m/%Y')}. "
                     "Use as ferramentas para responder. Se pedirem conselhos médicos, recuse educadamente e diga que apenas a doutora pode fornecer essa orientação na consulta.")
    
    mensagens_para_ia: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in historico_recente: mensagens_para_ia.append({"role": msg.role, "content": msg.content})

    try:
        modelo_chat = "google/gemini-flash-1.5-preview-0514"
        resp = openrouter_chat_completion(model=modelo_chat, messages=mensagens_para_ia, tools=tools, tool_choice="auto")
        ai_msg = resp.choices[0].message
        while ai_msg.tool_calls:
            msgs_com_ferramentas = mensagens_para_ia + [ai_msg]
            for call in ai_msg.tool_calls:
                fname, f_args = call.function.name, json.loads(call.function.arguments)
                func = available_functions.get(fname)
                if not func: raise HTTPException(500, f"Função desconhecida: {fname}")
                result = func(db=db, telefone_paciente=telefone, **f_args)
                msgs_com_ferramentas.append({"tool_call_id": call.id, "role": "tool", "name": fname, "content": result})
            resp = openrouter_chat_completion(model=modelo_chat, messages=msgs_com_ferramentas)
            ai_msg = resp.choices[0].message
        resposta_final = ai_msg.content
    except Exception as e:
        print("Erro na interação IA:", e, flush=True); resposta_final = "Desculpe, ocorreu um problema técnico. Tente novamente."
    db.add(HistoricoConversa(paciente_id=paciente.id, role="assistant", content=resposta_final)); db.commit()
    await enviar_resposta_whatsapp(telefone, resposta_final)
    return {"status": "ok", "resposta": resposta_final}
