"""
API principal do OdontoBot AI. Versão Final para Demonstração.
"""

import os, json, asyncio, re
from datetime import datetime, timedelta, time
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, and_, Float, Date, func as sql_func
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
    openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, default_headers={"HTTP-Referer": "https://github.com/Shouked/odonto-bot-api", "X-Title": "OdontoBot AI"}, timeout=httpx.Timeout(40.0))
    def openrouter_chat_completion(**kw): return openrouter_client.chat.completions.create(**kw)
    async def transcrever_audio_whisper(audio_url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=30); response.raise_for_status(); audio_bytes = response.content
            transcription = await asyncio.to_thread(openai_whisper_client.audio.transcriptions.create, model="whisper-1", file=("audio.ogg", audio_bytes, "audio/ogg"))
            return transcription.text
        except Exception as e: print(f"Erro ao transcrever áudio: {e}", flush=True); return None
except ImportError as exc:
    raise RuntimeError("Pacote 'openai' não instalado.") from exc

# ───────────────── 3. BANCO DE DADOS ───────────────────────── #
Base = declarative_base()
class Paciente(Base):
    __tablename__ = "pacientes"
    id = Column(Integer, primary_key=True); nome_completo = Column(String, nullable=True); telefone = Column(String, unique=True, nullable=False); endereco = Column(String, nullable=True); email = Column(String, nullable=True); data_nascimento = Column(Date, nullable=True)
    agendamentos = relationship("Agendamento", back_populates="paciente", cascade="all, delete-orphan"); historico = relationship("HistoricoConversa", back_populates="paciente", cascade="all, delete-orphan")
class Agendamento(Base): __tablename__ = "agendamentos"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); data_hora, procedimento, status = Column(DateTime, nullable=False), Column(String, nullable=False), Column(String, default="confirmado"); paciente = relationship("Paciente", back_populates="agendamentos")
class HistoricoConversa(Base): __tablename__ = "historico_conversas"; id, paciente_id = Column(Integer, primary_key=True), Column(Integer, ForeignKey("pacientes.id"), nullable=False); role, content, timestamp = Column(String, nullable=False), Column(Text, nullable=False), Column(DateTime, default=datetime.utcnow); paciente = relationship("Paciente", back_populates="historico")
# [ATUALIZADO] Tabela de Procedimentos com valor base
class Procedimento(Base):
    __tablename__ = "procedimentos"
    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
    categoria = Column(String, index=True)
    valor_descritivo = Column(String, nullable=False)
    valor_base = Column(Float, nullable=True) # Nova coluna para o preço inicial

engine = create_engine(DATABASE_URL, pool_pre_ping=True); SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
def get_db(): db = SessionLocal();_ = db;yield db;db.close()
def criar_tabelas(): Base.metadata.create_all(bind=engine)
# [ATUALIZADO] Função para popular a tabela de procedimentos com valor base
def popular_procedimentos_iniciais(db: Session):
    if db.query(Procedimento).first(): return
    print("Populando tabela de procedimentos com valores base...", flush=True)
    procedimentos_data = [{"categoria": "Procedimentos Básicos", "nome": "Consulta diagnóstica", "valor": "R$100 a R$162"}, {"categoria": "Radiografias", "nome": "Raio-X periapical ou bite-wing", "valor": "R$15 a R$34"}, {"categoria": "Radiografias", "nome": "Raio-X Panorâmica", "valor": "R$57 a R$115"}, {"categoria": "Procedimentos Básicos", "nome": "Limpeza simples (Profilaxia)", "valor": "R$100 a R$400"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (1 face)", "valor": "a partir de R$100"}, {"categoria": "Restaurações (Obturações)", "nome": "Restauração de Resina (2 faces)", "valor": "a partir de R$192"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Incisivo/Canino)", "valor": "R$517 a R$630"}, {"categoria": "Endodontia (Canal)", "nome": "Tratamento de Canal (Pré-molar/Molar)", "valor": "R$432 a R$876"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração simples de dente permanente", "valor": "R$150 a R$172"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente de leite", "valor": "R$96 a R$102"}, {"categoria": "Exodontia (Procedimentos Cirúrgicos)", "nome": "Extração de dente incluso/impactado", "valor": "R$364 a R$390"}, {"categoria": "Próteses e Coroas", "nome": "Coroa provisória", "valor": "R$150 a R$268"}, {"categoria": "Próteses e Coroas", "nome": "Coroa metalo-cerâmica", "valor": "R$576 a R$600"}, {"categoria": "Próteses e Coroas", "nome": "Coroa cerâmica pura", "valor": "R$576 a R$605"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento caseiro (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Clareamento Dentário", "nome": "Clareamento em consultório (por arcada)", "valor": "R$316 a R$330"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Implante dentário unitário", "valor": "a partir de R$576"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Enxertos ósseos", "valor": "R$200 a R$800"}, {"categoria": "Implantes e Cirurgias Ósseas", "nome": "Levantamento de seio maxilar", "valor": "R$576 a R$800"}]
    for p_data in procedimentos_data:
        numeros = re.findall(r'\d+', p_data["valor"])
        valor_base = float(numeros[0]) if numeros else None
        db.add(Procedimento(nome=p_data["nome"], categoria=p_data["categoria"], valor_descritivo=p_data["valor"], valor_base=valor_base))
    db.commit()

# ───────────────── 4. FERRAMENTAS ──────────────────────────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente: paciente = Paciente(telefone=tel); db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente
def obter_proxima_etapa(db: Session, telefone_paciente: str) -> str:
    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    if not paciente.nome_completo: return "Ação: Peça o nome completo do paciente. Apresente-se e explique que precisa do nome para iniciar o cadastro."
    if not paciente.data_nascimento: return "Ação: Agradeça pelo nome e peça a data de nascimento no formato AAAA-MM-DD."
    if not paciente.email: return "Ação: Agradeça e peça o e-mail do paciente."
    if not paciente.endereco: return "Ação: Agradeça e peça o endereço completo."
    return "Ação: Cadastro completo. Pergunte qual procedimento o paciente deseja agendar ou sobre o qual deseja informações."
def atualizar_dados_paciente(db: Session, telefone_paciente: str, nome_completo: Optional[str] = None, endereco: Optional[str] = None, email: Optional[str] = None, data_nascimento: Optional[str] = None) -> str:
    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    campos_atualizados = []
    if nome_completo and not paciente.nome_completo: paciente.nome_completo = nome_completo; campos_atualizados.append("nome")
    if endereco and not paciente.endereco: paciente.endereco = endereco; campos_atualizados.append("endereço")
    if email and not paciente.email: paciente.email = email; campos_atualizados.append("email")
    if data_nascimento and not paciente.data_nascimento:
        try: paciente.data_nascimento = datetime.strptime(data_nascimento, "%Y-%m-%d").date(); campos_atualizados.append("data de nascimento")
        except ValueError: return "Formato de data de nascimento inválido. Peça novamente no formato AAAA-MM-DD."
    db.commit()
    return f"Dados atualizados com sucesso: {', '.join(campos_atualizados)}." if campos_atualizados else "Nenhum dado novo foi fornecido para atualização."
def agendar_consulta(db: Session, telefone_paciente: str, data_hora_agendamento: str, procedimento: str) -> str:
    try: dt = datetime.strptime(data_hora_agendamento, "%Y-%m-%d %H:%M")
    except ValueError: return "Formato de data/hora inválido. Use AAAA-MM-DD HH:MM."
    if dt < datetime.now(): return "Não é possível agendar no passado."
    if dt.weekday() >= 5: return "A clínica não funciona aos fins de semana."
    if not (9 <= dt.hour < 18): return "O horário de funcionamento é das 09:00 às 18:00."
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente); db.add(Agendamento(paciente_id=pac.id, data_hora=dt, procedimento=procedimento)); db.commit()
    return f"Sucesso! Agendamento para '{procedimento}' criado para {dt.strftime('%d/%m/%Y às %H:%M')}."
def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.status == "confirmado", Agendamento.data_hora >= datetime.now()).order_by(Agendamento.data_hora).all()
    if not ags: return "Você não possui agendamentos futuros."
    linhas = [f"- ID {a.id}: {a.procedimento} em {a.data_hora.strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos são:\n" + "\n".join(linhas)
def consultar_horarios_disponiveis(db: Session, telefone_paciente: str, dia: str) -> str:
    try: data_consulta = datetime.strptime(dia, "%Y-%m-%d").date()
    except ValueError: return "Formato de data inválido. Use AAAA-MM-DD."
    agendamentos_do_dia = db.query(Agendamento.data_hora).filter(sql_func.date(Agendamento.data_hora) == data_consulta, Agendamento.status == 'confirmado').all()
    horarios_ocupados = {ag.data_hora.time() for ag in agendamentos_do_dia}
    slots_possiveis = {time(h) for h in range(9, 18)}
    horarios_disponiveis = sorted(list(slots_possiveis - horarios_ocupados))
    if not horarios_disponiveis: return f"Não há mais horários disponíveis para o dia {data_consulta.strftime('%d/%m/%Y')}."
    horarios_formatados = [t.strftime('%H:%M') for t in horarios_disponiveis]
    return f"Os horários livres para o dia {data_consulta.strftime('%d/%m/%Y')} são: {', '.join(horarios_formatados)}."
def consultar_e_cancelar_inteligente(db: Session, telefone_paciente: str, dica: Optional[str] = None) -> str:
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    query = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.status == "confirmado", Agendamento.data_hora >= datetime.now())
    if dica: query = query.filter(Agendamento.procedimento.ilike(f'%{dica}%'))
    ags = query.all()
    if not ags: return "Não encontrei um agendamento futuro para cancelar."
    if len(ags) > 1: return "Encontrei mais de um agendamento. Qual deles você gostaria de cancelar? Informe o ID.\n" + consultar_meus_agendamentos(db, telefone_paciente)
    ag_cancelar = ags[0]; ag_cancelar.status = "cancelado"; db.commit()
    return f"Ok, cancelei seu agendamento de {ag_cancelar.procedimento} do dia {ag_cancelar.data_hora.strftime('%d/%m/%Y às %H:%M')}."
def listar_todos_os_procedimentos(db: Session, telefone_paciente: str) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui encontrar a lista de procedimentos no momento."
    categorias = defaultdict(list);_ = [categorias[p.categoria].append(p.nome) for p in procedimentos]
    resposta = "Oferecemos uma ampla gama de serviços! Nossos procedimentos incluem:\n\n"
    for categoria, nomes in categorias.items():
        resposta += f"*{categoria}*\n";_ = [resposta := resposta + f"- {nome}\n" for nome in nomes];resposta += "\n"
    return resposta
# [BLINDADO] Ferramenta de preços com lógica inequívoca
def consultar_precos_procedimentos(db: Session, telefone_paciente: str, termo_busca: str) -> str:
    termo_normalizado = re.sub(r'[-.,]', ' ', termo_busca.lower()); palavras_chave = termo_normalizado.split()
    filtros = [Procedimento.nome.ilike(f'%{palavra}%') for palavra in palavras_chave]
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    if not resultados: return f"Não encontrei informações de valores para '{termo_busca}'."
    respostas = []
    for r in resultados:
        if r.valor_base: respostas.append(f"O valor para {r.nome} é a partir de R$ {int(r.valor_base):,}.00".replace(",", "."))
        else: respostas.append(f"Para {r.nome}, o valor é {r.valor_descritivo}")
    return "\n".join(respostas)

available_functions = {"obter_proxima_etapa": obter_proxima_etapa, "atualizar_dados_paciente": atualizar_dados_paciente, "agendar_consulta": agendar_consulta, "consultar_meus_agendamentos": consultar_meus_agendamentos, "consultar_e_cancelar_inteligente": consultar_e_cancelar_inteligente, "consultar_horarios_disponiveis": consultar_horarios_disponiveis, "listar_todos_os_procedimentos": listar_todos_os_procedimentos, "consultar_precos_procedimentos": consultar_precos_procedimentos}
tools = [{"type": "function", "function": {"name": "obter_proxima_etapa", "description": "Verifica o status do cadastro do paciente para determinar a próxima ação.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "atualizar_dados_paciente", "description": "Atualiza os dados de um paciente.", "parameters": {"type": "object", "properties": {"nome_completo": {"type": "string"}, "endereco": {"type": "string"}, "email": {"type": "string"}, "data_nascimento": {"type": "string", "description": "Formato AAAA-MM-DD"}}}}},
         {"type": "function", "function": {"name": "agendar_consulta", "description": "Ação FINAL para agendar uma consulta.", "parameters": {"type": "object", "properties": {"data_hora_agendamento": {"type": "string"}, "procedimento": {"type": "string"}}, "required": ["data_hora_agendamento", "procedimento"]}}},
         {"type": "function", "function": {"name": "consultar_horarios_disponiveis", "description": "Verifica os horários livres em um dia específico.", "parameters": {"type": "object", "properties": {"dia": {"type": "string", "description": "O dia a ser verificado no formato AAAA-MM-DD."}}, "required": ["dia"]}}},
         {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Lista agendamentos futuros confirmados.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "consultar_e_cancelar_inteligente", "description": "Cancela uma consulta. Pode receber o nome do procedimento como dica.", "parameters": {"type": "object", "properties": {"dica": {"type": "string", "description": "O procedimento que o usuário quer cancelar, se ele especificar."}}}}},
         {"type": "function", "function": {"name": "listar_todos_os_procedimentos", "description": "Lista todos os serviços e procedimentos oferecidos pela clínica.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "consultar_precos_procedimentos", "description": "Consulta preços de procedimentos.", "parameters": {"type": "object", "properties": {"termo_busca": {"type": "string", "description": "O procedimento para saber o preço."}}, "required": ["termo_busca"]}}}]

# ───────────────── 5. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp para DI DONATO ODONTO.", version="9.0.0-demo")
@app.on_event("startup")
async def startup_event(): await asyncio.to_thread(criar_tabelas); print("Tabelas verificadas/criadas.", flush=True);_ = SessionLocal(); with _ as db: popular_procedimentos_iniciais(db)
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
        await enviar_resposta_whatsapp(telefone, "Olá! Sou o assistente virtual da DI DONATO ODONTO. Como posso te ajudar hoje?")
        return {"status": "ignorado", "motivo": "sem conteúdo processável"}

    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)); db.commit()
    historico_recente = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= datetime.utcnow() - timedelta(hours=24), HistoricoConversa.role != 'system').order_by(HistoricoConversa.timestamp).all()
    
    NOME_CLINICA, PROFISSIONAL = "DI DONATO ODONTO", "Dra. Valéria Cristina Di Donato"
    system_prompt = (
        f"Você é Sofia, assistente virtual da {NOME_CLINICA}, onde os atendimentos são realizados pela {PROFISSIONAL}. "
        f"Seja sempre educado, prestativo e converse de forma natural. Hoje é {datetime.now().strftime('%d/%m/%Y')}. "
        "Determine a intenção do usuário. Se for sobre AGENDAR, CANCELAR ou REAGENDAR, use a ferramenta `obter_proxima_etapa` para iniciar o fluxo de onboarding. "
        "Siga as instruções da ferramenta `obter_proxima_etapa` para coletar dados faltantes, UM DE CADA VEZ. "
        "Quando o cadastro estiver completo, use as outras ferramentas para atender o pedido do usuário (agendar, consultar preços, etc.). "
        "Antes de agendar, sempre confirme com um resumo: 'Posso confirmar seu agendamento de [Procedimento] para [Data] às [Hora] com a {PROFISSIONAL}?'. Só agende após a confirmação."
    )
    
    mensagens_para_ia: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in historico_recente: mensagens_para_ia.append({"role": msg.role, "content": msg.content})
    try:
        modelo_chat = "google/gemini-2.5-flash-preview-05-20"
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
