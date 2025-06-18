"""
API principal do OdontoBot AI. Versão Final para Produção.
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
def get_today_br() -> DateObject: return get_now().date()
def get_tomorrow_br_str() -> str: return (get_now() + timedelta(days=1)).strftime("%Y-%m-%d")


# ───────────────── 2. CONFIGURAÇÃO DOS CLIENTES DE IA ───────── #
try:
    import openai
    from dateparser import parse as parse_date
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

# ───────────────── 4. FERRAMENTAS ──────────────────────────── #
def buscar_ou_criar_paciente(db: Session, tel: str) -> Paciente:
    paciente = db.query(Paciente).filter_by(telefone=tel).first()
    if not paciente: paciente = Paciente(telefone=tel); db.add(paciente); db.commit(); db.refresh(paciente)
    return paciente
def verificar_e_coletar_dados_paciente(db: Session, telefone_paciente: str, nome_completo: Optional[str] = None, email: Optional[str] = None, data_nascimento: Optional[str] = None, endereco: Optional[str] = None) -> str:
    paciente = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    dados_coletados_nesta_chamada = {}
    if nome_completo and not paciente.nome_completo: dados_coletados_nesta_chamada['nome_completo'] = nome_completo
    if email and not paciente.email: dados_coletados_nesta_chamada['email'] = email
    if endereco and not paciente.endereco: dados_coletados_nesta_chamada['endereco'] = endereco
    if data_nascimento and not paciente.data_nascimento:
        try: dados_coletados_nesta_chamada['data_nascimento'] = parse_date(data_nascimento, languages=['pt']).date()
        except (ValueError, AttributeError): return "Formato de data de nascimento inválido. Peça novamente no formato DD/MM/AAAA."
    for campo, valor in dados_coletados_nesta_chamada.items(): setattr(paciente, campo, valor)
    if dados_coletados_nesta_chamada: db.commit(); db.refresh(paciente)
    if not paciente.nome_completo: return "Ação: Paciente novo ou sem nome. Apresente-se cordialmente e peça o NOME COMPLETO."
    if not paciente.data_nascimento: return f"Ação: Agradeça pelo nome e peça a DATA DE NASCIMENTO (formato DD/MM/AAAA)."
    if not paciente.email: return "Ação: Agradeça e peça o E-MAIL."
    if not paciente.endereco: return "Ação: Agradeça e peça o ENDEREÇO COMPLETO (Rua, Número, Bairro, Cidade, CEP)."
    return f"Cadastro completo para {paciente.nome_completo}. Agora você pode prosseguir com a ação principal."
def agendar_consulta(db: Session, telefone_paciente: str, dia: str, hora: str, procedimento: str) -> str:
    try: data_obj = datetime.strptime(dia, "%Y-%m-%d").date(); hora_obj = datetime.strptime(hora, "%H:%M").time(); dt_agendamento_br = BR_TIMEZONE.localize(datetime.combine(data_obj, hora_obj))
    except ValueError: return "Formato de data ou hora inválido. Use dia como AAAA-MM-DD e hora como HH:MM."
    if dt_agendamento_br < get_now(): return "Não é possível agendar no passado."
    if dt_agendamento_br.weekday() >= 5: return "A clínica não funciona aos fins de semana (Sábado e Domingo)."
    if not (time(9) <= dt_agendamento_br.time() < time(18)): return "O horário de funcionamento é das 09:00 às 18:00."
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    if not all([pac.nome_completo, pac.data_nascimento, pac.email, pac.endereco]): return "O cadastro do paciente está incompleto. Use a ferramenta 'verificar_e_coletar_dados_paciente' antes de agendar."
    db.add(Agendamento(paciente_id=pac.id, data_hora=dt_agendamento_br, procedimento=procedimento)); db.commit()
    return f"Sucesso! Agendamento para '{procedimento}' criado para {dt_agendamento_br.strftime('%d/%m/%Y às %H:%M')}."
def consultar_meus_agendamentos(db: Session, telefone_paciente: str) -> str:
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    ags = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.status == "confirmado", Agendamento.data_hora >= get_now()).order_by(Agendamento.data_hora).all()
    if not ags: return "Você não possui agendamentos futuros confirmados."
    linhas = [f"- ID {a.id}: {a.procedimento} em {a.data_hora.astimezone(BR_TIMEZONE).strftime('%d/%m/%Y às %H:%M')}" for a in ags]
    return "Seus próximos agendamentos são:\n" + "\n".join(linhas)
def consultar_horarios_disponiveis(db: Session, telefone_paciente: str, dia: str) -> str:
    data_parseada = parse_date(dia, languages=['pt'], settings={'PREFER_DATES_FROM': 'future'})
    if not data_parseada: return "Formato de data inválido. Use um formato claro como 'amanhã' ou '25/12/2025'."
    data_consulta_obj = data_parseada.date()
    if data_consulta_obj < get_today_br(): return f"Não é possível verificar horários para um dia que já passou ({data_consulta_obj.strftime('%d/%m/%Y')})."
    agendamentos_do_dia = db.query(Agendamento.data_hora).filter(sql_func.date(Agendamento.data_hora) == data_consulta_obj, Agendamento.status == 'confirmado').all()
    horarios_ocupados = {ag.data_hora.astimezone(BR_TIMEZONE).time() for ag in agendamentos_do_dia}
    slots_possiveis = {time(h) for h in range(9, 18)}
    horarios_disponiveis = sorted(list(slots_possiveis - horarios_ocupados))
    if not horarios_disponiveis: return f"Não há mais horários disponíveis para o dia {data_consulta_obj.strftime('%d/%m/%Y')}."
    horarios_formatados = [t.strftime('%H:%M') for t in horarios_disponiveis]
    return f"Para o dia {data_consulta_obj.strftime('%d/%m/%Y')}, os horários livres são: {', '.join(horarios_formatados)}."
def reagendar_consulta_inteligente(db: Session, telefone_paciente: str, novo_dia: str, nova_hora: str, id_agendamento_original: Optional[int] = None) -> str:
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    query = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.status == "confirmado", Agendamento.data_hora >= get_now())
    if id_agendamento_original: query = query.filter(Agendamento.id == id_agendamento_original)
    ags = query.all()
    if not ags: return "Não encontrei um agendamento futuro para reagendar. Verifique se o ID está correto ou se há agendamentos."
    if len(ags) > 1 and not id_agendamento_original: return "Encontrei mais de um agendamento. Qual deles você gostaria de reagendar? Por favor, informe o ID.\n" + consultar_meus_agendamentos(db, telefone_paciente)
    ag_reagendar = ags[0]
    try: nova_dt = datetime.combine(datetime.strptime(novo_dia, "%Y-%m-%d").date(), datetime.strptime(nova_hora, "%H:%M").time(), tzinfo=BR_TIMEZONE)
    except ValueError: return "O formato da nova data ou hora é inválido. Use AAAA-MM-DD e HH:MM."
    ag_reagendar.data_hora = nova_dt; db.commit()
    return f"Sucesso! Seu agendamento (ID {ag_reagendar.id}) foi reagendado para {nova_dt.strftime('%d/%m/%Y às %H:%M')}."
def cancelar_agendamentos(db: Session, telefone_paciente: str, ids_para_cancelar: Optional[List[int]] = None, dica_procedimento: Optional[str] = None) -> str:
    pac = buscar_ou_criar_paciente(db, tel=telefone_paciente)
    query = db.query(Agendamento).filter(Agendamento.paciente_id == pac.id, Agendamento.status == "confirmado", Agendamento.data_hora >= get_now())
    if ids_para_cancelar: query = query.filter(Agendamento.id.in_(ids_para_cancelar))
    elif dica_procedimento: query = query.filter(Agendamento.procedimento.ilike(f'%{dica_procedimento}%'))
    ags_para_cancelar = query.all()
    if not ags_para_cancelar: return "Não encontrei nenhum agendamento ativo para cancelar com os critérios fornecidos."
    if len(ags_para_cancelar) > 1 and not ids_para_cancelar: return "Encontrei mais de um agendamento que corresponde à sua solicitação. Para evitar erros, por favor, me diga o ID exato do(s) agendamento(s) que deseja cancelar:\n" + consultar_meus_agendamentos(db, telefone_paciente)
    nomes_cancelados = [f"{ag.procedimento} de {ag.data_hora.astimezone(BR_TIMEZONE).strftime('%d/%m/%Y às %H:%M')} (ID {ag.id})" for ag in ags_para_cancelar]
    for ag in ags_para_cancelar: ag.status = "cancelado"
    db.commit()
    return f"Ok, cancelei com sucesso os seguintes agendamentos: {', '.join(nomes_cancelados)}."
def listar_todos_os_procedimentos(db: Session, telefone_paciente: str) -> str:
    procedimentos = db.query(Procedimento).order_by(Procedimento.categoria, Procedimento.nome).all()
    if not procedimentos: return "Não consegui encontrar a lista de procedimentos no momento."
    categorias = defaultdict(list);_ = [categorias[p.categoria].append(p.nome) for p in procedimentos]
    resposta = "Oferecemos uma ampla gama de serviços! Nossos procedimentos incluem:\n\n"
    for categoria, nomes in categorias.items():
        resposta += f"*{categoria}*\n";_ = [resposta := resposta + f"- {nome}\n" for nome in nomes];resposta += "\n"
    return resposta
def consultar_precos_procedimentos(db: Session, telefone_paciente: str, termo_busca: str) -> str:
    termo_normalizado = re.sub(r'[-.,]', ' ', termo_busca.lower()); palavras_chave = termo_normalizado.split()
    filtros = [Procedimento.nome.ilike(f'%{palavra}%') for palavra in palavras_chave]
    resultados = db.query(Procedimento).filter(and_(*filtros)).all()
    if not resultados: return f"Não encontrei informações de valores para '{termo_busca}'."
    respostas = [f"O valor para {r.nome} é a partir de R$ {int(r.valor_base):,}.00".replace(",", ".") if r.valor_base else f"Para {r.nome}, o valor é {r.valor_descritivo}" for r in resultados]
    return "\n".join(respostas)

available_functions = {"verificar_e_coletar_dados_paciente": verificar_e_coletar_dados_paciente, "agendar_consulta": agendar_consulta, "consultar_meus_agendamentos": consultar_meus_agendamentos, "cancelar_agendamentos": cancelar_agendamentos, "reagendar_consulta_inteligente": reagendar_consulta_inteligente, "consultar_horarios_disponiveis": consultar_horarios_disponiveis, "listar_todos_os_procedimentos": listar_todos_os_procedimentos, "consultar_precos_procedimentos": consultar_precos_procedimentos}
tools = [{"type": "function", "function": {"name": "verificar_e_coletar_dados_paciente", "description": "DEVE ser usada ANTES de qualquer ação de agendamento, reagendamento ou cancelamento para garantir que o cadastro do paciente esteja completo. Forneça os dados que já possui para ela atualizar e ela retornará o próximo passo.", "parameters": {"type": "object", "properties": {"nome_completo": {"type": "string", "description": "Nome completo do paciente, se fornecido."}, "email": {"type": "string", "description": "Email do paciente, se fornecido."}, "data_nascimento": {"type": "string", "description": "Data de nascimento no formato AAAA-MM-DD, se fornecida."}, "endereco": {"type": "string", "description": "Endereço completo do paciente, se fornecido."}}}}},
         {"type": "function", "function": {"name": "agendar_consulta", "description": "Ação FINAL para agendar uma consulta, usada apenas após o usuário confirmar um resumo. Requer dia e hora exatos.", "parameters": {"type": "object", "properties": {"dia": {"type": "string", "description": "A data no formato AAAA-MM-DD"}, "hora": {"type": "string", "description": "A hora no formato HH:MM"}, "procedimento": {"type": "string", "description": "O nome do procedimento a ser agendado."}}, "required": ["dia", "hora", "procedimento"]}}},
         {"type": "function", "function": {"name": "reagendar_consulta_inteligente", "description": "Reagenda uma consulta existente para um novo dia e hora.", "parameters": {"type": "object", "properties": {"novo_dia": {"type": "string", "description": "A nova data no formato AAAA-MM-DD"}, "nova_hora": {"type": "string", "description": "A nova hora no formato HH:MM"}, "id_agendamento_original": {"type": "integer", "description": "O ID do agendamento a ser alterado, se houver mais de um."}}}}},
         {"type": "function", "function": {"name": "cancelar_agendamentos", "description": "Cancela um ou mais agendamentos com base em uma lista de IDs.", "parameters": {"type": "object", "properties": {"ids_para_cancelar": {"type": "array", "items": {"type": "integer"}, "description": "Uma lista com os IDs numéricos dos agendamentos a cancelar."}, "dica_procedimento": {"type": "string", "description": "Nome ou parte do nome do procedimento a ser cancelado, se o ID não for fornecido."}}}}},
         {"type": "function", "function": {"name": "consultar_horarios_disponiveis", "description": "Verifica os horários livres em um dia específico.", "parameters": {"type": "object", "properties": {"dia": {"type": "string", "description": "O dia a ser verificado (ex: 'hoje', 'amanhã', '25/12/2025')."}}, "required": ["dia"]}}},
         {"type": "function", "function": {"name": "consultar_meus_agendamentos", "description": "Lista agendamentos futuros confirmados.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "listar_todos_os_procedimentos", "description": "Lista todos os serviços e procedimentos oferecidos pela clínica.", "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "consultar_precos_procedimentos", "description": "Consulta preços de procedimentos.", "parameters": {"type": "object", "properties": {"termo_busca": {"type": "string", "description": "O procedimento para saber o preço."}}, "required": ["termo_busca"]}}}]

# ───────────────── 5. APP FASTAPI ───────────────────────────── #
app = FastAPI(title="OdontoBot AI", description="Automação de WhatsApp para DI DONATO ODONTO.", version="13.0.1-syntax-fix")

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

    paciente = buscar_ou_criar_paciente(db, tel=telefone)
    db.add(HistoricoConversa(paciente_id=paciente.id, role="user", content=mensagem_usuario)); db.commit()
    historico_recente = db.query(HistoricoConversa).filter(HistoricoConversa.paciente_id == paciente.id, HistoricoConversa.timestamp >= get_now() - timedelta(hours=24), HistoricoConversa.role != 'system').order_by(HistoricoConversa.timestamp).all()
    
    NOME_CLINICA, PROFISSIONAL = "DI DONATO ODONTO", "Dra. Valéria Cristina Di Donato"
    system_prompt = (
        f"**Persona:** Você é a Sofia, assistente virtual da clínica {NOME_CLINICA}, onde a especialista responsável é a {PROFISSIONAL}. "
        f"Seja sempre educada, prestativa e converse de forma natural. Hoje é {get_now().strftime('%d/%m/%Y')} (formato AAAA-MM-DD: {get_now().strftime('%Y-%m-%d')}). Amanhã será {get_tomorrow_br_str()}.\n\n"
        "**Regras de Fluxo:**\n"
        "1. **Seja Reativa para Informações:** Se o usuário pedir preços, use `consultar_precos_procedimentos`. Se perguntar sobre quais serviços oferecemos, use `listar_todos_os_procedimentos`. Você PODE e DEVE fazer isso ANTES de qualquer cadastro.\n"
        "2. **Inicie o Cadastro na Hora Certa:** SOMENTE quando o usuário expressar uma intenção clara de AGENDAR, REAGENDAR ou CANCELAR, você DEVE usar a ferramenta `verificar_e_coletar_dados_paciente`. Informe que precisa dos dados para prosseguir com a ação (agendar, etc.).\n"
        "3. **Coleta de Dados Inteligente:** A ferramenta `verificar_e_coletar_dados_paciente` te dirá qual o próximo dado faltante. Peça esse dado ao usuário. Se o usuário fornecer um dado diferente, use a ferramenta novamente passando o dado que ele forneceu (ela vai salvar o que for útil e te dizer o próximo passo).\n"
        "4. **Agendamento Detalhado:** Quando o cadastro estiver completo, pergunte o procedimento desejado e o DIA para agendamento. Use `consultar_horarios_disponiveis` para aquele dia. Após o usuário escolher um horário livre, apresente um resumo: 'Posso confirmar [Procedimento] para [Dia] às [Hora] com a {PROFISSIONAL}?' SÓ APÓS o 'sim' do usuário, use a ferramenta `agendar_consulta`.\n"
        "5. **Reagendamento Inteligente:** Se o usuário quiser reagendar: (a) Primeiro, use `consultar_meus_agendamentos` para listar os agendamentos existentes. (b) Se houver um único agendamento, pergunte o novo dia e hora e use `reagendar_consulta_inteligente` com o ID daquele agendamento. (c) Se houver vários, mostre a lista e peça o ID do agendamento a ser alterado antes de usar a ferramenta.\n"
        "6. **Cancelamento Flexível:** Se o usuário quiser cancelar: (a) Use `consultar_meus_agendamentos`. (b) Se houver apenas UM agendamento, confirme com o usuário ('Gostaria de cancelar seu agendamento de [Procedimento] para [Data]?') e, após o 'sim', use `cancelar_agendamentos` fornecendo uma lista com o ID daquele agendamento. (c) Se houver VÁRIOS, mostre a lista e peça o ID ou os IDs dos agendamentos a serem cancelados para usar na ferramenta."
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
