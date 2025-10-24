# Q-TAL - Quantitative Trend and Level Analyzer

Sistema di analisi quantitativa automatizzata per serie storiche finanziarie.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Caratteristiche

- **Analisi Trend**: Classificazione automatica tramite Slope EMA125
- **Analisi Momentum**: Z-Score 20 periodi per identificare overbought/oversold
- **Zone S/R**: Identificazione automatica di zone Supply/Demand
- **Setup Operativi**: Generazione segnali con risk/reward ratio
- **Grafici Interattivi**: Visualizzazione completa con Plotly
- **Parametri Configurabili**: Personalizzazione completa dell'analisi

## 📋 Requisiti

- Python 3.9+
- Account EODHD con API Key attiva
- Connessione internet per il download dei dati

## 🚀 Installazione Locale

### 1. Clona il repository

```bash
git clone https://github.com/your-username/qtal-streamlit.git
cd qtal-streamlit
```

### 2. Crea ambiente virtuale (opzionale ma consigliato)

```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

### 3. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura API Key

Crea il file `.streamlit/secrets.toml`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Modifica `.streamlit/secrets.toml` e inserisci la tua API Key:

```toml
EODHD_API_KEY = "your_actual_api_key_here"
```

**⚠️ IMPORTANTE**: NON committare il file `secrets.toml` su GitHub!

### 5. Avvia l'applicazione

```bash
streamlit run app.py
```

L'app sarà disponibile su `http://localhost:8501`

## ☁️ Deploy su Streamlit Cloud

### 1. Prepara il repository

Assicurati che il repository contenga:
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`

**NON includere** `.streamlit/secrets.toml` nel repository!

### 2. Deploy su Streamlit Cloud

1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Connetti il tuo account GitHub
3. Seleziona il repository `qtal-streamlit`
4. Specifica il file principale: `app.py`
5. Clicca su "Deploy"

### 3. Configura i Secrets

Nella dashboard di Streamlit Cloud:

1. Vai in **Settings** > **Secrets**
2. Inserisci:

```toml
EODHD_API_KEY = "your_actual_api_key_here"
```

3. Salva e riavvia l'app

## 📊 Utilizzo

### Parametri Configurabili

Nella sidebar puoi personalizzare:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| **Shadow Multiplier** | 1.5 | Moltiplicatore per identificare shadow significative |
| **Min Hits** | 4 | Numero minimo di tocchi per validare una zona |
| **Lookback Months** | 6 | Mesi di storia per identificare le zone |
| **Max Zones** | 10 | Numero massimo di zone da visualizzare |
| **Max Zone Width (%)** | 1.5 | Larghezza massima zona in % del prezzo |

### Workflow Tipico

1. **Inserisci Ticker**: Formato `TICKER.EXCHANGE` (es. `AAPL.US`, `SPY.US`)
2. **Personalizza Parametri**: (opzionale) Modifica i parametri nella sidebar
3. **Avvia Analisi**: Clicca su "🚀 Avvia Analisi"
4. **Analizza Risultati**: Visualizza dashboard, zone S/R e grafico interattivo

### Formato Ticker Supportati

- **US Stocks**: `AAPL.US`, `TSLA.US`, `MSFT.US`
- **ETF**: `SPY.US`, `QQQ.US`, `IWM.US`
- **Crypto**: `BTC-USD.CC`, `ETH-USD.CC`
- **Forex**: `EURUSD.FOREX`, `GBPUSD.FOREX`

Verifica i ticker disponibili su [EODHD](https://eodhistoricaldata.com/financial-apis/list-supported-exchanges/)

## 🔧 Struttura del Progetto

```
qtal-streamlit/
├── app.py                          # Applicazione principale Streamlit
├── requirements.txt                # Dipendenze Python
├── README.md                       # Documentazione
├── .streamlit/
│   ├── config.toml                # Configurazione UI Streamlit
│   └── secrets.toml.example       # Template per secrets
└── .gitignore                     # File da escludere dal repository
```

## 🧪 Test dell'Applicazione

Per testare l'app localmente:

```bash
# Test con ticker di esempio
streamlit run app.py
```

Nell'interfaccia:
1. Inserisci l'API Key (se non configurata nei secrets)
2. Prova con `SPY.US`
3. Lascia i parametri di default
4. Clicca "Avvia Analisi"

## 📈 Metodologia

### 1. Analisi Trend
- **Indicatore**: Slope EMA125 (regressione lineare su 20 periodi)
- **Classificazione**: 
  - UPTREND: slope > 0.05
  - DOWNTREND: slope < -0.05
  - LATERAL: altrimenti

### 2. Analisi Momentum
- **Indicatore**: Z-Score a 20 periodi
- **Classificazione**:
  - OVERBOUGHT: z-score > 1.5
  - OVERSOLD: z-score < -1.5
  - NEUTRAL: altrimenti

### 3. Zone Supporto/Resistenza
- **Metodo**: Clustering di shadow delle candele
- **Validazione**: Numero minimo di tocchi configurabile
- **Filtraggio**: Larghezza massima in % del prezzo

### 4. Setup Operativi
- **Long su Demand**: Trend UP + prezzo vicino a zona Demand
- **Short su Supply**: Trend DOWN + prezzo vicino a zona Supply
- **Confidenza**: Combinazione di trend, momentum e posizione

## ⚙️ Configurazione Avanzata

### Personalizzazione Tema

Modifica `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1e88e5"      # Colore primario
backgroundColor = "#ffffff"    # Sfondo principale
secondaryBackgroundColor = "#f5f5f5"  # Sfondo secondario
textColor = "#262730"         # Colore testo
```

### Rate Limiting API

Il client EODHD implementa:
- **Min interval**: 100ms tra richieste consecutive
- **Retry logic**: 3 tentativi con exponential backoff
- **Timeout**: 30 secondi per richiesta

## 🐛 Troubleshooting

### Errore: "API Key non configurata"

**Soluzione**: Verifica che l'API Key sia:
- Inserita correttamente nei secrets di Streamlit Cloud, oppure
- Inserita nella sidebar dell'app, oppure
- Configurata nel file `.streamlit/secrets.toml` (per uso locale)

### Errore: "Ticker non trovato"

**Soluzione**: 
- Verifica il formato: `TICKER.EXCHANGE`
- Controlla che il ticker sia supportato da EODHD
- Esempio corretto: `AAPL.US` (non `AAPL`)

### Errore: "Timeout"

**Soluzione**:
- Verifica la connessione internet
- Riduci il periodo di analisi (il default è 2 anni)
- Prova con un ticker più liquido (es. SPY.US)

### Nessuna zona S/R identificata

**Soluzione**:
- Riduci `Min Hits` (es. da 4 a 3)
- Aumenta `Lookback Months` (es. da 6 a 9)
- Aumenta `Max Zone Width %` (es. da 1.5 a 2.0)
- Alcuni ticker potrebbero non avere zone evidenti nel periodo analizzato

## 📝 Note per lo Sviluppo

### Aggiungere nuove feature

1. Modifica `app.py` con le nuove funzionalità
2. Aggiorna `requirements.txt` se servono nuove librerie
3. Testa localmente con `streamlit run app.py`
4. Committa e pusha su GitHub
5. Streamlit Cloud effettuerà il re-deploy automatico

### Debug

Abilita la modalità debug in `.streamlit/config.toml`:

```toml
[server]
enableCORS = false
enableXsrfProtection = false

[logger]
level = "debug"
```

## 📄 Licenza

Proprietà di **Kriterion Quant**. Tutti i diritti riservati.

## 🤝 Supporto

Per domande o supporto:
- **Email**: info@kriterionquant.com
- **Sito**: [kriterionquant.com](https://kriterionquant.com)
- **Issues**: Usa la sezione Issues di GitHub

## ⚠️ Disclaimer

Questo software è fornito **"as-is"** a scopo **educativo e informativo**.

**NON costituisce consulenza finanziaria**. L'utilizzo di questo sistema per operazioni reali sui mercati è a proprio rischio e pericolo. L'autore non si assume alcuna responsabilità per perdite derivanti dall'uso di questo software.

Prima di operare sui mercati:
- Consulta un consulente finanziario abilitato
- Comprendi i rischi del trading
- Opera solo con capitale che puoi permetterti di perdere
- Testa sempre le strategie in demo prima di usarle in reale

---

**© 2024 Kriterion Quant** - Sistema proprietario per trading quantitativo
