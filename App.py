"""
Q-TAL - Quantitative Trend and Level Analyzer
Versione Streamlit: 1.0
Proprietario: Kriterion Quant
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import linregress
import requests
import time

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="Q-TAL - Quantitative Trend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    .zone-demand {
        color: #26a69a;
        font-weight: 700;
    }
    .zone-supply {
        color: #ef5350;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Q-TAL - Quantitative Trend and Level Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema di analisi quantitativa automatizzata - Kriterion Quant</div>', unsafe_allow_html=True)


# ============================================================================
# CLASSI DEL SISTEMA (IDENTICHE AL NOTEBOOK)
# ============================================================================

class EODHDClient:
    """Client per interagire con l'API EODHD."""
    
    BASE_URL = "https://eodhistoricaldata.com/api"
    
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key EODHD non configurata")
        
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 0.1
    
    def _wait_for_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint, params=None, max_retries=3):
        if params is None:
            params = {}
        
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    raise Exception(f"Ticker non trovato: {url}")
                else:
                    response.raise_for_status()
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception("Timeout: impossibile connettersi all'API EODHD")
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"Errore nella richiesta API: {e}")
        
        raise Exception(f"Impossibile completare la richiesta dopo {max_retries} tentativi")
    
    def get_historical_data(self, ticker, start_date, end_date):
        endpoint = f"eod/{ticker}"
        params = {
            'from': start_date,
            'to': end_date,
            'order': 'd'
        }
        
        data = self._make_request(endpoint, params)
        
        if not data:
            raise Exception(f"Nessun dato disponibile per {ticker}")
        
        df = pd.DataFrame(data)
        
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise Exception(f"Dati incompleti per {ticker}")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        
        return df


class FeatureEngine:
    """Calcolo di indicatori tecnici e feature quantitative."""
    
    @staticmethod
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_zscore(series, window=20):
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        zscore = (series - mean) / std
        return zscore
    
    @staticmethod
    def calculate_slope(series, window=20):
        slopes = []
        x = np.arange(window)
        
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window and not np.isnan(y).any():
                    slope, _, _, _, _ = linregress(x, y)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        
        return pd.Series(slopes, index=series.index)
    
    @staticmethod
    def add_all_features(df):
        df = df.copy()
        
        df['ema_21'] = FeatureEngine.calculate_ema(df['close'], 21)
        df['ema_50'] = FeatureEngine.calculate_ema(df['close'], 50)
        df['ema_125'] = FeatureEngine.calculate_ema(df['close'], 125)
        df['ema_200'] = FeatureEngine.calculate_ema(df['close'], 200)
        
        df['atr_14'] = FeatureEngine.calculate_atr(df, 14)
        
        df['zscore_20'] = FeatureEngine.calculate_zscore(df['close'], 20)
        
        df['slope_ema125'] = FeatureEngine.calculate_slope(df['ema_125'], 20)
        
        df = df.dropna()
        
        return df


class AnalysisEngine:
    """Engine per l'analisi del trend e identificazione zone S/R."""
    
    @staticmethod
    def classify_trend(slope_value, threshold_up=0.05, threshold_down=-0.05):
        if slope_value > threshold_up:
            return "UPTREND"
        elif slope_value < threshold_down:
            return "DOWNTREND"
        else:
            return "LATERAL"
    
    @staticmethod
    def classify_momentum(zscore_value):
        if zscore_value > 1.5:
            return "OVERBOUGHT"
        elif zscore_value < -1.5:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def identify_support_resistance_zones(df, shadow_multiplier=1.5, min_hits=4, 
                                         lookback_months=6, max_zones=10, 
                                         max_zone_width_pct=1.5):
        """
        Identifica zone di supporto e resistenza basate su shadow delle candele.
        
        Parametri configurabili:
        - shadow_multiplier: moltiplicatore per le shadow
        - min_hits: numero minimo di tocchi per validare una zona
        - lookback_months: mesi di lookback per l'analisi
        - max_zones: numero massimo di zone da identificare
        - max_zone_width_pct: larghezza massima della zona in percentuale
        """
        
        lookback_days = lookback_months * 21
        df_recent = df.tail(lookback_days).copy()
        
        if len(df_recent) < 50:
            return [], []
        
        current_price = df_recent['close'].iloc[-1]
        atr = df_recent['atr_14'].iloc[-1]
        
        min_zone_width = atr * 0.5
        max_zone_width = current_price * (max_zone_width_pct / 100)
        
        df_recent['body'] = abs(df_recent['close'] - df_recent['open'])
        df_recent['upper_shadow'] = df_recent['high'] - df_recent[['close', 'open']].max(axis=1)
        df_recent['lower_shadow'] = df_recent[['close', 'open']].min(axis=1) - df_recent['low']
        
        avg_body = df_recent['body'].median()
        
        supply_candles = df_recent[
            (df_recent['upper_shadow'] > df_recent['body'] * shadow_multiplier) &
            (df_recent['upper_shadow'] > avg_body * 0.5)
        ].copy()
        
        demand_candles = df_recent[
            (df_recent['lower_shadow'] > df_recent['body'] * shadow_multiplier) &
            (df_recent['lower_shadow'] > avg_body * 0.5)
        ].copy()
        
        def cluster_zones(candles_df, zone_type):
            if len(candles_df) == 0:
                return []
            
            if zone_type == 'supply':
                levels = candles_df['high'].values
            else:
                levels = candles_df['low'].values
            
            levels_sorted = np.sort(levels)
            
            zones = []
            i = 0
            
            while i < len(levels_sorted):
                cluster_start = levels_sorted[i]
                cluster_levels = [cluster_start]
                
                j = i + 1
                while j < len(levels_sorted):
                    if abs(levels_sorted[j] - cluster_start) <= max_zone_width:
                        cluster_levels.append(levels_sorted[j])
                        j += 1
                    else:
                        break
                
                if len(cluster_levels) >= min_hits:
                    zone_center = np.mean(cluster_levels)
                    zone_std = np.std(cluster_levels) if len(cluster_levels) > 1 else min_zone_width
                    zone_width = max(min_zone_width, min(zone_std * 2, max_zone_width))
                    
                    zone_low = zone_center - zone_width / 2
                    zone_high = zone_center + zone_width / 2
                    
                    strength = len(cluster_levels)
                    
                    touches_in_zone = sum(
                        1 for level in levels 
                        if zone_low <= level <= zone_high
                    )
                    
                    zones.append({
                        'center': zone_center,
                        'low': zone_low,
                        'high': zone_high,
                        'strength': strength,
                        'touches': touches_in_zone,
                        'type': zone_type
                    })
                
                i = j if j > i else i + 1
            
            zones_sorted = sorted(zones, key=lambda x: x['strength'], reverse=True)
            return zones_sorted[:max_zones]
        
        supply_zones = cluster_zones(supply_candles, 'supply')
        demand_zones = cluster_zones(demand_candles, 'demand')
        
        return supply_zones, demand_zones


class SignalGenerator:
    """Generazione di setup operativi e segnali di trading."""
    
    @staticmethod
    def generate_setup(df, supply_zones, demand_zones):
        if df is None or len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        current_price = latest['close']
        atr = latest['atr_14']
        trend = AnalysisEngine.classify_trend(latest['slope_ema125'])
        momentum = AnalysisEngine.classify_momentum(latest['zscore_20'])
        
        nearest_supply = min(
            [z for z in supply_zones if z['center'] > current_price],
            key=lambda x: x['center'] - current_price,
            default=None
        )
        
        nearest_demand = max(
            [z for z in demand_zones if z['center'] < current_price],
            key=lambda x: current_price - x['center'],
            default=None
        )
        
        setup_type = "NESSUN SETUP"
        entry_suggestion = "Attendere conferme"
        confidence = "BASSA"
        stop_loss = None
        take_profit = None
        risk_reward = "N/A"
        
        distance_to_demand = (current_price - nearest_demand['center']) / current_price * 100 if nearest_demand else 999
        distance_to_supply = (nearest_supply['center'] - current_price) / current_price * 100 if nearest_supply else 999
        
        if trend == "UPTREND" and nearest_demand and distance_to_demand < 2:
            setup_type = "LONG su Demand"
            entry_suggestion = f"Entry: ${nearest_demand['low']:.2f} - ${nearest_demand['high']:.2f}"
            stop_loss = nearest_demand['low'] - atr * 1.5
            take_profit = nearest_supply['center'] if nearest_supply else current_price + (current_price - stop_loss) * 2
            
            if momentum == "OVERSOLD":
                confidence = "ALTA"
            else:
                confidence = "MEDIA"
        
        elif trend == "DOWNTREND" and nearest_supply and distance_to_supply < 2:
            setup_type = "SHORT su Supply"
            entry_suggestion = f"Entry: ${nearest_supply['low']:.2f} - ${nearest_supply['high']:.2f}"
            stop_loss = nearest_supply['high'] + atr * 1.5
            take_profit = nearest_demand['center'] if nearest_demand else current_price - (stop_loss - current_price) * 2
            
            if momentum == "OVERBOUGHT":
                confidence = "ALTA"
            else:
                confidence = "MEDIA"
        
        if stop_loss and take_profit:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk > 0:
                risk_reward = f"{reward / risk:.2f}"
        
        setup = {
            'type': setup_type,
            'entry_suggestion': entry_suggestion,
            'confidence': confidence,
            'stop_loss': f"${stop_loss:.2f}" if stop_loss else "N/A",
            'take_profit': f"${take_profit:.2f}" if take_profit else "N/A",
            'risk_reward': risk_reward,
            'nearest_supply': nearest_supply,
            'nearest_demand': nearest_demand
        }
        
        return setup


class Visualizer:
    """Generazione di grafici interattivi con Plotly."""
    
    @staticmethod
    def create_interactive_chart(df, supply_zones, demand_zones, ticker):
        """Crea grafico interattivo principale con candele e zone S/R."""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(
                f'{ticker} - Analisi Quantitativa',
                'Z-Score (Momentum)',
                'Slope EMA125 (Trend)'
            )
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # EMA
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_21'],
                mode='lines',
                name='EMA 21',
                line=dict(color='#ffa726', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_50'],
                mode='lines',
                name='EMA 50',
                line=dict(color='#42a5f5', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_125'],
                mode='lines',
                name='EMA 125',
                line=dict(color='#ab47bc', width=2)
            ),
            row=1, col=1
        )
        
        # Zone Supply
        for zone in supply_zones:
            fig.add_hrect(
                y0=zone['low'],
                y1=zone['high'],
                fillcolor='rgba(239, 83, 80, 0.2)',
                line_width=0,
                row=1, col=1
            )
            
            mid_x = df.index[len(df.index) // 2]
            fig.add_annotation(
                x=mid_x,
                y=zone['center'],
                text=f"Supply ({zone['touches']})",
                showarrow=False,
                font=dict(size=10, color='#c62828'),
                row=1, col=1
            )
        
        # Zone Demand
        for zone in demand_zones:
            fig.add_hrect(
                y0=zone['low'],
                y1=zone['high'],
                fillcolor='rgba(38, 166, 154, 0.2)',
                line_width=0,
                row=1, col=1
            )
            
            mid_x = df.index[len(df.index) // 2]
            fig.add_annotation(
                x=mid_x,
                y=zone['center'],
                text=f"Demand ({zone['touches']})",
                showarrow=False,
                font=dict(size=10, color='#00695c'),
                row=1, col=1
            )
        
        # Z-Score
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['zscore_20'],
                mode='lines',
                name='Z-Score',
                line=dict(color='#7e57c2', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-1.5, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=0, line_color="gray", row=2, col=1)
        
        # Slope EMA125
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['slope_ema125'],
                mode='lines',
                name='Slope EMA125',
                line=dict(color='#e91e63', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.add_hline(y=0, line_color="gray", row=3, col=1)
        
        # Layout
        fig.update_layout(
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Data", row=3, col=1)
        fig.update_yaxes(title_text="Prezzo ($)", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig.update_yaxes(title_text="Slope", row=3, col=1)
        
        return fig


# ============================================================================
# INTERFACCIA STREAMLIT
# ============================================================================

# Sidebar con parametri
st.sidebar.header("⚙️ Configurazione")

# Caricamento API Key dai secrets
try:
    api_key = st.secrets["EODHD_API_KEY"]
except:
    st.sidebar.error("❌ API Key EODHD non configurata nei secrets")
    st.sidebar.info("Configura EODHD_API_KEY in Settings > Secrets")
    api_key = None

# Input Ticker
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY.US",
    help="Formato: TICKER.EXCHANGE (es. AAPL.US, TSLA.US)"
).upper()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Parametri Zone S/R")

shadow_multiplier = st.sidebar.slider(
    "Shadow Multiplier",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Moltiplicatore per identificare shadow significative"
)

min_hits = st.sidebar.slider(
    "Min Hits",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="Numero minimo di tocchi per validare una zona"
)

lookback_months = st.sidebar.slider(
    "Lookback Months",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
    help="Mesi di storia per identificare le zone"
)

max_zones = st.sidebar.slider(
    "Max Zones",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Numero massimo di zone da mostrare"
)

max_zone_width_pct = st.sidebar.slider(
    "Max Zone Width (%)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Larghezza massima zona in % del prezzo"
)

# Bottone Analizza
analyze_button = st.sidebar.button("🚀 Avvia Analisi", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Q-TAL v1.0**  
Sistema di analisi quantitativa  
© Kriterion Quant
""")


# ============================================================================
# LOGICA PRINCIPALE
# ============================================================================

if analyze_button:
    if not api_key:
        st.error("❌ Inserisci l'API Key EODHD nella sidebar")
    elif not ticker:
        st.error("❌ Inserisci un ticker valido")
    else:
        try:
            with st.spinner(f"🔄 Scaricamento dati per {ticker}..."):
                # Inizializza client
                client = EODHDClient(api_key)
                
                # Calcola date
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2 anni
                
                # Scarica dati
                df = client.get_historical_data(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                st.success(f"✅ Scaricati {len(df)} giorni di dati")
            
            with st.spinner("🔄 Calcolo indicatori tecnici..."):
                # Aggiungi feature
                df = FeatureEngine.add_all_features(df)
                st.success(f"✅ Calcolati indicatori (EMA, ATR, Z-Score, Slope)")
            
            with st.spinner("🔄 Identificazione zone S/R..."):
                # Identifica zone
                supply_zones, demand_zones = AnalysisEngine.identify_support_resistance_zones(
                    df,
                    shadow_multiplier=shadow_multiplier,
                    min_hits=min_hits,
                    lookback_months=lookback_months,
                    max_zones=max_zones,
                    max_zone_width_pct=max_zone_width_pct
                )
                
                st.success(f"✅ Identificate {len(supply_zones)} zone Supply e {len(demand_zones)} zone Demand")
            
            with st.spinner("🔄 Generazione setup operativo..."):
                # Genera setup
                setup = SignalGenerator.generate_setup(df, supply_zones, demand_zones)
            
            with st.spinner("🔄 Creazione grafico interattivo..."):
                # Crea grafico
                fig = Visualizer.create_interactive_chart(df, supply_zones, demand_zones, ticker)
            
            # ============================================================================
            # VISUALIZZAZIONE RISULTATI
            # ============================================================================
            
            st.markdown("---")
            st.markdown("## 📊 Dashboard Riassuntiva")
            
            # Metriche principali
            latest = df.iloc[-1]
            current_price = latest['close']
            atr = latest['atr_14']
            zscore = latest['zscore_20']
            slope = latest['slope_ema125']
            trend = AnalysisEngine.classify_trend(slope)
            momentum = AnalysisEngine.classify_momentum(zscore)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Prezzo Attuale", f"${current_price:,.2f}")
            
            with col2:
                trend_emoji = "📈" if trend == "UPTREND" else "📉" if trend == "DOWNTREND" else "↔️"
                st.metric(f"{trend_emoji} Trend", trend)
            
            with col3:
                momentum_color = "🔴" if momentum == "OVERBOUGHT" else "🟢" if momentum == "OVERSOLD" else "🟡"
                st.metric(f"{momentum_color} Momentum", momentum)
            
            with col4:
                st.metric("📊 Volatilità ATR", f"${atr:.2f}")
            
            # Dettagli tecnici
            st.markdown("### 📈 Analisi Tecnica Dettagliata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Indicatori Trend**")
                st.write(f"- **Slope EMA125**: {slope:.4f}")
                st.write(f"- **Classificazione**: {trend}")
                st.write(f"- **EMA 21**: ${latest['ema_21']:.2f}")
                st.write(f"- **EMA 50**: ${latest['ema_50']:.2f}")
                st.write(f"- **EMA 125**: ${latest['ema_125']:.2f}")
            
            with col2:
                st.markdown("**Indicatori Momentum**")
                st.write(f"- **Z-Score 20**: {zscore:.2f}")
                st.write(f"- **Stato**: {momentum}")
                st.write(f"- **ATR 14**: ${atr:.2f} ({atr/current_price*100:.2f}%)")
            
            # Zone S/R
            st.markdown("### 🎯 Zone Supporto/Resistenza")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Zone DEMAND (Supporto)**")
                if demand_zones:
                    for i, zone in enumerate(demand_zones[:5], 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Zona #{i}</strong><br>
                            Range: ${zone['low']:.2f} - ${zone['high']:.2f}<br>
                            Centro: <span class="zone-demand">${zone['center']:.2f}</span><br>
                            Tocchi: {zone['touches']} | Forza: {zone['strength']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("")
                else:
                    st.info("Nessuna zona Demand identificata")
            
            with col2:
                st.markdown("**🔴 Zone SUPPLY (Resistenza)**")
                if supply_zones:
                    for i, zone in enumerate(supply_zones[:5], 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Zona #{i}</strong><br>
                            Range: ${zone['low']:.2f} - ${zone['high']:.2f}<br>
                            Centro: <span class="zone-supply">${zone['center']:.2f}</span><br>
                            Tocchi: {zone['touches']} | Forza: {zone['strength']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("")
                else:
                    st.info("Nessuna zona Supply identificata")
            
            # Setup Operativo
            st.markdown("### 💡 Setup Operativo")
            
            if setup:
                # Colori condizionali
                if setup['confidence'] == "ALTA":
                    conf_color = "🟢"
                elif setup['confidence'] == "MEDIA":
                    conf_color = "🟡"
                else:
                    conf_color = "⚪"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📋 Tipo Setup", setup['type'])
                    st.metric(f"{conf_color} Confidenza", setup['confidence'])
                
                with col2:
                    st.write("**Entry Suggestion**")
                    st.info(setup['entry_suggestion'])
                    st.metric("🛑 Stop Loss", setup['stop_loss'])
                
                with col3:
                    st.metric("🎯 Take Profit", setup['take_profit'])
                    st.metric("⚖️ Risk/Reward", setup['risk_reward'])
            else:
                st.warning("⚠️ Nessun setup operativo identificato")
            
            # Grafico interattivo
            st.markdown("---")
            st.markdown("## 📈 Grafico Interattivo")
            st.plotly_chart(fig, use_container_width=True)
            
            # Disclaimer
            st.markdown("---")
            st.info("""
            **⚠️ Disclaimer**: Questo report è fornito a scopo informativo e educativo.  
            Non costituisce consulenza finanziaria. Fai sempre le tue ricerche prima di operare sui mercati.
            """)
            
        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {str(e)}")
            import traceback
            with st.expander("Dettagli errore"):
                st.code(traceback.format_exc())

else:
    # Schermata iniziale
    st.info("""
    👋 **Benvenuto in Q-TAL**
    
    Sistema di analisi quantitativa automatizzata per identificare:
    - **Trend** di medio periodo (EMA125 Slope)
    - **Momentum** (Z-Score)
    - **Zone di Supporto/Resistenza** (Supply/Demand)
    - **Setup operativi** con risk/reward
    
    **📝 Istruzioni:**
    1. Inserisci l'**API Key EODHD** nella sidebar (o configurala nei secrets)
    2. Specifica il **ticker** (formato: TICKER.EXCHANGE)
    3. Personalizza i **parametri** delle zone S/R (opzionale)
    4. Clicca su **"🚀 Avvia Analisi"**
    
    **🔧 Parametri Configurabili:**
    - **Shadow Multiplier**: sensibilità nell'identificazione delle shadow
    - **Min Hits**: numero minimo di tocchi per validare una zona
    - **Lookback Months**: profondità storica dell'analisi
    - **Max Zones**: numero massimo di zone da visualizzare
    - **Max Zone Width**: larghezza massima delle zone in %
    
    ---
    
    **📚 Metodologia:**
    
    Il sistema Q-TAL utilizza un approccio quantitativo multi-livello:
    
    1. **Analisi Trend** → Slope EMA125 per classificare UPTREND/DOWNTREND/LATERAL
    2. **Analisi Momentum** → Z-Score 20 periodi per identificare OVERBOUGHT/OVERSOLD
    3. **Zone S/R** → Clustering di shadow delle candele con validazione statistica
    4. **Setup Generation** → Combinazione di trend, momentum e prossimità alle zone
    
    ---
    
    **© Kriterion Quant** - Sistema proprietario per trading quantitativo
    """)
