import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import os

# Kendi modüllerimizi import ediyoruz
from modules.data_fetcher import BISTDataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.alert_system import AlertSystem
from modules.config import BIST_SYMBOLS, INDICATORS_CONFIG

# Yeni modüller
from modules.simple_ml_predictor import SimpleMLPredictor
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.stock_screener import StockScreener
from modules.pattern_recognition import PatternRecognition
from fpdf import FPDF
import base64


# Navigation için
from streamlit_option_menu import option_menu

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="BIST Teknik Analiz Uygulaması",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean dark blue borders on expanders
st.markdown("""
<style>
    /* Genel Sayfa Stili */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }

    /* Kenar Çubuğu Stili */
    .css-1d391kg {
        background-color: #252526;
    }

    /* Buton Stilleri */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: transparent;
        color: #4CAF50;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:active {
        background-color: #45a049;
    }

    /* Kart Stilleri */
    .metric-card {
        background-color: #2D2D30;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .metric-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    /* Sekme Stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D2D30;
    }

/* Expander styling with clean dark blue borders */
div[data-testid="stExpander"] {
    border: 2px solid #2E86AB !important;
    border-radius: 8px !important;
    background: transparent !important;
    margin-bottom: 12px !important;
    transition: border-color 0.2s ease !important;
}

div[data-testid="stExpander"]:hover {
    border-color: #4A9FD1 !important;
}

/* Expander header */
div[data-testid="stExpander"] > div:first-child {
    background: rgba(46, 134, 171, 0.1) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}

/* Expander header text */
div[data-testid="stExpander"] summary {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Expander arrow */
div[data-testid="stExpander"] summary svg {
    fill: #2E86AB !important;
}

/* Expander content */
div[data-testid="stExpander"] > div:last-child {
    background: transparent !important;
    border-top: 1px solid rgba(46, 134, 171, 0.3) !important;
    padding: 16px !important;
}

/* Checkbox styling inside expanders */
div[data-testid="stExpander"] label {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    letter-spacing: 0.2px !important;
}

/* Paragraph styling inside expanders */
div[data-testid="stExpander"] p {
    color: #ffffff !important;
    font-size: 14px !important;
}

/* All text elements inside expanders */
div[data-testid="stExpander"] * {
    color: #ffffff !important;
}

div[data-testid="stExpander"] input[type="checkbox"] {
    accent-color: #2E86AB !important;
    transform: scale(1.1) !important;
}

/* Left column width adjustment */
div[data-testid="column"]:first-child {
    max-width: 240px !important;
    min-width: 240px !important;
    border-right: 1px solid rgba(46, 134, 171, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

def create_chart(df, analyzer, selected_indicators):
    """Modern Plotly grafik oluşturur"""
    
    # Alt grafikler oluştur (ana grafik + volume + RSI)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=['Price & Indicators', 'Volume', 'RSI'],
        row_heights=[0.65, 0.12, 0.23],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}], 
               [{"secondary_y": False}]]
    )
    
    # Ana mum grafik
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'], 
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume grafik
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # RSI grafiği (eğer RSI indikatörü seçilmişse)
    if selected_indicators.get('rsi', False) and 'rsi' in analyzer.indicators:
        config = INDICATORS_CONFIG.get('rsi', {})
        rsi_data = analyzer.indicators['rsi']
        
        # Ana RSI çizgisi
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data,
                name="RSI",
                line=dict(
                    color='#ff9800', 
                    width=config.get('line_width', 1)
                )
            ),
            row=3, col=1
        )
        
        # RSI EMA çizgisi (eğer varsa)
        if 'rsi_ema' in analyzer.indicators:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=analyzer.indicators['rsi_ema'],
                    name="RSI EMA",
                    line=dict(color='#2196f3', width=1, dash='dot'),
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # RSI seviyeler
        top_color = config.get('top_line_color', 'red')
        bottom_color = config.get('bottom_line_color', 'blue')
        
        fig.add_hline(y=70, line_dash="dash", line_color=top_color, opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=bottom_color, opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
        
        # Pivot noktaları (eğer varsa ve gösterilmek isteniyorsa)
        if config.get('show_pivot_points', True):
            if 'rsi_pivot_highs' in analyzer.indicators:
                pivot_highs = analyzer.indicators['rsi_pivot_highs']
                valid_highs = pivot_highs.dropna()
                if not valid_highs.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_highs.index,
                            y=valid_highs.values,
                            mode='markers',
                            name="RSI Pivot Highs",
                            marker=dict(
                                color='red',
                                size=6,
                                symbol='triangle-down'
                            ),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
            
            if 'rsi_pivot_lows' in analyzer.indicators:
                pivot_lows = analyzer.indicators['rsi_pivot_lows']
                valid_lows = pivot_lows.dropna()
                if not valid_lows.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_lows.index,
                            y=valid_lows.values,
                            mode='markers',
                            name="RSI Pivot Lows",
                            marker=dict(
                                color='green',
                                size=6,
                                symbol='triangle-up'
                            ),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
        
        # Trend çizgileri (eğer varsa ve gösterilmek isteniyorsa)
        if config.get('show_broken_trend_lines', True) and 'rsi_trend_lines' in analyzer.indicators:
            trend_lines = analyzer.indicators['rsi_trend_lines']
            
            # Direnç çizgileri
            for line in trend_lines.get('resistance_lines', []):
                if config.get('extend_lines', False):
                    # Çizgiyi genişlet
                    end_idx = len(df.index) - 1
                    end_val = line['slope'] * end_idx + line['intercept']
                else:
                    end_idx = line['end_idx']
                    end_val = line['end_val']
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[line['start_idx']], df.index[end_idx]],
                        y=[line['start_val'], end_val],
                        mode='lines',
                        name="RSI Resistance",
                        line=dict(
                            color=top_color,
                            width=1,
                            dash='solid' if config.get('line_style') == 'solid' else 'dash'
                        ),
                        showlegend=False,
                        opacity=0.6
                    ),
                    row=3, col=1
                )
            
            # Destek çizgileri
            for line in trend_lines.get('support_lines', []):
                if config.get('extend_lines', False):
                    # Çizgiyi genişlet
                    end_idx = len(df.index) - 1
                    end_val = line['slope'] * end_idx + line['intercept']
                else:
                    end_idx = line['end_idx']
                    end_val = line['end_val']
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[line['start_idx']], df.index[end_idx]],
                        y=[line['start_val'], end_val],
                        mode='lines',
                        name="RSI Support",
                        line=dict(
                            color=bottom_color,
                            width=1,
                            dash='solid' if config.get('line_style') == 'solid' else 'dash'
                        ),
                        showlegend=False,
                        opacity=0.6
                    ),
                    row=3, col=1
                )
    
    # Teknik indikatörleri ana grafiğe ekle
    for indicator, enabled in selected_indicators.items():
        if enabled and indicator in analyzer.indicators:
            indicator_data = analyzer.indicators[indicator]
            config = INDICATORS_CONFIG.get(indicator, {})
            
            if indicator.startswith('ema') or indicator.startswith('ma_'):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator_data,
                        name=config.get('name', indicator),
                        line=dict(
                            color=config.get('color', '#2196f3'),
                            width=2
                        )
                    ),
                    row=1, col=1
                )
            elif indicator.startswith('vwma') or indicator.startswith('vwema'):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator_data,
                        name=config.get('name', indicator),
                        line=dict(
                            color=config.get('color', '#2196f3'),
                            width=2,
                            dash='dot'
                        )
                    ),
                    row=1, col=1
                )
            
            elif indicator == 'vwap':
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator_data,
                        name=config.get('name', 'VWAP'),
                        line=dict(
                            color=config.get('color', '#ff9ff3'),
                            width=2,
                            dash='dot'
                        )
                    ),
                    row=1, col=1
                )
            
            elif indicator in ['supertrend', 'ott']:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator_data,
                        name=config.get('name', indicator),
                        line=dict(
                            color=config.get('color', '#9c27b0'),
                            width=2
                        )
                    ),
                    row=1, col=1
                )
            
            elif indicator == 'bollinger':
                # Bollinger bantları için özel işlem
                # Ayrı ayrı kaydedilen bb_upper, bb_lower, bb_middle verilerini al
                bb_upper = analyzer.indicators.get('bb_upper')
                bb_lower = analyzer.indicators.get('bb_lower')
                bb_middle = analyzer.indicators.get('bb_middle')
                
                if bb_upper is not None and bb_lower is not None and bb_middle is not None:
                    # Üst bant
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=bb_upper,
                            name="BB Upper",
                            line=dict(color='rgba(158,158,158,0.5)', width=1),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    # Alt bant
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=bb_lower,
                            name="BB Lower",
                            line=dict(color='rgba(158,158,158,0.5)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(158,158,158,0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    # Orta çizgi
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=bb_middle,
                            name="BB Middle",
                            line=dict(color='#9e9e9e', width=1)
                        ),
                        row=1, col=1
                    )
            
            # Gelişmiş indikatörler için görselleştirme
            elif indicator in ['fvg', 'order_block', 'bos', 'fvg_ob_combo', 'fvg_bos_combo']:
                if isinstance(indicator_data, dict):
                    # FVG (Fair Value Gap) görselleştirmesi
                    if 'fvg' in indicator and 'bullish' in indicator_data:
                        bullish_fvg = indicator_data.get('bullish', [])
                        bearish_fvg = indicator_data.get('bearish', [])
                        
                        # Bullish FVG'ler
                        for fvg in bullish_fvg:
                            if len(fvg) >= 4:  # [index, low, high, volume]
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[fvg[0]], x1=df.index[min(fvg[0]+5, len(df.index)-1)],
                                    y0=fvg[1], y1=fvg[2],
                                    fillcolor="rgba(76, 175, 80, 0.2)",
                                    line=dict(color="rgba(76, 175, 80, 0.5)", width=1),
                                    row=1, col=1
                                )
                        
                        # Bearish FVG'ler
                        for fvg in bearish_fvg:
                            if len(fvg) >= 4:  # [index, low, high, volume]
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[fvg[0]], x1=df.index[min(fvg[0]+5, len(df.index)-1)],
                                    y0=fvg[1], y1=fvg[2],
                                    fillcolor="rgba(244, 67, 54, 0.2)",
                                    line=dict(color="rgba(244, 67, 54, 0.5)", width=1),
                                    row=1, col=1
                                )
                    
                    # Order Block görselleştirmesi
                    if 'order_block' in indicator and 'bullish' in indicator_data:
                        bullish_ob = indicator_data.get('bullish', [])
                        bearish_ob = indicator_data.get('bearish', [])
                        
                        # Bullish Order Blocks
                        for ob in bullish_ob:
                            if len(ob) >= 4:  # [index, low, high, volume]
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[ob[0]], x1=df.index[min(ob[0]+10, len(df.index)-1)],
                                    y0=ob[1], y1=ob[2],
                                    fillcolor="rgba(33, 150, 243, 0.15)",
                                    line=dict(color="rgba(33, 150, 243, 0.6)", width=2),
                                    row=1, col=1
                                )
                        
                        # Bearish Order Blocks
                        for ob in bearish_ob:
                            if len(ob) >= 4:  # [index, low, high, volume]
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[ob[0]], x1=df.index[min(ob[0]+10, len(df.index)-1)],
                                    y0=ob[1], y1=ob[2],
                                    fillcolor="rgba(255, 152, 0, 0.15)",
                                    line=dict(color="rgba(255, 152, 0, 0.6)", width=2),
                                    row=1, col=1
                                )
                    
                    # BOS (Break of Structure) görselleştirmesi
                    if 'bos' in indicator and 'bullish' in indicator_data:
                        bullish_bos = indicator_data.get('bullish', [])
                        bearish_bos = indicator_data.get('bearish', [])
                        
                        # Bullish BOS işaretleri
                        for bos in bullish_bos:
                            if len(bos) >= 2:  # [index, price]
                                fig.add_annotation(
                                    x=df.index[bos[0]],
                                    y=bos[1],
                                    text="BOS↑",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="green",
                                    arrowwidth=2,
                                    font=dict(color="green", size=10),
                                    bgcolor="rgba(76, 175, 80, 0.8)",
                                    bordercolor="green",
                                    row=1, col=1
                                )
                        
                        # Bearish BOS işaretleri
                        for bos in bearish_bos:
                            if len(bos) >= 2:  # [index, price]
                                fig.add_annotation(
                                    x=df.index[bos[0]],
                                    y=bos[1],
                                    text="BOS↓",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="red",
                                    arrowwidth=2,
                                    font=dict(color="red", size=10),
                                    bgcolor="rgba(244, 67, 54, 0.8)",
                                    bordercolor="red",
                                    row=1, col=1
                                )
                
                # FVG + Order Block Kombinasyonu görselleştirmesi
                elif indicator == 'fvg_ob_combo' and isinstance(indicator_data, list):
                    for combo in indicator_data:
                        if isinstance(combo, dict) and 'type' in combo:
                            combo_type = combo['type']
                            fvg_zone = combo.get('fvg_zone', (0, 0))
                            order_block = combo.get('order_block', (0, 0))
                            date = combo.get('date')
                            
                            if date and date in df.index:
                                date_idx = df.index.get_loc(date)
                                
                                # FVG bölgesi
                                color = "rgba(108, 92, 231, 0.3)" if combo_type == 'bullish' else "rgba(225, 112, 85, 0.3)"
                                border_color = "rgba(108, 92, 231, 0.7)" if combo_type == 'bullish' else "rgba(225, 112, 85, 0.7)"
                                
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[date_idx], x1=df.index[min(date_idx+8, len(df.index)-1)],
                                    y0=fvg_zone[0], y1=fvg_zone[1],
                                    fillcolor=color,
                                    line=dict(color=border_color, width=2, dash="dash"),
                                    row=1, col=1
                                )
                                
                                # Order Block bölgesi
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[date_idx], x1=df.index[min(date_idx+8, len(df.index)-1)],
                                    y0=order_block[0], y1=order_block[1],
                                    fillcolor=color.replace('0.3', '0.2'),
                                    line=dict(color=border_color, width=3),
                                    row=1, col=1
                                )
                                
                                # Kombinasyon etiketi
                                fig.add_annotation(
                                    x=df.index[date_idx],
                                    y=(fvg_zone[1] + order_block[1]) / 2,
                                    text=f"FVG+OB {'↑' if combo_type == 'bullish' else '↓'}",
                                    showarrow=False,
                                    font=dict(color="white", size=8, family="Arial Black"),
                                    bgcolor=border_color,
                                    bordercolor="white",
                                    borderwidth=1,
                                    row=1, col=1
                                )
                
                # FVG + BOS Kombinasyonu görselleştirmesi
                elif indicator == 'fvg_bos_combo' and isinstance(indicator_data, list):
                    for combo in indicator_data:
                        if isinstance(combo, dict) and 'type' in combo:
                            combo_type = combo['type']
                            fvg_zone = combo.get('fvg_zone', (0, 0))
                            bos_price = combo.get('bos_price', 0)
                            date = combo.get('date')
                            strength = combo.get('strength', 0)
                            confidence = combo.get('confidence', 50)
                            
                            if date and date in df.index:
                                date_idx = df.index.get_loc(date)
                                
                                # FVG bölgesi
                                color = "rgba(0, 184, 148, 0.3)" if combo_type == 'bullish' else "rgba(214, 48, 49, 0.3)"
                                border_color = "rgba(0, 184, 148, 0.8)" if combo_type == 'bullish' else "rgba(214, 48, 49, 0.8)"
                                
                                fig.add_shape(
                                    type="rect",
                                    x0=df.index[date_idx], x1=df.index[min(date_idx+6, len(df.index)-1)],
                                    y0=fvg_zone[0], y1=fvg_zone[1],
                                    fillcolor=color,
                                    line=dict(color=border_color, width=2, dash="dot"),
                                    row=1, col=1
                                )
                                
                                # BOS çizgisi
                                fig.add_hline(
                                    y=bos_price,
                                    line=dict(color=border_color, width=3, dash="solid"),
                                    row=1, col=1
                                )
                                
                                # Kombinasyon etiketi - daha detaylı bilgi
                                fig.add_annotation(
                                    x=df.index[date_idx],
                                    y=bos_price,
                                    text=f"FVG+BOS {'↑' if combo_type == 'bullish' else '↓'}<br>Güven: {confidence:.0f}%<br>Güç: {strength:.0f}",
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowcolor=border_color,
                                    arrowwidth=2,
                                    font=dict(color="white", size=8, family="Arial Black"),
                                    bgcolor=border_color,
                                    bordercolor="white",
                                    borderwidth=1,
                                    row=1, col=1
                                )
    
    # Grafik düzeni ve stil
    fig.update_layout(
        title="",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800
    )
    
    # X ekseni ayarları
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=False,
        gridcolor='rgba(255,255,255,0.1)',
        showline=True,
        linecolor='rgba(255,255,255,0.2)'
    )
    
    # Y ekseni ayarları
    fig.update_yaxes(
        showgrid=False,
        gridcolor='rgba(255,255,255,0.1)',
        showline=True,
        linecolor='rgba(255,255,255,0.2)'
    )
    
    # Volume grafiği için özel ayarlar
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    
    return fig

def main():
    
    # Modern SaaS Dashboard CSS - Tam Shadcn/UI tarzı (Eski CSS sınıfları dahil)
    st.markdown("""
    <style>
        /* Global Reset */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* Modern SaaS Dashboard Theme - Navy Blue */
        .main {
            background-color: hsl(220, 40%, 8%);
            color: hsl(210, 40%, 98%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        
        /* Override Streamlit default backgrounds */
        .stApp {
            background-color: hsl(220, 40%, 8%) !important;
        }
        
        .stApp > header {
            background-color: transparent !important;
        }
        
        .stApp > div > div {
            background-color: hsl(220, 40%, 8%) !important;
        }
        
        /* Additional Streamlit overrides */
        .main .block-container {
            background-color: hsl(220, 40%, 8%) !important;
        }
        
        /* stMainBlockContainer padding override */
        .stMainBlockContainer {
            padding: 2rem !important;
        }
        
        /* Streamlit sidebar overrides */
        .css-1d391kg, section[data-testid="stSidebar"] {
            background-color: hsl(220, 40%, 8%) !important;
            border-right: 1px solid rgba(46, 134, 171, 0.3) !important;
        }
        
        /* Streamlit metric cards */
        div[data-testid="metric-container"] {
            background-color: hsl(220, 45%, 12%) !important;
            border: 1px solid hsl(215, 35%, 18%) !important;
            border-radius: 0.75rem !important;
            padding: 1rem !important;
        }
        
        /* Streamlit columns */
        div[data-testid="column"] {
            background-color: transparent !important;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: visible; color: hsl(210, 40%, 98%); font-weight: 700; font-size: 1.2rem; padding: 0.5rem 1rem;}
        
        /* Sidebar */
        .css-1d391kg {
            background-color: hsl(220, 40%, 8%);
            border-right: 1px solid rgba(46, 134, 171, 0.3);
            width: 200px !important;
            display: block;
            padding-left: 0.5rem;
        }
        
        /* Sidebar Titles */
        .sidebar-section-title {
            color: hsl(210, 40%, 98%) !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.5rem 0 !important;
            margin-left: 0 !important;
        }
        
        /* Sidebar Buttons */
        button[role="button"] {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100% !important;
            color: hsl(210, 40%, 98%) !important;
            background-color: transparent !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
        }
        
        button[role="button"]:hover {
            background-color: hsl(215, 28%, 20%) !important;
        }
        
        /* Main content area */
        .main .block-container {
            padding: 0.5rem 2.5rem 0.5rem 5rem;
            max-width: none;
        }
        
        /* Dashboard Header */
        .dashboard-header {
            margin-bottom: 2rem;
        }
        
        .dashboard-title {
            font-size: 2rem;
            font-weight: 600;
            color: hsl(210, 40%, 98%);
            margin-bottom: 0.5rem;
        }
        
        /* Tab Navigation */
        .tab-navigation {
            display: flex;
            gap: 0.25rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid hsl(215, 28%, 17%);
            padding-bottom: 0;
        }
        
        .tab-item {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem 0.5rem 0 0;
            background: transparent;
            border: none;
            color: hsl(215, 20%, 65%);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }
        
        .tab-item.active {
            color: hsl(210, 40%, 98%);
            background: hsl(215, 28%, 17%);
        }
        
        .tab-item:hover {
            color: hsl(210, 40%, 98%);
            background: hsl(215, 28%, 12%);
        }
        
        /* Modern KPI Cards Grid */
        .kpi-grid, .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        /* Universal Card Styles */
        .kpi-card, .metric-card, .metric-card-modern, .modern-card, .chart-card, .info-card {
            background: hsl(220, 45%, 12%);
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.75rem;
            padding: 1.5rem;
            position: relative;
            transition: border-color 0.2s;
        }
        
        .kpi-card:hover, .metric-card:hover, .metric-card-modern:hover, .modern-card:hover, .chart-card:hover, .info-card:hover {
            border-color: hsl(215, 40%, 25%);
            background: hsl(220, 45%, 15%);
        }
        
        /* KPI Card Elements */
        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .kpi-title, .metric-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: hsl(215, 20%, 65%);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .kpi-trend {
            width: 16px;
            height: 16px;
            color: hsl(215, 20%, 65%);
        }
        
        .kpi-value, .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: hsl(210, 40%, 98%);
            margin-bottom: 0.25rem;
        }
        
        .kpi-change, .metric-change {
            font-size: 0.75rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .kpi-change.positive, .metric-change.positive {
            color: hsl(142, 76%, 36%);
        }
        
        .kpi-change.negative, .metric-change.negative {
            color: hsl(0, 84%, 60%);
        }
        
        .kpi-change.neutral, .metric-change.neutral {
            color: hsl(215, 20%, 65%);
        }
        
        /* Page Headers (Eski ve Yeni) */
        .page-header, .page-header-modern {
            background: hsl(220, 45%, 12%);
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .page-header h1, .page-header-modern h1 {
            font-size: 2rem;
            font-weight: 700;
            color: hsl(210, 40%, 98%) !important;
            margin: 0 0 0.5rem 0;
        }
        
        .page-header p, .page-header-modern p {
            color: hsl(215, 20%, 65%);
            font-size: 1rem;
            margin: 0;
        }
        
        /* Charts Section */
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .chart-header {
            margin-bottom: 1rem;
        }
        
        .chart-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: hsl(210, 40%, 98%);
            margin-bottom: 0.25rem;
        }
        
        .chart-subtitle {
            font-size: 0.875rem;
            color: hsl(215, 20%, 65%);
        }
        
        /* Bottom Section */
        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        .info-card-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: hsl(210, 40%, 98%);
            margin-bottom: 1rem;
        }
        
        .info-card-content {
            color: hsl(215, 20%, 65%);
            font-size: 0.875rem;
        }
        
        /* Signal Cards */
        .signal-card {
            background: hsl(220, 45%, 12%);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: all 0.15s ease-in-out;
        }
        
        .signal-card.buy {
            border: 1px solid hsl(142, 76%, 36%);
            background: hsl(142, 76%, 6%);
        }
        
        .signal-card.sell {
            border: 1px solid hsl(0, 84%, 60%);
            background: hsl(0, 84%, 6%);
        }
        
        .signal-card.hold {
            border: 1px solid hsl(47, 96%, 53%);
            background: hsl(47, 96%, 6%);
        }
        
        .signal-card.neutral {
            border: 1px solid hsl(215, 20%, 65%);
            background: hsl(215, 20%, 6%);
        }
        
        .signal-icon {
            font-size: 1.5rem;
        }
        
        .signal-text {
            font-size: 1.125rem;
            font-weight: 600;
            color: hsl(210, 40%, 98%);
        }
        
        /* Info Boxes (Eski stil uyumlu) */
        .info-box, .warning-box, .error-box, .info-box-modern {
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 1rem 0;
            background: hsl(220, 45%, 12%);
        }
        
        .info-box.success, .info-box-modern.success {
            border-color: hsl(142, 76%, 36%);
            background: hsl(142, 76%, 6%);
        }
        
        .warning-box, .info-box-modern.warning {
            border-color: hsl(47, 96%, 53%);
            background: hsl(47, 96%, 6%);
        }
        
        .error-box, .info-box-modern.error {
            border-color: hsl(0, 84%, 60%);
            background: hsl(0, 84%, 6%);
        }
        
        .info-box h4, .warning-box h4, .error-box h4, .info-box-modern h4 {
            color: hsl(210, 40%, 98%) !important;
            margin-bottom: 0.5rem;
        }
        
        .info-box p, .warning-box p, .error-box p, .info-box-modern p {
            color: hsl(215, 20%, 65%);
            margin: 0;
        }
        
        /* Sidebar Navigation */
        .sidebar-nav {
            padding: 1.5rem 1rem;
        }
        
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem 0.75rem;
            margin-bottom: 2rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: hsl(210, 40%, 98%);
            background: linear-gradient(135deg, hsl(220, 45%, 12%) 0%, hsl(215, 40%, 16%) 100%);
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.75rem;
            backdrop-filter: blur(10px);
        }
        
        .sidebar-section {
            margin-bottom: 1.5rem;
        }
        
        .sidebar-section-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: hsl(215, 20%, 65%);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
            padding: 0 0.75rem;
        }
        
        /* Modern Form Elements */
        .stSelectbox > div > div {
            background: hsl(220, 45%, 12%);
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.5rem;
            color: hsl(210, 40%, 98%);
        }
        
        .stCheckbox > label {
            color: hsl(210, 40%, 98%);
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Modern Sidebar Button Styling */
        .stButton > button {
            width: 100% !important;
            height: 40px !important;
            border-radius: 6px !important;
            border: none !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            transition: all 0.2s ease !important;
            margin-bottom: 4px !important;
            display: flex !important;
            align-items: center !important;
            padding: 10px 12px !important;
        }
        
        /* Secondary button styling (non-active) */
        .stButton > button[kind="secondary"] {
            background-color: transparent !important;
            color: #8B8B8B !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
        }
        
        /* Primary button styling (active) */
        .stButton > button[kind="primary"] {
            background-color: #3B82F6 !important;
            color: #ffffff !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #2563EB !important;
        }

        /* Tertiary button styling (yellow) */
        .stButton > button[kind="tertiary"] {
            background-color: #FFC107 !important;
            color: #000000 !important;
        }

        .stButton > button[kind="tertiary"]:hover {
            background-color: #FFA000 !important;
        }


        
        .stButton > button:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Settings Section Styling */
        .sidebar-settings {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid hsl(215, 28%, 17%);
        }
        
        .sidebar-settings .stButton > button {
            background: hsl(215, 28%, 12%);
            border: 1px solid hsl(215, 28%, 17%);
            color: hsl(215, 20%, 65%);
        }
        
        .sidebar-settings .stButton > button:hover {
            background: hsl(215, 28%, 17%);
            border-color: hsl(215, 28%, 25%);
            color: hsl(210, 40%, 98%);
        }
        
        /* Separator */
        .menu-separator {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, hsl(215, 28%, 17%) 50%, transparent 100%);
            margin: 1rem 0;
        }
        
        /* Main content buttons */
        .stButton > button:not(.css-1d391kg .stButton > button) {
            background: hsl(210, 40%, 98%);
            color: hsl(224, 71%, 4%);
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.15s;
        }
        
        .stButton > button:hover:not(.css-1d391kg .stButton > button) {
            background: hsl(210, 40%, 95%);
        }
        
        /* Chart containers */
        .plotly-graph-div {
            background: transparent;
            border: 1px solid hsl(215, 28%, 15%);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        /* Modern Table */
        .dataframe {
            background: hsl(220, 45%, 12%);
            border: 1px solid hsl(215, 35%, 18%);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: hsl(210, 40%, 98%) !important;
            font-weight: 600;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: hsl(215, 28%, 17%);
        }
        
        ::-webkit-scrollbar-thumb {
            background: hsl(215, 28%, 25%);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: hsl(215, 28%, 30%);
        }
        
        /* Hover effects */
        .hover-glow:hover {
            border-color: hsl(215, 28%, 30%);
            box-shadow: 0 0 0 1px hsl(215, 28%, 30%);
        }
        
        /* Modern Shadcn/UI Sidebar Items */
        .sidebar-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #8B8B8B;
            font-size: 14px;
            font-weight: 500;
            position: relative;
            height: 40px;
            box-sizing: border-box;
        }
        
        .sidebar-item-container:hover .sidebar-item {
            background-color: rgba(255, 255, 255, 0.05);
            color: #ffffff;
        }
        
        .sidebar-item-active {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #ffffff;
            font-size: 14px;
            font-weight: 500;
            background-color: #3B82F6;
            position: relative;
            height: 40px;
            box-sizing: border-box;
        }
        
        .sidebar-icon {
            margin-right: 10px;
            font-size: 16px;
            width: 20px;
            display: inline-block;
        }
        
        .sidebar-text {
            flex-grow: 1;
        }
        
        .sidebar-arrow {
            margin-left: auto;
            font-size: 16px;
            color: #8B8B8B;
        }
        
        /* Sidebar Item Container */
        .sidebar-item-container {
            position: relative;
            margin-bottom: 2px;
            width: 100%;
        }
        
        /* Streamlit Button Reset */
        .stButton {
            position: relative !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .stButton > button {
            background-color: transparent !important;
            border: none !important;
            padding: 10px 12px !important;
            margin: 0 !important;
            height: 40px !important;
            width: 100% !important;
            border-radius: 6px !important;
            cursor: pointer !important;
            color: transparent !important;
            font-size: 0 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            display: flex !important;
            align-items: center !important;
        }
        
        .stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        
        .stButton > button:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Modern Sidebar Brand */
        .sidebar-brand {
            text-align: center;
            padding: 2rem 0 1rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Section Headers */
        .sidebar-section {
            margin: 2rem 0 1rem 0;
        }
        
        .sidebar-section-title {
            color: #8B8B8B;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        
        /* Checkbox Styling */
        .stCheckbox {
            margin-bottom: 8px;
        }
        
        .stCheckbox > label {
            display: flex !important;
            align-items: center !important;
            padding: 6px 8px !important;
            border-radius: 6px !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            font-size: 0.85rem !important;
            color: #CCCCCC !important;
        }
        
        .stCheckbox > label:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
        }
        
        .stCheckbox > label > div:first-child {
            margin-right: 8px !important;
        }
        
        /* Checkbox input styling */
        .stCheckbox input[type="checkbox"] {
            width: 16px !important;
            height: 16px !important;
            border: 1px solid #555 !important;
            border-radius: 3px !important;
            background-color: transparent !important;
        }
        
        .stCheckbox input[type="checkbox"]:checked {
            background-color: #0066CC !important;
            border-color: #0066CC !important;
        }
        
        /* Signal Card Tooltip Styles */
        .signal-card {
            position: relative;
        }
        
        .signal-info-icon {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 18px;
            height: 18px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: hsl(215, 20%, 65%);
            cursor: help;
            transition: all 0.2s ease;
            z-index: 10;
        }
        
        .signal-info-icon:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
            color: hsl(210, 40%, 98%);
            transform: scale(1.1);
        }
        
        .signal-tooltip {
            position: absolute;
            top: -10px;
            right: 35px;
            background: hsl(224, 71%, 8%);
            border: 1px solid hsl(215, 28%, 25%);
            border-radius: 8px;
            padding: 12px;
            width: 280px;
            font-size: 12px;
            line-height: 1.4;
            color: hsl(210, 40%, 98%);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            opacity: 0;
            visibility: hidden;
            transform: translateY(10px);
            transition: all 0.3s ease;
            z-index: 1000;
            pointer-events: none;
        }
        
        .signal-info-icon:hover + .signal-tooltip,
        .signal-tooltip:hover {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
            pointer-events: auto;
        }
        
        .signal-tooltip::before {
            content: '';
            position: absolute;
            top: 15px;
            right: -6px;
            width: 12px;
            height: 12px;
            background: hsl(224, 71%, 8%);
            border-right: 1px solid hsl(215, 28%, 25%);
            border-bottom: 1px solid hsl(215, 28%, 25%);
            transform: rotate(-45deg);
        }
        
        .tooltip-title {
            font-weight: 600;
            font-size: 13px;
            color: hsl(210, 40%, 98%);
            margin-bottom: 6px;
            border-bottom: 1px solid hsl(215, 28%, 17%);
            padding-bottom: 4px;
        }
        
        .tooltip-description {
            color: hsl(215, 20%, 80%);
            margin-bottom: 8px;
        }
        
        .tooltip-criteria {
            color: hsl(215, 20%, 70%);
            font-size: 11px;
        }
        
        .tooltip-criteria strong {
            color: hsl(210, 40%, 90%);
        }
        
        /* Indicator Card Tooltip Styles */
        .metric-card {
            position: relative;
        }
        
        .metric-info-icon {
            position: absolute;
            top: 6px;
            right: 6px;
            width: 16px;
            height: 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 600;
            color: hsl(215, 20%, 60%);
            cursor: help;
            transition: all 0.2s ease;
            z-index: 10;
        }
        
        .metric-info-icon:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.25);
            color: hsl(210, 40%, 95%);
            transform: scale(1.1);
        }
        
        .metric-tooltip {
            position: absolute;
            top: -8px;
            right: 28px;
            background: hsl(224, 71%, 6%);
            border: 1px solid hsl(215, 28%, 20%);
            border-radius: 6px;
            padding: 10px;
            width: 260px;
            font-size: 11px;
            line-height: 1.3;
            color: hsl(210, 40%, 95%);
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            opacity: 0;
            visibility: hidden;
            transform: translateY(8px);
            transition: all 0.25s ease;
            z-index: 1000;
            pointer-events: none;
        }
        
        .metric-info-icon:hover + .metric-tooltip,
        .metric-tooltip:hover {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
            pointer-events: auto;
        }
        
        .metric-tooltip::before {
            content: '';
            position: absolute;
            top: 12px;
            right: -5px;
            width: 10px;
            height: 10px;
            background: hsl(224, 71%, 6%);
            border-right: 1px solid hsl(215, 28%, 20%);
            border-bottom: 1px solid hsl(215, 28%, 20%);
            transform: rotate(-45deg);
        }
        
        .metric-tooltip-title {
            font-weight: 600;
            font-size: 12px;
            color: hsl(210, 40%, 98%);
            margin-bottom: 4px;
            border-bottom: 1px solid hsl(215, 28%, 15%);
            padding-bottom: 3px;
        }
        
        .metric-tooltip-description {
            color: hsl(215, 20%, 75%);
            margin-bottom: 6px;
        }
        
        .metric-tooltip-range {
            color: hsl(215, 20%, 65%);
            font-size: 10px;
        }
        
        .metric-tooltip-range strong {
            color: hsl(210, 40%, 85%);
        }
        
        /* Tab Styling - Beyaz başlıklar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border: 1px solid hsl(215, 35%, 18%) !important;
            border-radius: 8px !important;
            color: hsl(210, 40%, 98%) !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 12px 16px !important;
            margin: 0 !important;
            height: auto !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: hsl(220, 45%, 15%) !important;
            border-color: hsl(215, 35%, 25%) !important;
            color: hsl(210, 40%, 100%) !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: hsl(220, 45%, 18%) !important;
            border-color: hsl(215, 35%, 30%) !important;
            color: hsl(210, 40%, 100%) !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Modern Shadcn/UI Sidebar Navigation
    with st.sidebar:
        # Brand Header
        st.markdown("""
        <div style="display: flex; align-items: center; padding: 1.5rem 1rem; margin-bottom: 1rem;">
            <div style="width: 32px; height: 32px; background: #3B82F6; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                <span style="color: white; font-size: 18px;">📊</span>
            </div>
            <div>
                <div style="color: #ffffff; font-weight: 600; font-size: 1rem; line-height: 1.2;">TraderLand</div>
                <div style="color: #8B8B8B; font-size: 0.75rem;">Dashboard + Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        if "selected_menu" not in st.session_state:
            st.session_state.selected_menu = "dashboard"
        
        current_menu = st.session_state.selected_menu
        
        # General Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">
                General
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dashboard
        if st.button("📊 Dashboard", key="dashboard_btn", use_container_width=True, 
                    type="primary" if current_menu == "dashboard" else "secondary"):
            st.session_state.selected_menu = "dashboard"
            st.rerun()
        
        # Technical Analysis
        if st.button("📈 Teknik Analiz", key="technical_btn", use_container_width=True,
                    type="primary" if current_menu == "technical" else "secondary"):
            st.session_state.selected_menu = "technical"
            st.rerun()
        
        # Analysis Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">
                Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Predictions
        if st.button("🤖 AI Tahminleri", key="ai_btn", use_container_width=True,
                    type="primary" if current_menu == "ai" else "secondary"):
            st.session_state.selected_menu = "ai"
            st.rerun()
        
        # Stock Screener
        if st.button("🔍 Hisse Tarayıcı", key="screener_btn", use_container_width=True,
                    type="primary" if current_menu == "screener" else "secondary"):
            st.session_state.selected_menu = "screener"
            st.rerun()
        
        # Pattern Analysis
        if st.button("🎯 Patern Analizi", key="pattern_btn", use_container_width=True,
                    type="primary" if current_menu == "pattern" else "secondary"):
            st.session_state.selected_menu = "pattern"
            st.rerun()
        

        
        # Tools Section
        st.markdown("""
        <div style="margin: 1.5rem 0 1rem 0;">
            <div style="color: #8B8B8B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; padding-left: 0.5rem;">
                Tools
            </div>
        </div>
        """, unsafe_allow_html=True)
        

        



    
    # Seçili menüye göre sayfa yönlendirmesi
    current_menu = st.session_state.selected_menu
    
    # Modern sayfa geçişi
    if current_menu == "dashboard":
        show_modern_dashboard()
    elif current_menu == "technical":
        show_technical_analysis()
    elif current_menu == "ai":
        show_ai_predictions()
    elif current_menu == "screener":
        show_stock_screener()
    elif current_menu == "pattern":
        show_pattern_analysis()



    else:
        # Varsayılan olarak dashboard göster
        show_modern_dashboard()

def show_technical_analysis():
    """Teknik analiz sayfası - Modern Shadcn stil"""
    
    st.markdown("""
    <div class="page-header">
        <h1 style="margin: 0;">📈 Teknik Analiz</h1>
    </div>
    """, unsafe_allow_html=True)
    


    # Hisse seçimi, zaman aralığı, dönem ve uyarı kontrolleri - Üst bölüm
    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 1])
        
    with control_col1:
            st.markdown("""
            <div style="background: hsl(220, 100%, 5%); padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border: 1px solid hsl(215, 28%, 18%);">
                <div style="color: hsl(210, 40%, 98%); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">📊 Hisse Seçimi</div>
            </div>
            """, unsafe_allow_html=True)
            selected_symbol = st.selectbox(
                "Hisse",
                options=sorted(list(BIST_SYMBOLS.keys())),
                format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
                label_visibility="collapsed",
                key="content_symbol"
            )
        
    with control_col2:
            st.markdown("""
            <div style="background: hsl(220, 100%, 5%); padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border: 1px solid hsl(215, 28%, 18%);">
                <div style="color: hsl(210, 40%, 98%); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">⏰ Zaman Aralığı</div>
            </div>
            """, unsafe_allow_html=True)
            time_interval = st.selectbox(
                "Aralık",
                ["5m", "15m", "1h", "2h", "4h", "1d"],
                index=5,
                format_func=lambda x: {
                    "5m": "5 Dakika", "15m": "15 Dakika", "1h": "1 Saat",
                    "2h": "2 Saat", "4h": "4 Saat", "1d": "1 Gün"
                }[x],
                label_visibility="collapsed",
                key="content_interval"
            )
        
    with control_col3:
            st.markdown("""
            <div style="background: hsl(220, 100%, 5%); padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border: 1px solid hsl(215, 28%, 18%);">
                <div style="color: hsl(210, 40%, 98%); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">📅 Dönem</div>
            </div>
            """, unsafe_allow_html=True)
            
            if time_interval in ["5m", "15m"]:
                # Yahoo Finance API limiti: 15m için maksimum 60 gün
                period_options = ["1d", "7d", "30d", "60d"]
                default_period = "30d"
            elif time_interval in ["1h", "2h", "4h"]:
                period_options = ["7d", "30d", "90d", "6mo", "1y", "2y"] 
                default_period = "1y"
            else:
                period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
                default_period = "1y"
            
            time_period = st.selectbox(
                "Dönem",
                period_options,
                index=period_options.index(default_period),
                format_func=lambda x: {
                    "1d": "1 Gün", "7d": "7 Gün", "30d": "30 Gün", "60d": "60 Gün", "90d": "90 Gün",
                    "1mo": "1 Ay", "3mo": "3 Ay", "6mo": "6 Ay", 
                    "1y": "1 Yıl", "2y": "2 Yıl", "5y": "5 Yıl"
                }.get(x, x),
                label_visibility="collapsed",
                key="content_period"
            )
    
    with control_col4:
            # Uyarı Ayarları
            st.markdown("""
            <div style="background: hsl(220, 100%, 5%); padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border: 1px solid hsl(215, 28%, 18%);">
                <div style="color: hsl(210, 40%, 98%); font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">🚨 Uyarılar</div>
            </div>
            """, unsafe_allow_html=True)
            alert_methods = st.multiselect(
                "Yöntem",
                ["Email", "Telegram", "Desktop"], 
                default=["Desktop"],
                key="alert_methods",
                label_visibility="collapsed"
            )
        
    st.markdown("<br>", unsafe_allow_html=True)  # Boşluk ekle
    
    # İndikatör Seçimi - Kompakt Dropdown'lar
    indicator_col1, indicator_col2, indicator_col3, indicator_col4 = st.columns(4)
    
    selected_indicators = {}
    
    with indicator_col1:
        # Hareketli Ortalamalar Dropdown
        st.markdown("""
        <div style="color: #00ff00; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
        📊 MA/EMA
        </div>
        """, unsafe_allow_html=True)
        ema_indicators = ['ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_50', 'ema_121', 'ma_200', 'vwma_5', 'vwema_5', 'vwema_20']
        selected_ema_list = st.multiselect(
            "MA/EMA Seç",
            options=[INDICATORS_CONFIG[ind]["name"] for ind in ema_indicators if ind in INDICATORS_CONFIG],
            default=[INDICATORS_CONFIG[ind]["name"] for ind in ema_indicators if ind in INDICATORS_CONFIG and INDICATORS_CONFIG[ind]["default"]],
            key="ema_dropdown",
            label_visibility="collapsed"
        )
        # Convert back to indicator keys
        for indicator in ema_indicators:
            if indicator in INDICATORS_CONFIG:
                selected_indicators[indicator] = INDICATORS_CONFIG[indicator]["name"] in selected_ema_list
    
    with indicator_col2:
        # Ana İndikatörler Dropdown
        st.markdown("""
        <div style="color: #00ff00; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
        📈 Ana İnd.
        </div>
        """, unsafe_allow_html=True)
        main_indicators = ['ott', 'supertrend', 'vwap', 'rsi', 'macd']
        selected_main_list = st.multiselect(
            "Ana İndikatör",
            options=[INDICATORS_CONFIG[ind]["name"] for ind in main_indicators if ind in INDICATORS_CONFIG],
            default=[INDICATORS_CONFIG[ind]["name"] for ind in main_indicators if ind in INDICATORS_CONFIG and INDICATORS_CONFIG[ind]["default"]],
            key="main_dropdown",
            label_visibility="collapsed"
        )
        # Convert back to indicator keys
        for indicator in main_indicators:
            if indicator in INDICATORS_CONFIG:
                selected_indicators[indicator] = INDICATORS_CONFIG[indicator]["name"] in selected_main_list
    
    with indicator_col3:
        # Diğer İndikatörler Dropdown
        st.markdown("""
        <div style="color: #00ff00; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
        📊 Diğer İnd.
        </div>
        """, unsafe_allow_html=True)
        other_indicators = ['bollinger', 'stoch', 'williams_r', 'cci']
        selected_other_list = st.multiselect(
            "Diğer İndikatör",
            options=[INDICATORS_CONFIG[ind]["name"] for ind in other_indicators if ind in INDICATORS_CONFIG],
            default=[INDICATORS_CONFIG[ind]["name"] for ind in other_indicators if ind in INDICATORS_CONFIG and INDICATORS_CONFIG[ind]["default"]],
            key="other_dropdown",
            label_visibility="collapsed"
        )
        # Convert back to indicator keys
        for indicator in other_indicators:
            if indicator in INDICATORS_CONFIG:
                selected_indicators[indicator] = INDICATORS_CONFIG[indicator]["name"] in selected_other_list
    
    with indicator_col4:
        # Gelişmiş Formasyonlar
        st.markdown("""
        <div style="color: #00ff00; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
        🔍 Gelişmiş
        </div>
        """, unsafe_allow_html=True)
        advanced_indicators = ['fvg', 'order_block', 'bos', 'fvg_ob_combo', 'fvg_bos_combo']
        selected_advanced_list = st.multiselect(
            "Gelişmiş Form.",
            options=[INDICATORS_CONFIG[ind]["name"] for ind in advanced_indicators if ind in INDICATORS_CONFIG],
            default=[INDICATORS_CONFIG[ind]["name"] for ind in advanced_indicators if ind in INDICATORS_CONFIG and INDICATORS_CONFIG[ind]["default"]],
            key="advanced_dropdown",
            label_visibility="collapsed"
        )
        # Convert back to indicator keys
        for indicator in advanced_indicators:
            if indicator in INDICATORS_CONFIG:
                selected_indicators[indicator] = INDICATORS_CONFIG[indicator]["name"] in selected_advanced_list
    

    
    st.markdown("<br>", unsafe_allow_html=True)  # Boşluk ekle
    
    # Ana grafik alanı
    try:
        with st.spinner("Veriler yükleniyor..."):
            fetcher = BISTDataFetcher()
            df = fetcher.get_stock_data(selected_symbol, period=time_period, interval=time_interval)
    
        if df is not None and not df.empty:
            # Piyasa bilgilerini header'da güncelle
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = latest['Close'] - prev['Close']
            change_pct = (change / prev['Close']) * 100
            volume_change = ((latest['Volume'] - df['Volume'].tail(20).mean()) / df['Volume'].tail(20).mean()) * 100
            
            # Haftalık ve aylık performans hesapla
            weekly_performance = 0
            monthly_performance = 0
            
            try:
                # Haftalık performans (7 gün öncesi ile karşılaştır)
                if len(df) >= 7:
                    week_ago_price = df['Close'].iloc[-7]
                    weekly_performance = ((latest['Close'] - week_ago_price) / week_ago_price) * 100
                
                # Aylık performans (30 gün öncesi ile karşılaştır)
                if len(df) >= 30:
                    month_ago_price = df['Close'].iloc[-30]
                    monthly_performance = ((latest['Close'] - month_ago_price) / month_ago_price) * 100
            except:
                pass  # Hata durumunda 0 olarak kalacak
            
            # Piyasa bilgilerini header altında tek satır halinde göster
            st.markdown(f"""
            <div style='background: hsl(220, 100%, 6%); padding: 0.75rem; border-radius: 0.5rem; margin: 0.5rem 0; border: 1px solid hsl(215, 28%, 20%);'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;'>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <span style='color: hsl(215, 20%, 70%); font-size: 1.1rem;'>📊 {selected_symbol}</span>
                        <span style='color: hsl(210, 40%, 98%); font-weight: 600; font-size: 1.3rem;'>₺{latest['Close']:.2f}</span>
                        <span style='color: {'hsl(142, 76%, 36%)' if change >= 0 else 'hsl(0, 84%, 60%)'}; font-size: 1.1rem;'>{change:+.2f} ({change_pct:+.2f}%)</span>
                    </div>
                    <div style='display: flex; gap: 1.5rem; font-size: 1rem;'>
                        <div>
                            <span style='color: hsl(215, 20%, 70%);'>Yüksek: </span>
                            <span style='color: hsl(210, 40%, 98%);'>₺{latest['High']:.2f}</span>
                        </div>
                        <div>
                            <span style='color: hsl(215, 20%, 70%);'>Düşük: </span>
                            <span style='color: hsl(210, 40%, 98%);'>₺{latest['Low']:.2f}</span>
                        </div>
                        <div>
                            <span style='color: hsl(215, 20%, 70%);'>Hacim: </span>
                            <span style='color: hsl(210, 40%, 98%);'>{latest['Volume']:,.0f}</span>
                            <span style='color: {'hsl(142, 76%, 36%)' if volume_change >= 0 else 'hsl(0, 84%, 60%)'}; margin-left: 0.25rem;'>({volume_change:+.1f}%)</span>
                        </div>
                        <div>
                            <span style='color: hsl(215, 20%, 70%);'>Haftalık: </span>
                            <span style='color: {'hsl(142, 76%, 36%)' if weekly_performance >= 0 else 'hsl(0, 84%, 60%)'};'>{weekly_performance:+.1f}%</span>
                        </div>
                        <div>
                            <span style='color: hsl(215, 20%, 70%);'>Aylık: </span>
                            <span style='color: {'hsl(142, 76%, 36%)' if monthly_performance >= 0 else 'hsl(0, 84%, 60%)'};'>{monthly_performance:+.1f}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            analyzer = TechnicalAnalyzer(df)
            
            # İndikatörleri hesapla
            for indicator, enabled in selected_indicators.items():
                if enabled:
                    analyzer.add_indicator(indicator)
            
            # Ayı sinyalleri için gerekli indikatörleri hesapla
            try:
                # MA 200 için 1 yıllık veri gerekli, eğer mevcut veri yetersizse 1y ile çek
                if len(df) < 200:
                    df_long = fetcher.get_stock_data(selected_symbol, period="1y", interval=time_interval)
                    if df_long is not None and len(df_long) >= 200:
                        analyzer_ma200 = TechnicalAnalyzer(df_long)
                        analyzer_ma200.add_indicator('ma_200')
                        # MA200 değerini ana analyzer'a aktar
                        analyzer.indicators['ma_200'] = analyzer_ma200.indicators['ma_200'].tail(len(df))
                else:
                    analyzer.add_indicator('ma_200')
            except:
                pass  # MA 200 hesaplanamazsa devam et
                
            # Diğer kısa vadeli indikatörler
            for short_indicator in ['ema_5', 'ema_8', 'vwap']:
                try:
                    analyzer.add_indicator(short_indicator)
                except:
                    pass
            
            # Grafik
            fig = create_chart(df, analyzer, selected_indicators)
            st.plotly_chart(fig, use_container_width=True)
            
            # İndikatör Değerleri - Grafik Altında
            if any(selected_indicators.values()):
                st.markdown("""
                <div style='
                    margin: 1rem 0;
                    padding: 1rem;
                    border: 1px solid hsl(215, 28%, 20%);
                    border-radius: 0.5rem;
                    background: hsl(220, 45%, 12%);
                '>
                    <h3 style='
                        color: hsl(210, 40%, 98%); 
                        margin: 0 0 1rem 0; 
                        font-size: 1.1rem;
                        font-weight: 600;
                    '>İndikatör Değerleri</h3>
                """, unsafe_allow_html=True)
                
                indicator_values = analyzer.get_latest_indicators()
                current_price = latest['Close']
                
                # EMA olmayan indikatörler için
                non_ema_indicators = {k: v for k, v in selected_indicators.items() 
                                    if v and k not in ['ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_50', 'ema_121', 'ma_200', 'vwma_5', 'vwema_5']}
                
                if non_ema_indicators:
                    # İndikatör kartları - 4 sütunlu grid
                    indicator_cols = st.columns(min(len(non_ema_indicators), 4))
                    
                    col_idx = 0
                    for indicator, enabled in non_ema_indicators.items():
                        if enabled and indicator in indicator_values:
                            value = indicator_values[indicator]
                            config = INDICATORS_CONFIG.get(indicator, {})
                            
                            # İndikatör durumunu belirleme
                            status_class = "neutral"
                            status_text = "Nötr"
                            status_icon = "⚪"
                            status_color = "hsl(215, 20%, 70%)"
                            
                            if indicator == 'rsi':
                                if value > 70:
                                    status_class = "negative"
                                    status_text = "Aşırı Alım"
                                    status_icon = "🔴"
                                    status_color = "hsl(0, 84%, 60%)"
                                elif value < 30:
                                    status_class = "positive"
                                    status_text = "Aşırı Satım"
                                    status_icon = "🟢"
                                    status_color = "hsl(142, 76%, 36%)"
                                else:
                                    status_class = "neutral"
                                    status_text = "Normal"
                                    status_icon = "⚪"
                                    status_color = "hsl(215, 20%, 70%)"
                            
                            elif indicator == 'macd':
                                if value > 0:
                                    status_class = "positive"
                                    status_text = "Pozitif"
                                    status_icon = "🟢"
                                    status_color = "hsl(142, 76%, 36%)"
                                else:
                                    status_class = "negative"
                                    status_text = "Negatif"
                                    status_icon = "🔴"
                                    status_color = "hsl(0, 84%, 60%)"
                            
                            elif indicator == 'vwap':
                                if current_price > value:
                                    status_class = "positive"
                                    status_text = "Üzeri"
                                    status_icon = "🟢"
                                    status_color = "hsl(142, 76%, 36%)"
                                else:
                                    status_class = "negative"
                                    status_text = "Altı"
                                    status_icon = "🔴"
                                    status_color = "hsl(0, 84%, 60%)"
                            
                            with indicator_cols[col_idx % len(indicator_cols)]:
                                st.markdown(f"""
                                <div style='
                                    background: hsl(215, 35%, 18%);
                                    border: 1px solid hsl(215, 35%, 25%);
                                    border-radius: 0.375rem;
                                    padding: 0.75rem;
                                    text-align: center;
                                    margin-bottom: 0.5rem;
                                '>
                                    <div style='
                                        color: hsl(215, 20%, 70%);
                                        font-size: 0.75rem;
                                        font-weight: 500;
                                        margin-bottom: 0.25rem;
                                    '>{config.get('name', indicator)}</div>
                                    <div style='
                                        color: hsl(210, 40%, 98%);
                                        font-size: 1rem;
                                        font-weight: 600;
                                        margin-bottom: 0.25rem;
                                    '>{value:.2f}</div>
                                    <div style='
                                        color: {status_color};
                                        font-size: 0.7rem;
                                        font-weight: 500;
                                    '>{status_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        col_idx += 1
                
                # EMA değerleri için ayrı bölüm
                ema_indicators = ['ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_50', 'ema_121', 'ma_200', 'vwma_5', 'vwema_5', 'vwema_20']
                selected_emas = [ind for ind in ema_indicators if selected_indicators.get(ind, False)]
                
                if selected_emas:
                    st.markdown("""
                    <div style='
                        margin-top: 1.5rem;
                        padding-top: 1.5rem;
                        border-top: 1px solid hsl(215, 28%, 20%);
                    '>
                        <h4 style='
                            color: hsl(210, 40%, 98%);
                            margin: 0 0 1rem 0;
                            font-size: 1rem;
                            font-weight: 600;
                        '>📏 Hareketli Ortalama Değerleri</h4>
                    """, unsafe_allow_html=True)
                    
                    # EMA değerleri - yan yana grid düzeni
                    ema_cols = st.columns(min(len(selected_emas), 3))  # Maksimum 3 sütun
                    
                    for i, indicator in enumerate(selected_emas):
                        if indicator in indicator_values:
                            ema_value = indicator_values[indicator]
                            distance = current_price - ema_value
                            distance_pct = (distance / ema_value) * 100
                            
                            config = INDICATORS_CONFIG.get(indicator, {})
                            distance_color = "hsl(142, 76%, 36%)" if distance >= 0 else "hsl(0, 84%, 60%)"
                            
                            with ema_cols[i % len(ema_cols)]:
                                st.markdown(f"""
                                <div style='
                                    background: hsl(215, 35%, 18%);
                                    border: 1px solid hsl(215, 35%, 25%);
                                    border-radius: 0.5rem;
                                    padding: 0.75rem;
                                    margin-bottom: 0.75rem;
                                    text-align: center;
                                    height: 120px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: center;
                                '>
                                    <div style='
                                        color: hsl(215, 20%, 70%);
                                        font-size: 0.875rem;
                                        font-weight: 500;
                                        margin-bottom: 0.5rem;
                                    '>{config.get('name', indicator)}</div>
                                    <div style='
                                        color: hsl(210, 40%, 98%);
                                        font-size: 1.1rem;
                                        font-weight: 600;
                                        margin-bottom: 0.5rem;
                                    '>₺{ema_value:.2f}</div>
                                    <div style='
                                        color: {distance_color};
                                        font-size: 0.875rem;
                                        font-weight: 500;
                                    '>{distance:+.2f} ({distance_pct:+.1f}%)</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Signal
                alert_system = AlertSystem()
                signal = alert_system.generate_signal(analyzer)
                
                # Bear Signal
                bear_signal = alert_system.generate_bear_signal(analyzer)
                
                # Kapsamlı Risk Analizi
                risk_analysis = alert_system.generate_comprehensive_risk_analysis(analyzer)
                
                # Pozisyon Önerisi (Yeni Sistem)
                position_recommendation = alert_system.generate_position_recommendation(analyzer)
                
                # VWAP Boğa Sinyali Kontrolü
                vwap_bull_signal = False
                vwap_signal_strength = "Zayıf"
                
                if 'vwap' in analyzer.indicators and len(df) >= 10:
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    vwap_current = analyzer.indicators['vwap'].iloc[-1]
                    vwap_prev = analyzer.indicators['vwap'].iloc[-2]
                    
                    # VWAP Crossover kontrolü (fiyat VWAP'i yukarı kesmiş mi?)
                    if prev_price <= vwap_prev and current_price > vwap_current:
                        vwap_bull_signal = True
                        
                        # Hacim artışı kontrolü
                        current_volume = df['Volume'].iloc[-1]
                        avg_volume = df['Volume'].tail(20).mean()
                        volume_increase = current_volume > (avg_volume * 1.2)  # 20% hacim artışı
                        
                        # RSI(5) ve MACD onayı
                        rsi_confirm = False
                        macd_confirm = False
                        
                        if 'rsi' in analyzer.indicators:
                            rsi_value = analyzer.indicators['rsi'].iloc[-1]
                            rsi_confirm = rsi_value > 50
                        
                        if 'macd' in analyzer.indicators:
                            macd_current = analyzer.indicators['macd'].iloc[-1]
                            macd_prev = analyzer.indicators['macd'].iloc[-2]
                            macd_confirm = macd_current > macd_prev  # MACD yukarı trend
                        
                        # Sinyal gücünü belirleme
                        confirmations = sum([volume_increase, rsi_confirm, macd_confirm])
                        if confirmations >= 2:
                            vwap_signal_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            vwap_signal_strength = "Güçlü"
                        else:
                            vwap_signal_strength = "Orta"
                    
                # Golden Cross Boğa Sinyali Kontrolü
                golden_cross_signal = False
                golden_cross_strength = "Zayıf"
                
                if ('ema_21' in analyzer.indicators and 'ema_50' in analyzer.indicators and 
                    len(df) >= 50):
                        
                        ema21_current = analyzer.indicators['ema_21'].iloc[-1]
                        ema21_prev = analyzer.indicators['ema_21'].iloc[-2]
                        ema50_current = analyzer.indicators['ema_50'].iloc[-1]
                        ema50_prev = analyzer.indicators['ema_50'].iloc[-2]
                        
                        # Golden Cross kontrolü (EMA21 EMA50'yi yukarı kesmiş mi?)
                        if (ema21_prev <= ema50_prev and ema21_current > ema50_current):
                            golden_cross_signal = True
                            
                            # Hacim onayı
                            current_volume = df['Volume'].iloc[-1]
                            avg_volume_20 = df['Volume'].tail(20).mean()
                            volume_confirm = current_volume > (avg_volume_20 * 1.3)  # 30% hacim artışı
                            
                            # RSI ve MACD güç onayı
                            rsi_strong = False
                            macd_strong = False
                            
                            if 'rsi' in analyzer.indicators:
                                rsi_value = analyzer.indicators['rsi'].iloc[-1]
                                rsi_strong = rsi_value > 55
                            
                            if 'macd' in analyzer.indicators:
                                macd_value = analyzer.indicators['macd'].iloc[-1]
                                macd_strong = macd_value > 0
                            
                            # Sinyal gücünü belirleme
                            power_confirmations = sum([volume_confirm, rsi_strong, macd_strong])
                            if power_confirmations >= 2:
                                golden_cross_strength = "Çok Güçlü"
                            elif power_confirmations == 1:
                                golden_cross_strength = "Güçlü"
                            else:
                                golden_cross_strength = "Orta"
                    
                # MACD Boğa Sinyali Kontrolü
                macd_bull_signal = False
                macd_signal_strength = "Zayıf"
                
                if ('macd' in analyzer.indicators and 'macd_signal' in analyzer.indicators and 
                    len(df) >= 26):
                        
                        macd_current = analyzer.indicators['macd'].iloc[-1]
                        macd_prev = analyzer.indicators['macd'].iloc[-2]
                        macd_signal_current = analyzer.indicators['macd_signal'].iloc[-1]
                        macd_signal_prev = analyzer.indicators['macd_signal'].iloc[-2]
                    
                        # MACD Bullish Crossover kontrolü
                        if (macd_prev <= macd_signal_prev and macd_current > macd_signal_current):
                            macd_bull_signal = True
                            
                            # Hacim onayı
                            current_volume = df['Volume'].iloc[-1]
                            avg_volume_15 = df['Volume'].tail(15).mean()
                            volume_confirm = current_volume > (avg_volume_15 * 1.25)  # 25% hacim artışı
                            
                            # RSI ve fiyat trend onayı
                            rsi_confirm = False
                            price_trend_confirm = False
                            
                            if 'rsi' in analyzer.indicators:
                                rsi_value = analyzer.indicators['rsi'].iloc[-1]
                                rsi_confirm = rsi_value > 45  # RSI nötral üstünde
                            
                            # Fiyat son 5 mum üzerinde yukarı trend mi?
                            if len(df) >= 5:
                                price_trend = df['Close'].tail(5).is_monotonic_increasing
                                price_trend_confirm = price_trend or (df['Close'].iloc[-1] > df['Close'].iloc[-3])
                            
                            # Sinyal gücünü belirleme
                            confirmations = sum([volume_confirm, rsi_confirm, price_trend_confirm])
                            if confirmations >= 2:
                                macd_signal_strength = "Çok Güçlü"
                            elif confirmations == 1:
                                macd_signal_strength = "Güçlü"
                            else:
                                macd_signal_strength = "Orta"
                
                # RSI Toparlanma Sinyali Kontrolü
                rsi_recovery_signal = False
                rsi_recovery_strength = "Zayıf"
                
                if 'rsi' in analyzer.indicators and len(df) >= 14:
                    rsi_current = analyzer.indicators['rsi'].iloc[-1]
                    rsi_prev = analyzer.indicators['rsi'].iloc[-2]
                    rsi_3_candles_ago = analyzer.indicators['rsi'].iloc[-4] if len(df) >= 4 else rsi_prev
                    
                    # RSI Oversold Recovery kontrolü (30'un altından 40'ın üzerine çıkış)
                    if (rsi_3_candles_ago <= 30 and rsi_current > 40 and rsi_current > rsi_prev):
                        rsi_recovery_signal = True
                        
                        # Hacim ve momentum onayı
                        current_volume = df['Volume'].iloc[-1]
                        avg_volume_10 = df['Volume'].tail(10).mean()
                        volume_confirm = current_volume > avg_volume_10
                        
                        # Fiyat momentum onayı
                        price_momentum = df['Close'].iloc[-1] > df['Close'].iloc[-2]
                        
                        # MACD onayı
                        macd_confirm = False
                        if 'macd' in analyzer.indicators:
                            macd_current = analyzer.indicators['macd'].iloc[-1]
                            macd_prev = analyzer.indicators['macd'].iloc[-2]
                            macd_confirm = macd_current > macd_prev
                        
                        # Sinyal gücünü belirleme
                        confirmations = sum([volume_confirm, price_momentum, macd_confirm])
                        if confirmations >= 2:
                            rsi_recovery_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            rsi_recovery_strength = "Güçlü"
                        else:
                            rsi_recovery_strength = "Orta"
                
                # Bollinger Sıkışma Sinyali Kontrolü
                bollinger_breakout_signal = False
                bollinger_breakout_strength = "Zayıf"
                
                if ('bollinger_upper' in analyzer.indicators and 'bollinger_lower' in analyzer.indicators and 
                    len(df) >= 20):
                    
                    bb_upper = analyzer.indicators['bollinger_upper'].iloc[-1]
                    bb_lower = analyzer.indicators['bollinger_lower'].iloc[-1]
                    bb_middle = analyzer.indicators['bollinger_middle'].iloc[-1]
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    
                    # Bollinger Band Squeeze kontrolü (bantlar dar mı?)
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    bb_width_5_ago = (analyzer.indicators['bollinger_upper'].iloc[-6] - 
                                     analyzer.indicators['bollinger_lower'].iloc[-6]) / \
                                    analyzer.indicators['bollinger_middle'].iloc[-6] if len(df) >= 6 else bb_width
                    
                    # Fiyat üst banda kırılım yaptı mı?
                    if (prev_price <= bb_middle and current_price > bb_upper and bb_width < bb_width_5_ago):
                        bollinger_breakout_signal = True
                        
                        # Hacim patlaması onayı
                        current_volume = df['Volume'].iloc[-1]
                        avg_volume_20 = df['Volume'].tail(20).mean()
                        volume_explosion = current_volume > (avg_volume_20 * 1.5)  # 50% hacim artışı
                        
                        # RSI destekli momentum
                        rsi_support = False
                        if 'rsi' in analyzer.indicators:
                            rsi_value = analyzer.indicators['rsi'].iloc[-1]
                            rsi_support = 50 < rsi_value < 80  # Güçlü ama aşırı alım değil
                        
                        # Fiyat momentum onayı
                        price_momentum = (current_price - prev_price) / prev_price > 0.02  # 2% üzeri hareket
                        
                        # Sinyal gücünü belirleme
                        confirmations = sum([volume_explosion, rsi_support, price_momentum])
                        if confirmations >= 2:
                            bollinger_breakout_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            bollinger_breakout_strength = "Güçlü"
                        else:
                            bollinger_breakout_strength = "Orta"
                
                # Higher High + Higher Low Pattern Sinyali
                hh_hl_signal = False
                hh_hl_strength = "Zayıf"
                
                if len(df) >= 10:
                    # Son 8 mum için yüksek ve alçak değerler
                    recent_highs = df['High'].tail(8)
                    recent_lows = df['Low'].tail(8)
                    
                    # Higher High kontrolü (son 4 mum vs önceki 4 mum)
                    first_half_high = recent_highs.iloc[:4].max()
                    second_half_high = recent_highs.iloc[4:].max()
                    higher_high = second_half_high > first_half_high
                    
                    # Higher Low kontrolü
                    first_half_low = recent_lows.iloc[:4].min()
                    second_half_low = recent_lows.iloc[4:].min()
                    higher_low = second_half_low > first_half_low
                    
                    if higher_high and higher_low:
                        hh_hl_signal = True
                        
                        # Trend gücü onayları
                        current_volume = df['Volume'].iloc[-1]
                        avg_volume = df['Volume'].tail(10).mean()
                        volume_support = current_volume > avg_volume
                        
                        # RSI trend onayı
                        rsi_trend = False
                        if 'rsi' in analyzer.indicators:
                            rsi_current = analyzer.indicators['rsi'].iloc[-1]
                            rsi_prev = analyzer.indicators['rsi'].iloc[-3]
                            rsi_trend = rsi_current > rsi_prev and rsi_current > 50
                        
                        # Fiyat momentum onayı
                        price_momentum = df['Close'].iloc[-1] > df['Close'].iloc[-4]
                        
                        # Sinyal gücü
                        confirmations = sum([volume_support, rsi_trend, price_momentum])
                        if confirmations >= 2:
                            hh_hl_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            hh_hl_strength = "Güçlü"
                        else:
                            hh_hl_strength = "Orta"
                
                # VWAP Altında Açılır, Üstünde Kapanır Sinyali
                vwap_reversal_signal = False
                vwap_reversal_strength = "Zayıf"
                
                if 'vwap' in analyzer.indicators and len(df) >= 5:
                    vwap_current = analyzer.indicators['vwap'].iloc[-1]
                    open_price = df['Open'].iloc[-1]
                    close_price = df['Close'].iloc[-1]
                    
                    # Altında açılıp üstünde kapanma kontrolü
                    if open_price < vwap_current and close_price > vwap_current:
                        vwap_reversal_signal = True
                        
                        # Hacim ve momentum onayları
                        current_volume = df['Volume'].iloc[-1]
                        avg_volume = df['Volume'].tail(20).mean()
                        volume_confirm = current_volume > (avg_volume * 1.3)
                        
                        # Gün içi performans (kapanış açılıştan ne kadar yüksek)
                        daily_performance = (close_price - open_price) / open_price
                        performance_strong = daily_performance > 0.02  # 2% üzeri
                        
                        # RSI momentum
                        rsi_momentum = False
                        if 'rsi' in analyzer.indicators:
                            rsi_value = analyzer.indicators['rsi'].iloc[-1]
                            rsi_momentum = rsi_value > 55
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, performance_strong, rsi_momentum])
                        if confirmations >= 2:
                            vwap_reversal_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            vwap_reversal_strength = "Güçlü"
                        else:
                            vwap_reversal_strength = "Orta"
                
                # ADX > 25 + DI+ > DI− Sinyali
                adx_trend_signal = False
                adx_trend_strength = "Zayıf"
                
                # ADX hesaplama (basit yaklaşım)
                if len(df) >= 14:
                    # True Range hesaplama
                    tr1 = df['High'] - df['Low']
                    tr2 = abs(df['High'] - df['Close'].shift(1))
                    tr3 = abs(df['Low'] - df['Close'].shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Directional Movement hesaplama
                    dm_plus = df['High'].diff()
                    dm_minus = df['Low'].diff() * -1
                    
                    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
                    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
                    
                    # 14 günlük ortalamalar
                    atr = true_range.rolling(window=14).mean()
                    di_plus = (dm_plus.rolling(window=14).mean() / atr) * 100
                    di_minus = (dm_minus.rolling(window=14).mean() / atr) * 100
                    
                    # ADX hesaplama
                    dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
                    adx = dx.rolling(window=14).mean()
                    
                    if not pd.isna(adx.iloc[-1]) and not pd.isna(di_plus.iloc[-1]):
                        current_adx = adx.iloc[-1]
                        current_di_plus = di_plus.iloc[-1]
                        current_di_minus = di_minus.iloc[-1]
                        
                        if current_adx > 25 and current_di_plus > current_di_minus:
                            adx_trend_signal = True
                            
                            # Trend gücü onayları
                            trend_strength = current_adx > 30  # Çok güçlü trend
                            di_gap = (current_di_plus - current_di_minus) > 5  # DI+ DI- farkı
                            
                            # Hacim onayı
                            volume_trend = df['Volume'].iloc[-1] > df['Volume'].tail(10).mean()
                            
                            # Sinyal gücü
                            confirmations = sum([trend_strength, di_gap, volume_trend])
                            if confirmations >= 2:
                                adx_trend_strength = "Çok Güçlü"
                            elif confirmations == 1:
                                adx_trend_strength = "Güçlü"
                            else:
                                adx_trend_strength = "Orta"
                
                # Volume Spike + Yatay Direnç Kırılımı Sinyali
                volume_breakout_signal = False
                volume_breakout_strength = "Zayıf"
                
                if len(df) >= 20:
                    # Son 10 mumda yatay direnç seviyesi bulma
                    recent_highs = df['High'].tail(10)
                    resistance_level = recent_highs.quantile(0.8)  # En yüksek %20'lik dilim
                    
                    current_price = df['Close'].iloc[-1]
                    current_volume = df['Volume'].iloc[-1]
                    avg_volume = df['Volume'].tail(20).mean()
                    
                    # Direnç kırılımı ve hacim patlaması
                    resistance_break = current_price > resistance_level
                    volume_spike = current_volume > (avg_volume * 2.0)  # 2x hacim artışı
                    
                    if resistance_break and volume_spike:
                        volume_breakout_signal = True
                        
                        # Kırılım gücü onayları
                        breakout_strength = (current_price - resistance_level) / resistance_level > 0.01  # %1 üzeri kırılım
                        
                        # RSI momentum onayı
                        rsi_strong = False
                        if 'rsi' in analyzer.indicators:
                            rsi_value = analyzer.indicators['rsi'].iloc[-1]
                            rsi_strong = 50 < rsi_value < 80
                        
                        # Trend onayı
                        trend_confirm = df['Close'].iloc[-1] > df['Close'].iloc[-5]
                        
                        # Sinyal gücü
                        confirmations = sum([breakout_strength, rsi_strong, trend_confirm])
                        if confirmations >= 2:
                            volume_breakout_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            volume_breakout_strength = "Güçlü"
                        else:
                            volume_breakout_strength = "Orta"
                
                # Gap Up + İlk 30 Dakika Güçlü Kapanış Sinyali
                gap_up_signal = False
                gap_up_strength = "Zayıf"
                
                if len(df) >= 2:
                    prev_close = df['Close'].iloc[-2]
                    current_open = df['Open'].iloc[-1]
                    current_close = df['Close'].iloc[-1]
                    current_volume = df['Volume'].iloc[-1]
                    
                    # Gap up kontrolü (%1 üzeri)
                    gap_percent = (current_open - prev_close) / prev_close
                    gap_up = gap_percent > 0.01
                    
                    # Güçlü kapanış (açılıştan %2 üzeri)
                    strong_close = (current_close - current_open) / current_open > 0.02
                    
                    if gap_up and strong_close:
                        gap_up_signal = True
                        
                        # Hacim onayı
                        avg_volume = df['Volume'].tail(10).mean()
                        volume_confirm = current_volume > (avg_volume * 1.5)
                        
                        # Gap büyüklüğü
                        big_gap = gap_percent > 0.03  # %3 üzeri gap
                        
                        # RSI momentum
                        rsi_momentum = False
                        if 'rsi' in analyzer.indicators:
                            rsi_value = analyzer.indicators['rsi'].iloc[-1]
                            rsi_momentum = rsi_value > 60
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, big_gap, rsi_momentum])
                        if confirmations >= 2:
                            gap_up_strength = "Çok Güçlü"
                        elif confirmations == 1:
                            gap_up_strength = "Güçlü"
                        else:
                            gap_up_strength = "Orta"
                
                # Sinyal kartları - 3 sıra, 4 sütunlu layout
                st.markdown("""
                <div style='border: 1px solid hsl(215, 28%, 20%); border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; background: hsl(220, 100%, 6%);'>
                    <h3 style='color: hsl(210, 40%, 98%); margin: 0; margin-bottom: 1rem;'>🐂 Boğa Sinyalleri</h3>
                """, unsafe_allow_html=True)
                
                # İlk sıra - Ana sinyaller
                signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
                
                # Ana sinyal
                with signal_col1:
                    if signal == "AL":
                        st.markdown("""
                        <div class="signal-card buy">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Güçlü Alış Sinyali</div>
                                <div class="tooltip-description">Birden fazla teknik indikatör aynı anda pozitif sinyal veriyor.</div>
                                <div class="tooltip-criteria">
                                    <strong>Kriterler:</strong><br>
                                    • RSI > 70 (aşırı alım değil)<br>
                                    • MACD pozitif crossover<br>
                                    • SuperTrend AL sinyali<br>
                                    • Hacim artışı var
                                </div>
                            </div>
                            <div class="signal-icon">🐂</div>
                            <div class="signal-text">Güçlü Alış Sinyali</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif signal == "SAT":
                        st.markdown("""
                        <div class="signal-card sell">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Güçlü Satış Sinyali</div>
                                <div class="tooltip-description">Birden fazla teknik indikatör aynı anda negatif sinyal veriyor.</div>
                                <div class="tooltip-criteria">
                                    <strong>Kriterler:</strong><br>
                                    • RSI < 30 (aşırı satım)<br>
                                    • MACD negatif crossover<br>
                                    • SuperTrend SAT sinyali<br>
                                    • Hacim artışı var
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">Güçlü Satış Sinyali</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card hold">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Pozisyon Tut</div>
                                <div class="tooltip-description">Mevcut durumda net bir alış/satış sinyali yok.</div>
                                <div class="tooltip-criteria">
                                    <strong>Durum:</strong><br>
                                    • İndikatörler karışık sinyal veriyor<br>
                                    • Trend belirsiz<br>
                                    • Hacim yetersiz<br>
                                    • Bekleme modunda kalın
                                </div>
                            </div>
                            <div class="signal-icon">⏳</div>
                            <div class="signal-text">Pozisyon Tut</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # VWAP Boğa Sinyali
                with signal_col2:
                    if vwap_bull_signal:
                        signal_class = "buy" if vwap_signal_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">VWAP Boğa Sinyali</div>
                                <div class="tooltip-description">Fiyat VWAP'ın altından başlayıp yukarı kesmesi. Güçlü momentum sinyali.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • Önceki mum VWAP altında<br>
                                    • Mevcut fiyat VWAP üstünde<br>
                                    • %20+ hacim artışı<br>
                                    • RSI > 50 + MACD yukarı trend
                                </div>
                            </div>
                            <div class="signal-icon">🐂</div>
                            <div class="signal-text">VWAP Boğa Sinyali</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{vwap_signal_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">VWAP Sinyali Bekleniyor</div>
                                <div class="tooltip-description">Fiyat henüz VWAP crossover yapmadı.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • Fiyatın VWAP altına düşmesi<br>
                                    • Sonra VWAP üzerine çıkması<br>
                                    • Hacim artışı ile desteklenmesi<br>
                                    • RSI ve MACD onayı
                                </div>
                            </div>
                            <div class="signal-icon">📊</div>
                            <div class="signal-text">VWAP Sinyali Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Bekleme Modunda</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Golden Cross Boğa Sinyali
                with signal_col3:
                    if golden_cross_signal:
                        signal_class = "buy" if golden_cross_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Golden Cross</div>
                                <div class="tooltip-description">EMA21'in EMA50'yi yukarı kesmesi. Klasik güçlü alış sinyali.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • EMA21 > EMA50 crossover<br>
                                    • %30+ hacim artışı<br>
                                    • RSI > 55<br>
                                    • MACD > 0 (pozitif bölge)
                                </div>
                            </div>
                            <div class="signal-icon">🥇</div>
                            <div class="signal-text">Golden Cross</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{golden_cross_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Golden Cross Bekleniyor</div>
                                <div class="tooltip-description">EMA21 henüz EMA50'nin altında.</div>
                                <div class="tooltip-criteria">
                                    <strong>Mevcut Durum:</strong><br>
                                    • EMA21 < EMA50<br>
                                    • Kısa vadeli ortalama düşük<br>
                                    • Yukarı momentum bekleniyor<br>
                                    • Crossover için izlenmeli
                                </div>
                            </div>
                            <div class="signal-icon">📈</div>
                            <div class="signal-text">Golden Cross Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">EMA21 < EMA50</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # MACD Boğa Sinyali
                with signal_col4:
                    if macd_bull_signal:
                        signal_class = "buy" if macd_signal_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">MACD Boğa Sinyali</div>
                                <div class="tooltip-description">MACD çizgisinin sinyal çizgisini yukarı kesmesi. Momentum değişimi.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • MACD > Signal Line crossover<br>
                                    • %25+ hacim artışı<br>
                                    • RSI > 45<br>
                                    • Fiyat yukarı trend
                                </div>
                            </div>
                            <div class="signal-icon">📊</div>
                            <div class="signal-text">MACD Boğa Sinyali</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{macd_signal_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">MACD Crossover Bekleniyor</div>
                                <div class="tooltip-description">MACD henüz sinyal çizgisini yukarı kesmedi.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • MACD çizgisinin yukarı hareketi<br>
                                    • Signal line'ı geçmesi<br>
                                    • Hacim artışı ile onaylanması<br>
                                    • Pozitif momentum değişimi
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">MACD Sinyali Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Crossover Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # İkinci sıra - Ek sinyaller
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                signal_col5, signal_col6, signal_col7, signal_col8 = st.columns(4)
                
                # RSI Toparlanma Sinyali
                with signal_col5:
                    if rsi_recovery_signal:
                        signal_class = "buy" if rsi_recovery_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">RSI Toparlanma Sinyali</div>
                                <div class="tooltip-description">RSI aşırı satım bölgesinden (30 altı) toparlanıp 40 üzerine çıkması.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • RSI 30 altından 40 üzerine<br>• Hacim artışı var<br>• Fiyat momentum pozitif<br>• MACD yukarı trend
                                </div>
                            </div>
                            <div class="signal-icon">📈</div>
                            <div class="signal-text">RSI Toparlanma</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{rsi_recovery_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">RSI Toparlanma Bekleniyor</div>
                                <div class="tooltip-description">RSI henüz oversold seviyesine gelmedi veya toparlanma başlamadı.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • RSI 30 altına düşmeli<br>• Sonra 40 üzerine çıkmalı<br>• Hacim artışı beklendir<br>• Momentum değişimi aranır
                                </div>
                            </div>
                            <div class="signal-icon">⚡</div>
                            <div class="signal-text">RSI Toparlanma Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Oversold Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Bollinger Sıkışma Sinyali
                with signal_col6:
                    if bollinger_breakout_signal:
                        signal_class = "buy" if bollinger_breakout_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Bollinger Kırılımı</div>
                                <div class="tooltip-description">Bollinger bantlarının sıkışmasından sonra üst banda kırılım.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • Fiyat üst banda kırılım<br>• %50+ hacim patlaması<br>• RSI 50-80 arası<br>• %2+ fiyat hareketi
                                </div>
                            </div>
                            <div class="signal-icon">🎯</div>
                            <div class="signal-text">Bollinger Kırılımı</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{bollinger_breakout_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Bollinger Kırılımı Bekleniyor</div>
                                <div class="tooltip-description">Bantlar henüz sıkışmadı veya kırılım gerçekleşmedi.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • Bantların sıkışması<br>• Üst banda yaklaşım<br>• Hacim artışı bekleniyor<br>• Volatilite patlaması
                                </div>
                            </div>
                            <div class="signal-icon">🔒</div>
                            <div class="signal-text">Bollinger Sıkışma Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Kırılım Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Higher High + Higher Low Pattern Sinyali
                with signal_col7:
                    if hh_hl_signal:
                        signal_class = "buy" if hh_hl_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Higher High + Higher Low</div>
                                <div class="tooltip-description">Son 8 mumda hem daha yüksek tepe hem daha yüksek dip. Sağlıklı yükseliş trendi.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • Daha yüksek tepe formasyonu<br>• Daha yüksek dip formasyonu<br>• Hacim desteği<br>• RSI trend onayı
                                </div>
                            </div>
                            <div class="signal-icon">📈</div>
                            <div class="signal-text">Higher High + Higher Low Pattern</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{hh_hl_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">HH+HL Pattern Bekleniyor</div>
                                <div class="tooltip-description">Henüz sağlıklı yükseliş trend formasyonu oluşmadı.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • Düşük seviyelerden yükseliş<br>• Ardışık yüksek tepeler<br>• Ardışık yüksek dipler<br>• Trend devamlılığı
                                </div>
                            </div>
                            <div class="signal-icon">📈</div>
                            <div class="signal-text">Higher High + Higher Low Pattern Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Trend Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # VWAP Altında Açılır, Üstünde Kapanır Sinyali
                with signal_col8:
                    if vwap_reversal_signal:
                        signal_class = "buy" if vwap_reversal_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">VWAP Reversal</div>
                                <div class="tooltip-description">Gün VWAP altında açılıp üstünde kapanma. Day-trade momentum sinyali.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • VWAP altında açılış<br>• VWAP üstünde kapanış<br>• %30+ hacim artışı<br>• %2+ günlük performans
                                </div>
                            </div>
                            <div class="signal-icon">🔄</div>
                            <div class="signal-text">VWAP Reversal</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{vwap_reversal_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">VWAP Reversal Bekleniyor</div>
                                <div class="tooltip-description">Henüz VWAP reversal pattern oluşmadı.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • VWAP altında açılış<br>• Gün içi toparlanma<br>• VWAP üstünde kapanış<br>• Güçlü hacim desteği
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">VWAP Reversal Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Düşüş Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Üçüncü sıra - Gelişmiş sinyaller
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                signal_col9, signal_col10, signal_col11, signal_col12 = st.columns(4)
                
                # ADX > 25 + DI+ > DI− Sinyali
                with signal_col9:
                    if adx_trend_signal:
                        signal_class = "buy" if adx_trend_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">ADX Trend Sinyali</div>
                                <div class="tooltip-description">ADX > 25 ve DI+ > DI-. Güçlü yukarı trend doğrulaması.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • ADX > 25 (güçlü trend)<br>• DI+ > DI- (yukarı yön)<br>• ADX > 30 bonus<br>• Hacim desteği
                                </div>
                            </div>
                            <div class="signal-icon">📈</div>
                            <div class="signal-text">ADX Trend</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{adx_trend_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">ADX Trend Bekleniyor</div>
                                <div class="tooltip-description">Trend gücü yetersiz veya yön belirsiz.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • ADX 25 üzerine çıkmalı<br>• DI+ DI-'yi geçmeli<br>• Trend gücü artmalı<br>• Yön netleşmeli
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">ADX Trend Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Trend Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Volume Spike + Yatay Direnç Kırılımı Sinyali
                with signal_col10:
                    if volume_breakout_signal:
                        signal_class = "buy" if volume_breakout_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Volume Breakout</div>
                                <div class="tooltip-description">2x hacim patlaması ile yatay direnç kırılımı. Güçlü momentum sinyali.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • Yatay direnç kırılımı<br>• 2x hacim patlaması<br>• %1+ kırılım gücü<br>• RSI 50-80 arası
                                </div>
                            </div>
                            <div class="signal-icon">💥</div>
                            <div class="signal-text">Volume Breakout</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{volume_breakout_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Volume Breakout Bekleniyor</div>
                                <div class="tooltip-description">Henüz hacimli direnç kırılımı gerçekleşmedi.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • Yatay direnç seviyesi<br>• Hacim birikimi<br>• Kırılım hazırlığı<br>• Momentum beklentisi
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">Volume Breakout Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Yatay Direnç Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Gap Up + İlk 30 Dakika Güçlü Kapanış Sinyali
                with signal_col11:
                    if gap_up_signal:
                        signal_class = "buy" if gap_up_strength in ["Güçlü", "Çok Güçlü"] else "hold"
                        st.markdown(f"""
                        <div class="signal-card {signal_class}">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Gap Up Sinyali</div>
                                <div class="tooltip-description">%1+ gap açılış ve %2+ güçlü kapanış. Kurumsal talep işareti.</div>
                                <div class="tooltip-criteria">
                                    <strong>Koşullar:</strong><br>
                                    • %1+ gap açılış<br>• %2+ güçlü kapanış<br>• %50+ hacim artışı<br>• RSI > 60
                                </div>
                            </div>
                            <div class="signal-icon">⬆️</div>
                            <div class="signal-text">Gap Up</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{gap_up_strength}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="signal-card neutral">
                            <div class="signal-info-icon">i</div>
                            <div class="signal-tooltip">
                                <div class="tooltip-title">Gap Up Bekleniyor</div>
                                <div class="tooltip-description">Henüz gap açılış veya güçlü performans yok.</div>
                                <div class="tooltip-criteria">
                                    <strong>Beklenen:</strong><br>
                                    • Pozitif gap açılış<br>• Güçlü gün içi performans<br>• Hacim patlaması<br>• Momentum devamlılığı
                                </div>
                            </div>
                            <div class="signal-icon">📉</div>
                            <div class="signal-text">Gap Up Yok</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">Yükseliş Bekleniyor</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("""
                </div>
                """, unsafe_allow_html=True)
                
                # Ayı Sinyalleri - Modern ve Kompakt Tasarım
                st.markdown("""
                <div style='
                    border: 1px solid hsl(215, 28%, 20%); 
                    border-radius: 0.75rem; 
                    padding: 1.5rem; 
                    margin: 1.5rem 0; 
                    background: linear-gradient(135deg, hsl(220, 100%, 6%) 0%, hsl(215, 40%, 10%) 100%);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                '>
                    <div style='
                        display: flex;
                        align-items: center;
                        margin-bottom: 1rem;
                        padding-bottom: 0.75rem;
                        border-bottom: 1px solid hsl(215, 28%, 20%);
                    '>
                        <span style='
                            font-size: 1.5rem;
                            margin-right: 0.75rem;
                        '>🐻</span>
                        <h3 style='
                            color: hsl(210, 40%, 98%); 
                            margin: 0; 
                            font-size: 1.25rem;
                            font-weight: 600;
                        '>Ayı Sinyalleri</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Kompakt Bear Signal Layout
                bear_col1, bear_col2, bear_col3 = st.columns([1, 1, 1], gap="medium")
                
                with bear_col1:
                    # Ana Bear Signal Kartı
                    st.markdown(f"""
                    <div style='
                        background: hsl(220, 45%, 12%);
                        border: 1px solid hsl(215, 35%, 18%);
                        border-radius: 0.5rem;
                        padding: 1rem;
                        text-align: center;
                    '>
                        <div style='
                            color: hsl(210, 40%, 98%);
                            font-size: 0.875rem;
                            font-weight: 500;
                            margin-bottom: 0.5rem;
                        '>🐻 Ayı Sinyali</div>
                        <div style='
                            color: {'hsl(142, 76%, 36%)' if bear_signal['strength'] < 5 else 'hsl(0, 84%, 60%)'};
                            font-size: 1.25rem;
                            font-weight: 700;
                            margin-bottom: 0.25rem;
                        '>{bear_signal['strength_level']}</div>
                        <div style='
                            color: hsl(215, 20%, 70%);
                            font-size: 0.75rem;
                        '>{bear_signal['signal_count']} Sinyal Aktif</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with bear_col2:
                    # Güç Skoru
                    progress_value = min(bear_signal['strength'] / 10, 1.0)
                    progress_color = "hsl(142, 76%, 36%)" if bear_signal['strength'] < 5 else "hsl(0, 84%, 60%)"
                    
                    st.markdown(f"""
                    <div style='
                        background: hsl(220, 45%, 12%);
                        border: 1px solid hsl(215, 35%, 18%);
                        border-radius: 0.5rem;
                        padding: 1rem;
                    '>
                        <div style='
                            color: hsl(210, 40%, 98%);
                            font-size: 0.875rem;
                            font-weight: 500;
                            margin-bottom: 0.5rem;
                        '>💪 Güç Skoru</div>
                        <div style='
                            background: hsl(215, 35%, 18%);
                            border-radius: 0.25rem;
                            height: 0.5rem;
                            margin-bottom: 0.5rem;
                            overflow: hidden;
                        '>
                            <div style='
                                background: {progress_color};
                                height: 100%;
                                width: {progress_value * 100}%;
                                transition: width 0.3s ease;
                            '></div>
                        </div>
                        <div style='
                            color: {progress_color};
                            font-size: 0.875rem;
                            font-weight: 600;
                        '>{bear_signal['strength']:.1f}/10</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with bear_col3:
                    # Durum Özeti
                    status_icon = "✅" if not bear_signal['signals'] else "🚨"
                    status_text = "Güvenli" if not bear_signal['signals'] else "Dikkat"
                    status_color = "hsl(142, 76%, 36%)" if not bear_signal['signals'] else "hsl(0, 84%, 60%)"
                    
                    st.markdown(f"""
                    <div style='
                        background: hsl(220, 45%, 12%);
                        border: 1px solid hsl(215, 35%, 18%);
                        border-radius: 0.5rem;
                        padding: 1rem;
                        text-align: center;
                    '>
                        <div style='
                            font-size: 1.5rem;
                            margin-bottom: 0.5rem;
                        '>{status_icon}</div>
                        <div style='
                            color: {status_color};
                            font-size: 0.875rem;
                            font-weight: 600;
                        '>{status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detaylı bilgiler (expandable)
                if bear_signal['details']:
                    with st.expander("📊 Detaylı Sinyal Analizi", expanded=False):
                        for detail in bear_signal['details']:
                            st.markdown(f"• {detail}")
                else:
                    st.markdown("""
                    <div style='
                        background: hsl(220, 45%, 12%);
                        border: 1px solid hsl(215, 35%, 18%);
                        border-radius: 0.5rem;
                        padding: 1rem;
                        text-align: center;
                        margin-top: 1rem;
                    '>
                        <div style='
                            color: hsl(142, 76%, 36%);
                            font-size: 0.875rem;
                            font-weight: 500;
                        '>✅ Ayı Sinyali Tespit Edilmedi</div>
                        <div style='
                            color: hsl(215, 20%, 70%);
                            font-size: 0.75rem;
                            margin-top: 0.25rem;
                        '>Mevcut durumda güçlü düşüş sinyali bulunmuyor</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Risk Analizi & Pozisyon Önerileri bölümü kaldırıldı
                
        else:
                    st.markdown("""
                    <div style='
                        background: hsl(220, 45%, 12%);
                        border: 1px solid hsl(0, 84%, 60%);
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        margin: 1rem 0;
                        text-align: center;
                    '>
                        <div style='
                            color: hsl(0, 84%, 60%);
                            font-size: 1.5rem;
                            margin-bottom: 0.5rem;
                        '>⚠️</div>
                        <h4 style='
                            color: hsl(210, 40%, 98%);
                            margin: 0 0 0.5rem 0;
                            font-size: 1.1rem;
                        '>Veri Yüklenemedi</h4>
                        <p style='
                            color: hsl(215, 20%, 70%);
                            margin: 0;
                            font-size: 0.9rem;
                        '>Seçilen hisse için veri bulunamadı. Lütfen farklı bir hisse seçin veya daha sonra tekrar deneyin.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Hata</h4>
            <p>{str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

def scan_daytrading_opportunities():
    """Day trading fırsatlarını tarar ve puanlar"""
    opportunities = []
    fetcher = BISTDataFetcher()
    
    # Daha fazla hisse tara (BIST 100)
    sample_symbols = list(BIST_SYMBOLS.keys())[:50]  # İlk 50 hisse (performans dengeli)
    
    for symbol in sample_symbols:
        try:
            # Günlük veri çek (son 30 gün)
            df = fetcher.get_stock_data(symbol, period="30d", interval="1d")
            if df is None or len(df) < 20:
                continue
                
            # Teknik analiz
            analyzer = TechnicalAnalyzer(df)
            analyzer.add_indicator('rsi')
            analyzer.add_indicator('ema_21')
            analyzer.add_indicator('macd')
            
            # Güncel değerler
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            current_price = latest['Close']
            
            # Kriterleri hesapla
            # 1. Volatilite (günlük aralık %)
            daily_range = ((latest['High'] - latest['Low']) / latest['Low']) * 100
            
            # 2. Hacim oranı (son hacim / 20 günlük ortalama)
            avg_volume = df['Volume'].tail(20).mean()
            volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
            
            # 3. RSI değeri
            rsi = analyzer.indicators['rsi'].iloc[-1] if 'rsi' in analyzer.indicators else 50
            
            # 4. MACD durumu
            macd_line = analyzer.indicators['macd'].iloc[-1] if 'macd' in analyzer.indicators else 0
            macd_signal = analyzer.indicators['macd_signal'].iloc[-1] if 'macd_signal' in analyzer.indicators else 0
            macd_bullish = macd_line > macd_signal
            
            # 5. EMA durumu
            ema_21 = analyzer.indicators['ema_21'].iloc[-1] if 'ema_21' in analyzer.indicators else current_price
            price_above_ema = current_price > ema_21
            
            # 6. Momentum (son 3 günlük değişim)
            three_day_change = ((current_price - df['Close'].iloc[-4]) / df['Close'].iloc[-4]) * 100 if len(df) >= 4 else 0
            
            # Puanlama sistemi (1-10)
            score = 0
            reasons = []
            
            # Volatilite puanı (2-5% arası ideal day trade için)
            if 2 <= daily_range <= 5:
                score += 2.5
                reasons.append("İyi volatilite")
            elif 1.5 <= daily_range < 2 or 5 < daily_range <= 7:
                score += 1.5
                reasons.append("Orta volatilite")
            elif daily_range > 7:
                score += 1
                reasons.append("Yüksek volatilite")
            
            # Hacim puanı
            if volume_ratio >= 2.0:
                score += 2
                reasons.append("Yüksek hacim")
            elif volume_ratio >= 1.5:
                score += 1.5
                reasons.append("Artan hacim")
            elif volume_ratio >= 1.2:
                score += 1
                reasons.append("Normal hacim")
            
            # RSI puanı (aşırı bölgelerde fırsat)
            if rsi <= 30:
                score += 2
                reasons.append("RSI aşırı satım")
            elif rsi >= 70:
                score += 2
                reasons.append("RSI aşırı alım")
            elif 40 <= rsi <= 60:
                score += 1
                reasons.append("RSI nötr")
            
                # MACD puanı
                if macd_bullish and macd_line > 0:
                    score += 1.5
                    reasons.append("MACD pozitif")
                elif macd_bullish:
                    score += 1
                    reasons.append("MACD yukarı")
                
                # Trend puanı
                if price_above_ema:
                    score += 1
                    reasons.append("EMA üstünde")
                
                # Momentum puanı
                if abs(three_day_change) >= 3:
                    score += 1
                    reasons.append("Güçlü momentum")
                elif abs(three_day_change) >= 1.5:
                    score += 0.5
                    reasons.append("Momentum var")
                
                # Sinyal belirleme
                signal = "BEKLE"
                if rsi <= 35 and macd_bullish and volume_ratio >= 1.5:
                    signal = "AL"
                elif rsi >= 65 and not macd_bullish and volume_ratio >= 1.5:
                    signal = "SAT"
                elif price_above_ema and macd_bullish and volume_ratio >= 1.3:
                    signal = "AL"
                
                # Minimum puan kontrolü
                if score >= 4:  # En az 4 puan alan hisseleri dahil et
                    opportunity = {
                        'symbol': symbol.replace('.IS', ''),
                        'name': BIST_SYMBOLS[symbol],
                        'price': current_price,
                        'signal': signal,
                        'score': round(score, 1),
                        'volatility': daily_range,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi,
                        'macd_bullish': macd_bullish,
                        'three_day_change': three_day_change,
                        'reason': ", ".join(reasons[:3])  # İlk 3 sebep
                    }
                    opportunities.append(opportunity)
                    
        except Exception as e:
                # Hata durumunda geç, diğer hisseleri kontrol et
                continue
    
    # Puana göre sırala
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities

def show_modern_dashboard():
    """Modern SaaS Dashboard - Ekran görüntüsü stilinde"""
    
    # Dashboard Header with Stock Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Stock Selection and Time Interval
        st.markdown("""
        <div style="margin-top: 20px;">
        </div>
        """, unsafe_allow_html=True)
        
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            selected_symbol = st.selectbox(
                "📊 Hisse",
            options=sorted(list(BIST_SYMBOLS.keys())),
                format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
                key="dashboard_stock_select"
            )
        
        with subcol2:
            time_interval = st.selectbox(
                "⏰ Zaman Aralığı",
                options=["5m", "15m", "1h", "2h", "4h", "1d"],
                index=5,  # default to 1d
                key="dashboard_time_interval"
            )
    
    # Tab Navigation
    st.markdown("""
    <div class="tab-navigation">
        <div class="tab-item active">📊 Overview</div>
        <div class="tab-item">📈 Analytics</div>
        <div class="tab-item">📄 Reports</div>
        <div class="tab-item">🔔 Notifications</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    try:
        fetcher = BISTDataFetcher()
        # Adjust period based on interval respecting Yahoo Finance API limits
        if time_interval in ["5m", "15m"]:
            period = "60d"  # 60 days max for short intervals (Yahoo Finance limit)
        elif time_interval in ["1h", "2h"]:
            period = "3mo"  # 3 months for hourly intervals
        elif time_interval == "4h":
            period = "6mo"  # 6 months for 4-hour intervals
        else:
            period = "1y"   # 1 year for daily intervals
            
        df = fetcher.get_stock_data(selected_symbol, period=period, interval=time_interval)
    except Exception as e:
        st.error(f"Veri alınırken bir hata oluştu: {e}")
        df = None
    
    if df is not None and not df.empty:
        # Calculate metrics
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = latest['Close'] - prev['Close']
        change_pct = (change / prev['Close']) * 100
        volume_change = ((latest['Volume'] - df['Volume'].tail(20).mean()) / df['Volume'].tail(20).mean()) * 100
        
        # Weekly/Monthly changes
        week_ago = df.iloc[-7] if len(df) > 7 else prev
        month_ago = df.iloc[-22] if len(df) > 22 else prev
        week_change = ((latest['Close'] - week_ago['Close']) / week_ago['Close']) * 100
        month_change = ((latest['Close'] - month_ago['Close']) / month_ago['Close']) * 100
        
        # KPI Cards Grid
        st.markdown("""
        <div class="kpi-grid">
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Mevcut Fiyat - ilk sıraya taşındı
            price_class = "positive" if change > 0 else "negative"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <div class="kpi-title">
                        <span>💰</span> Fiyat Hareketi (Günlük)
                    </div>
                    <div class="kpi-trend">{'📈' if change > 0 else '📉'}</div>
                </div>
                <div class="kpi-value">₺{latest['Close']:.2f}</div>
                <div class="kpi-change {price_class}">
                    <span>{'↗' if change > 0 else '↘'}</span>
                    <span>{'+' if change > 0 else ''}{change_pct:.2f}% son kapanıştan</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Haftalık Performans
            week_trend_icon = "📈" if week_change > 0 else "📉"
            week_change_class = "positive" if week_change > 0 else "negative"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <div class="kpi-title">
                        <span>📊</span> Haftalık Performans
                    </div>
                    <div class="kpi-trend">{week_trend_icon}</div>
                </div>
                <div class="kpi-value">{abs(week_change):.1f}%</div>
                <div class="kpi-change {week_change_class}">
                    <span>{'↗' if week_change > 0 else '↘'}</span>
                    <span>{'+' if week_change > 0 else ''}{week_change:.2f}% son haftadan beri</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Aylık Performans - YENİ EKLENEN
            month_trend_icon = "📈" if month_change > 0 else "📉"
            month_change_class = "positive" if month_change > 0 else "negative"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <div class="kpi-title">
                        <span>📅</span> Aylık Performans
                    </div>
                    <div class="kpi-trend">{month_trend_icon}</div>
                </div>
                <div class="kpi-value">{abs(month_change):.1f}%</div>
                <div class="kpi-change {month_change_class}">
                    <span>{'↗' if month_change > 0 else '↘'}</span>
                    <span>{'+' if month_change > 0 else ''}{month_change:.2f}% son aydan beri</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Hacim Aktivitesi
            volume_class = "positive" if volume_change > 0 else "negative"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <div class="kpi-title">
                        <span>📊</span> Hacim Aktivitesi
                    </div>
                    <div class="kpi-trend">📊</div>
                </div>
                <div class="kpi-value">{latest['Volume']:,.0f}</div>
                <div class="kpi-change {volume_class}">
                    <span>{'↗' if volume_change > 0 else '↘'}</span>
                    <span>{'+' if volume_change > 0 else ''}{volume_change:.1f}% ortalamaya karşı</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === TEKNİK İNDİKATÖRLER BÖLÜMÜ ===
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate technical indicators
        try:
            from modules.technical_analysis import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer(df)
            
            # Add EMA, VWMA and VWEMA indicators
            analyzer.add_indicator('ema_5')
            analyzer.add_indicator('ema_8') 
            analyzer.add_indicator('ema_13')
            analyzer.add_indicator('ma_200')
            analyzer.add_indicator('vwma_5')
            analyzer.add_indicator('vwema_5')
            
            # Get latest values
            ema_5 = analyzer.indicators.get('ema_5', pd.Series()).iloc[-1] if not analyzer.indicators.get('ema_5', pd.Series()).empty else 0
            ema_8 = analyzer.indicators.get('ema_8', pd.Series()).iloc[-1] if not analyzer.indicators.get('ema_8', pd.Series()).empty else 0
            ema_13 = analyzer.indicators.get('ema_13', pd.Series()).iloc[-1] if not analyzer.indicators.get('ema_13', pd.Series()).empty else 0
            ma_200 = analyzer.indicators.get('ma_200', pd.Series()).iloc[-1] if not analyzer.indicators.get('ma_200', pd.Series()).empty else 0
            vwma_5 = analyzer.indicators.get('vwma_5', pd.Series()).iloc[-1] if not analyzer.indicators.get('vwma_5', pd.Series()).empty else 0
            vwema_5 = analyzer.indicators.get('vwema_5', pd.Series()).iloc[-1] if not analyzer.indicators.get('vwema_5', pd.Series()).empty else 0
            
            current_price = latest['Close']
            
            # EMA/MA Technical Indicators Grid
            st.markdown("""
            <div class="technical-indicators">
            </div>
            """, unsafe_allow_html=True)
            
            # Display indicators in 6 columns (including VWMA 5 and VWEMA 5)
            indicator_cols = st.columns(6)
            
            # EMA 5 Card
            with indicator_cols[0]:
                price_above_ema5 = current_price > ema_5
                status_color = "#00ff88" if price_above_ema5 else "#ff4757"
                status_icon = "🟢" if price_above_ema5 else "🔴"
                status_text = "Üzeri" if price_above_ema5 else "Altı"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} EMA 5</div>
                    <div class="metric-value">₺{ema_5:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # EMA 8 Card  
            with indicator_cols[1]:
                price_above_ema8 = current_price > ema_8
                status_color = "#00ff88" if price_above_ema8 else "#ff4757"
                status_icon = "🟢" if price_above_ema8 else "🔴"
                status_text = "Üzeri" if price_above_ema8 else "Altı"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} EMA 8</div>
                    <div class="metric-value">₺{ema_8:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # EMA 13 Card
            with indicator_cols[2]:
                price_above_ema13 = current_price > ema_13
                status_color = "#00ff88" if price_above_ema13 else "#ff4757"
                status_icon = "🟢" if price_above_ema13 else "🔴"
                status_text = "Üzeri" if price_above_ema13 else "Altı"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} EMA 13</div>
                    <div class="metric-value">₺{ema_13:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # MA 200 Card
            with indicator_cols[3]:
                price_above_ma200 = current_price > ma_200
                status_color = "#00ff88" if price_above_ma200 else "#ff4757"
                status_icon = "🟢" if price_above_ma200 else "🔴"
                status_text = "Üzeri" if price_above_ma200 else "Altı"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} MA 200</div>
                    <div class="metric-value">₺{ma_200:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # VWMA 5 Card
            with indicator_cols[4]:
                price_above_vwma5 = current_price > vwma_5
                status_color = "#00ff88" if price_above_vwma5 else "#ff4757"
                status_icon = "🟢" if price_above_vwma5 else "🔴"
                status_text = "Üzeri" if price_above_vwma5 else "Altı"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} VWMA 5</div>
                    <div class="metric-value">₺{vwma_5:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # VWEMA 5 Card (NEW)
            with indicator_cols[5]:
                price_above_vwema5 = current_price > vwema_5
                status_color = "#00ff88" if price_above_vwema5 else "#ff4757"
                status_icon = "🟢" if price_above_vwema5 else "🔴"
                status_text = "Üzeri" if price_above_vwema5 else "Altı"
                    
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{status_icon} VWEMA 5</div>
                    <div class="metric-value">₺{vwema_5:.2f}</div>
                    <div class="metric-change" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.warning(f"⚠️ Teknik indikatörler hesaplanamadı: {str(e)}")
        
        # Bottom Section
        st.markdown("""
        <div class="bottom-grid">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">Piyasa Analizi</div>
                <div class="info-card-content">
                Gelişmiş algoritmalarla desteklenen gerçek zamanlı teknik analiz.
                Piyasa trendleri ve işlem fırsatları hakkında bilgi edinin.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">Yapay Zeka Tahminleri</div>
                <div class="info-card-content">
                    Makine öğrenmesi modelleri, gelecekteki fiyat hareketlerini 
                    güven skorları ile tahmin etmek için geçmiş verileri analiz eder.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        
        # === HAFTALIK VE AYLIK PERFORMANS BÖLÜMÜ ===
        st.markdown("<br><br>", unsafe_allow_html=True)

    try:
        # Initialize screener and get performance data
        screener = StockScreener(BIST_SYMBOLS)
        
        # Load performance data (cache'de yoksa hesapla)
        if "performance_data_loaded_v9" not in st.session_state:
            with st.spinner("�� Performans verileri yükleniyor..."):
                weekly_results = screener.screen_weekly_performance(top_count=15)
                monthly_results = screener.screen_monthly_performance(top_count=15)
                st.session_state.weekly_results = weekly_results
                st.session_state.monthly_results = monthly_results
                st.session_state.performance_data_loaded_v9 = True
        
        # Weekly Performance
        weekly_data = st.session_state.weekly_results
        
        # Başlık ve yenileme butonu aynı satırda
        title_col, button_col = st.columns([3, 1])
        with title_col:
            st.markdown("### 📊 Haftalık & Aylık Performans")
        with button_col:
            st.markdown("<br>", unsafe_allow_html=True)  # Biraz boşluk için
            if st.button("🔄 Performans Verilerini Yenile", type="secondary", key="refresh_performance"):
                st.session_state.performance_data_loaded_v9 = False
                st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 En Çok Yükselenler (Haftalık)")
            if weekly_data["gainers"]:
                # Tablo için veri hazırla
                gainers_df = []
                for stock in weekly_data["gainers"][:10]:
                    gainers_df.append({
                        'Hisse': stock['symbol'],
                        'Değişim (%)': stock['weekly_change'],
                        'Fiyat (₺)': stock['current_price'],
                        'Hacim': stock['volume_ratio']
                    })
                
                # DataFrame oluştur ve renkli stil uygula
                df_gainers = pd.DataFrame(gainers_df)
                
                # Stil fonksiyonu - yeşil arkaplan
                def style_weekly_gainers(val):
                    if isinstance(val, (int, float)) and val > 0:
                        return 'background-color: #1f4e3d; color: #00ff88; font-weight: bold;'
                    return 'background-color: #1a202c; color: white;'
                
                styled_df = df_gainers.style.applymap(style_weekly_gainers, subset=['Değişim (%)']) \
                .format({
                    'Değişim (%)': '+{:.2f}%',
                    'Fiyat (₺)': '₺{:.2f}',
                    'Hacim': '{:.1f}x'
                }) \
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#2d3748')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Henüz haftalık yükselen hisse bulunamadı.")
        
        with col2:
            st.markdown("#### 📉 En Çok Düşenler (Haftalık)")
            if weekly_data["losers"]:
                # Tablo için veri hazırla
                losers_df = []
                for stock in weekly_data["losers"][:10]:
                    losers_df.append({
                        'Hisse': stock['symbol'],
                        'Değişim (%)': stock['weekly_change'],
                        'Fiyat (₺)': stock['current_price'],
                        'Hacim': stock['volume_ratio']
                    })
                
                # DataFrame oluştur ve renkli stil uygula
                df_losers = pd.DataFrame(losers_df)
                
                # Stil fonksiyonu - kırmızı arkaplan
                def style_weekly_losers(val):
                    if isinstance(val, (int, float)) and val < 0:
                        return 'background-color: #4a1e1e; color: #ff4757; font-weight: bold;'
                    return 'background-color: #1a202c; color: white;'
                
                styled_df = df_losers.style.applymap(style_weekly_losers, subset=['Değişim (%)']) \
                .format({
                    'Değişim (%)': '{:.2f}%',
                    'Fiyat (₺)': '₺{:.2f}',
                    'Hacim': '{:.1f}x'
                }) \
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#2d3748')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Henüz haftalık düşen hisse bulunamadı.")
        
        # Monthly Performance
        monthly_data = st.session_state.monthly_results
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 En Çok Yükselenler (Aylık)")
            if monthly_data["gainers"]:
                # Tablo için veri hazırla
                gainers_df = []
                for stock in monthly_data["gainers"][:10]:
                    gainers_df.append({
                        'Hisse': stock['symbol'],
                        'Değişim (%)': stock['monthly_change'],
                        'Fiyat (₺)': stock['current_price'],
                        'Volatilite (%)': stock['volatility'],
                        'Hacim': stock['volume_ratio']
                    })
                
                # DataFrame oluştur ve renkli stil uygula
                df_gainers = pd.DataFrame(gainers_df)
                
                # Stil fonksiyonu - yeşil arkaplan
                def style_monthly_gainers(val):
                    if isinstance(val, (int, float)) and val > 0:
                        return 'background-color: #1f4e3d; color: #00ff88; font-weight: bold;'
                    return 'background-color: #1a202c; color: white;'
                
                styled_df = df_gainers.style.applymap(style_monthly_gainers, subset=['Değişim (%)']) \
                .format({
                    'Değişim (%)': '+{:.2f}%',
                    'Fiyat (₺)': '₺{:.2f}',
                    'Volatilite (%)': '{:.1f}%',
                    'Hacim': '{:.1f}x'
                }) \
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#2d3748')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Henüz aylık yükselen hisse bulunamadı.")
        
        with col2:
            st.markdown("#### 📉 En Çok Düşenler (Aylık)")
            if monthly_data["losers"]:
                # Tablo için veri hazırla
                losers_df = []
                for stock in monthly_data["losers"][:10]:
                    losers_df.append({
                        'Hisse': stock['symbol'],
                        'Değişim (%)': stock['monthly_change'],
                        'Fiyat (₺)': stock['current_price'],
                        'Volatilite (%)': stock['volatility'],
                        'Hacim': stock['volume_ratio']
                    })
                
                # DataFrame oluştur ve renkli stil uygula
                df_losers = pd.DataFrame(losers_df)
                
                # Stil fonksiyonu - kırmızı arkaplan
                def style_monthly_losers(val):
                    if isinstance(val, (int, float)) and val < 0:
                        return 'background-color: #4a1e1e; color: #ff4757; font-weight: bold;'
                    return 'background-color: #1a202c; color: white;'
                
                styled_df = df_losers.style.applymap(style_monthly_losers, subset=['Değişim (%)']) \
                .format({
                    'Değişim (%)': '{:.2f}%',
                    'Fiyat (₺)': '₺{:.2f}',
                    'Volatilite (%)': '{:.1f}%',
                    'Hacim': '{:.1f}x'
                }) \
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2d3748'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#2d3748')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Henüz aylık düşen hisse bulunamadı.")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_ai_predictions():
    """AI tahminleri sayfası - Gelişmiş AI/ML Dashboard"""
    st.markdown("""
    <div class="page-header" style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="margin: 0;">🤖 AI Tahminleri</h1>
        <span style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">Çok modelli makine öğrenmesi ile gelişmiş fiyat tahmini ve analizi</span>
    </div>
    """, unsafe_allow_html=True)
    

    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "📈 Hisse Seç",
            options=sorted(list(BIST_SYMBOLS.keys())),
            format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
            key="ai_stock_select"
        )
    
    with col2:
        prediction_horizon = st.selectbox(
            "⏰ Tahmin Süresi",
            options=[3, 7, 14, 30],
            format_func=lambda x: {
                3: "3 gün sonra", 
                7: "1 hafta sonra",
                14: "2 hafta sonra",
                30: "1 ay sonra"
            }.get(x, f"{x} gün sonra"),
            index=0,
            key="prediction_horizon",
            help="AI modeli seçilen süre kadar sonrasını tahmin eder. Örneğin '1 hafta sonra' seçerseniz, modelin tahmini 7 gün sonraki fiyat için olacaktır."
        )
    
    with col3:
        model_type = st.selectbox(
            "🧠 Model Türü",
            options=["ensemble", "random_forest", "gradient_boosting", "all_models"],
            format_func=lambda x: {
                "ensemble": "🎯 Ensemble (En İyi)",
                "random_forest": "🌲 Rastgele Orman", 
                "gradient_boosting": "⚡ Gradyan Artırma",
                "all_models": "📊 Tüm Modeller"
            }[x],
            key="model_type"
        )
    
    # Prediction button
    predict_button = st.button("🚀 AI Tahminleri Oluştur", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("🧠 AI modelleri analiz ediyor... Bu biraz zaman alabilir"):
            try:
                # Veri çek
                fetcher = BISTDataFetcher()
                data = fetcher.get_stock_data(selected_symbol, period="2y", interval="1d")
                
                if data is None:
                    st.error(f"❌ {selected_symbol} için veri çekilemedi. Lütfen başka bir hisse deneyin.")
                    st.info("✅ Çalışan hisseler: THYAO.IS, GARAN.IS, ISCTR.IS")
                    return
                
                if len(data) < 100:
                    st.error(f"❌ Yetersiz veri: {len(data)} gün. AI tahmini için en az 100 gün gerekli.")
                    return
                
                # Veri kalitesi kontrolü
                if data.isnull().any().any():
                    st.warning("⚠️ Veride eksik değerler tespit edildi, temizleniyor...")
                    data = data.fillna(method='ffill').fillna(method='bfill')
                
                if (data <= 0).any().any():
                    st.warning("⚠️ Veride sıfır/negatif değerler tespit edildi, düzeltiliyor...")
                    # Volume sıfır olabilir, ama fiyatlar pozitif olmalı
                    for col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = data[col].where(data[col] > 0, data[col].rolling(3, min_periods=1).mean())
                
                st.success(f"✅ {selected_symbol} verisi hazır: {len(data)} gün")
                
                # Teknik analiz
                analyzer = TechnicalAnalyzer(data)
                indicators_to_add = ['rsi', 'ema_5', 'ema_8', 'ema_13', 'ema_21', 'vwap', 'bollinger', 'macd']
                
                successful_indicators = []
                failed_indicators = []
                
                for indicator in indicators_to_add:
                    try:
                        analyzer.add_indicator(indicator)
                        successful_indicators.append(indicator)
                    except Exception as e:
                        failed_indicators.append(f"{indicator}: {str(e)}")
                        st.warning(f"⚠️ {indicator} indikatörü eklenemedi: {str(e)}")
                
                if len(successful_indicators) < 3:
                    st.error("❌ Yeterli teknik indikatör hesaplanamadı. Veri kalitesi sorunu olabilir.")
                    return
                
                st.success(f"✅ Teknik analiz tamamlandı: {len(successful_indicators)} indikatör")
                
                # Gelişmiş ML tahmin modülü kullan
                from modules.ml_predictor import MLPredictor
                ml_predictor = MLPredictor()
                
                # Debug: Feature'ları kontrol et
                try:
                    test_features = ml_predictor.prepare_features(data, analyzer.indicators)
                    inf_count = np.isinf(test_features.values).sum()
                    nan_count = np.isnan(test_features.values).sum()
                    
                    if inf_count > 0 or nan_count > 0:
                        st.warning(f"⚠️ Özellik matrisinde sorunlar: {inf_count} sonsuz, {nan_count} NaN değer")
                        # Temizle
                        test_features = ml_predictor.clean_features(test_features)
                        st.info("✅ Özellik matrisi temizlendi")
                    
                except Exception as e:
                    st.error(f"❌ Özellik hazırlama hatası: {str(e)}")
                    return
                
                # Model eğit
                with st.status("🤖 Modeller eğitiliyor...", expanded=True) as status:
                    st.write("📊 Veri hazırlanıyor...")
                    training_results = ml_predictor.train_models(
                        data, 
                        analyzer.indicators, 
                        prediction_horizon=prediction_horizon
                    )
                    
                    if 'error' in training_results:
                        st.error(f"❌ Model eğitimi başarısız: {training_results['error']}")
                        if 'model_errors' in training_results:
                            with st.expander("🔍 Detaylı Hata Bilgileri"):
                                st.json(training_results['model_errors'])
                        return
                    
                    successful_models = training_results.get('successful_models', [])
                    if len(successful_models) == 0:
                        st.error("❌ Hiçbir model başarıyla eğitilemedi")
                        return
                    
                    st.write(f"✅ {len(successful_models)} model başarıyla eğitildi")
                    status.update(label="✅ Model eğitimi tamamlandı!", state="complete")
                
                # Tahmin yap
                predictions = ml_predictor.predict_price(
                    data, 
                    analyzer.indicators, 
                    prediction_horizon=prediction_horizon
                )
                
                if 'error' in predictions:
                    st.error(f"❌ Tahmin hesaplama hatası: {predictions['error']}")
                    return
                
                current_price = data['Close'].iloc[-1]
                
                # === ENSEMBLE PREDICTION CALCULATION ===
                # Ensemble prediction - predictions artık basit sayılar döndürüyor
                model_predictions = []
                for model_name, pred_value in predictions.items():
                    if isinstance(pred_value, (int, float)) and not np.isnan(pred_value) and not np.isinf(pred_value):
                        model_predictions.append(pred_value)
                
                if len(model_predictions) == 0:
                    st.error("❌ Hiçbir model geçerli tahmin üretemedi")
                    return
                
                ensemble_prediction = np.mean(model_predictions)
                ensemble_return = ((ensemble_prediction - current_price) / current_price) * 100
                
                # NaN kontrolü
                if np.isnan(ensemble_prediction) or np.isnan(ensemble_return):
                    st.error("❌ Ensemble tahmin hesaplamada NaN değer oluştu")
                    return
                
                # Model confidence (based on agreement)
                prediction_std = np.std(model_predictions)
                confidence = max(0.3, min(0.95, 1 - (prediction_std / current_price)))
                
                # Generate signal
                if ensemble_return > 2:
                    signal = "AL"
                elif ensemble_return < -2:
                    signal = "SAT"
                else:
                    signal = "BEKLE"
                
                # === PREDICTION DASHBOARD ===
                st.markdown("### 🎯 AI Tahmin Paneli")
                
                # Hedef tarih hesaplama ve Türkçe formatla
                today = datetime.now()
                target_date = today + timedelta(days=prediction_horizon)
                
                # Türkçe aylar ve günler
                turkish_months = {
                    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 
                    5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
                    9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
                }
                turkish_days = {
                    0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe',
                    4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'
                }
                
                target_day_tr = turkish_days[target_date.weekday()]
                target_month_tr = turkish_months[target_date.month]
                target_date_str = f"{target_date.day} {target_month_tr} {target_date.year}, {target_day_tr}"
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: hsl(210, 40%, 98%);">📅 Tahmin Hedefi</h4>
                            <p style="margin: 5px 0 0 0; color: #f0f0f0;">
                                <strong>{target_date_str}</strong> tarihindeki fiyat tahmini
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 24px; font-weight: bold;">{prediction_horizon}</div>
                            <div style="font-size: 14px;">gün sonra</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if model_type == "ensemble" or model_type == "all_models":
                    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">Mevcut Fiyat</div>
                            <div class="metric-value">₺{current_price:.2f}</div>
                            <div class="metric-change neutral">Canlı Piyasa</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        change_class = "positive" if ensemble_return > 0 else "negative"
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">{prediction_horizon} Gün Tahmini</div>
                            <div class="metric-value">₺{ensemble_prediction:.2f}</div>
                            <div class="metric-change {change_class}">{ensemble_return:+.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_class = "positive" if confidence > 0.7 else "negative" if confidence < 0.5 else "neutral"
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">AI Güveni</div>
                            <div class="metric-value">{confidence:.0%}</div>
                            <div class="metric-change {confidence_class}">Model Uyumu</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        signal_map = {
                            'AL': ('🚀', 'Güçlü Al', 'positive'), 
                            'SAT': ('📉', 'Güçlü Sat', 'negative'), 
                            'BEKLE': ('⏳', 'Bekle/Nötr', 'neutral')
                        }
                        icon, text, signal_class = signal_map.get(signal, ('⏳', 'Bekle', 'neutral'))
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">AI Sinyali</div>
                            <div class="metric-value">{icon}</div>
                            <div class="metric-change {signal_class}">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # === MODEL COMPARISON ===
                if model_type == "all_models":
                    st.markdown("### 🏆 Model Karşılaştırması")
                    
                    # Create comparison dataframe
                    model_data = []
                    model_names_tr = {
                        'random_forest': 'Rastgele Orman',
                        'gradient_boosting': 'Gradyan Artırma',
                        'linear_regression': 'Doğrusal Regresyon',
                        'svr': 'Destek Vektör Regresyonu',
                        'ensemble': 'Ensemble (Ortalama)'
                    }
                    
                    for model_name, prediction in predictions.items():
                        if isinstance(prediction, (int, float)) and not np.isnan(prediction) and not np.isinf(prediction):
                            return_pct = ((prediction - current_price) / current_price) * 100
                            model_data.append({
                                'Model': model_names_tr.get(model_name, model_name.replace('_', ' ').title()),
                                'Tahmin': f"₺{prediction:.2f}",
                                'Getiri %': f"{return_pct:+.2f}%",
                                'Yön': "🚀" if return_pct > 0 else "📉" if return_pct < 0 else "➡️"
                            })
                    
                    if model_data:
                        comparison_df = pd.DataFrame(model_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Model karşılaştırması için geçerli tahmin bulunamadı")
                
                # === SCENARIO ANALYSIS ===
                st.markdown("### 📊 Senaryo Analizi")
                
                scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                
                # Güvenli scenario hesaplama
                if np.isnan(prediction_std) or np.isinf(prediction_std):
                    prediction_std = abs(ensemble_prediction * 0.05)  # %5 default std
                
                # Optimistic scenario (+1 std)
                optimistic = ensemble_prediction + prediction_std
                optimistic_return = ((optimistic - current_price) / current_price) * 100
                
                # Pessimistic scenario (-1 std)
                pessimistic = ensemble_prediction - prediction_std
                pessimistic_return = ((pessimistic - current_price) / current_price) * 100
                
                # NaN kontrolü
                if np.isnan(optimistic) or np.isnan(pessimistic):
                    st.warning("⚠️ Senaryo analizi hesaplanamadı")
                    optimistic = ensemble_prediction * 1.05
                    pessimistic = ensemble_prediction * 0.95
                    optimistic_return = 5.0
                    pessimistic_return = -5.0
                
                with scenario_col1:
                    st.markdown(f"""
                    <div class="scenario-card optimistic">
                        <h4>🌟 İyimser</h4>
                        <div class="scenario-price">₺{optimistic:.2f}</div>
                        <div class="scenario-return">{optimistic_return:+.2f}%</div>
                        <div class="scenario-prob">%30 Olasılık</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with scenario_col2:
                    st.markdown(f"""
                    <div class="scenario-card neutral">
                        <h4>🎯 Beklenen</h4>
                        <div class="scenario-price">₺{ensemble_prediction:.2f}</div>
                        <div class="scenario-return">{ensemble_return:+.2f}%</div>
                        <div class="scenario-prob">%40 Olasılık</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with scenario_col3:
                    st.markdown(f"""
                    <div class="scenario-card pessimistic">
                        <h4>⚠️ Kötümser</h4>
                        <div class="scenario-price">₺{pessimistic:.2f}</div>
                        <div class="scenario-return">{pessimistic_return:+.2f}%</div>
                        <div class="scenario-prob">%30 Olasılık</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # === PREDICTION VISUALIZATION ===
                st.markdown(f"### 📈 Tahmin Görselleştirmesi ({prediction_horizon} Gün İleriye)")
                
                # Create prediction chart
                last_30_days = data.tail(30).copy()
                
                # Generate future dates
                last_date = last_30_days.index[-1]
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1), 
                    periods=prediction_horizon, 
                    freq='D'
                )
                
                # Create prediction line
                prediction_points = np.linspace(
                    current_price, 
                    ensemble_prediction, 
                    prediction_horizon + 1
                )[1:]  # Exclude the first point (current price)
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=last_30_days.index,
                    y=last_30_days['Close'],
                    mode='lines',
                    name='Geçmiş Fiyat',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                # Prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=prediction_points,
                    mode='lines+markers',
                    name=f'{prediction_horizon} Gün Tahmini',
                    line=dict(color='#ef4444', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Confidence band
                upper_band = prediction_points + prediction_std
                lower_band = prediction_points - prediction_std
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_band,
                    mode='lines',
                    name='Üst Güven Sınırı',
                    line=dict(color='rgba(239, 68, 68, 0.2)', width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_band,
                    mode='lines',
                    name='Güven Bandı',
                    line=dict(color='rgba(239, 68, 68, 0.2)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(239, 68, 68, 0.1)'
                ))
                
                fig.update_layout(
                    title=f'{selected_symbol} - {prediction_horizon} Gün AI Fiyat Tahmini',
                    xaxis_title='Tarih',
                    yaxis_title='Fiyat (₺)',
                    height=500,
                    showlegend=True,
                    template='plotly_white',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # === FEATURE IMPORTANCE ===
                importance_df = ml_predictor.get_feature_importance('random_forest')
                if not importance_df.empty:
                    st.markdown("### 🔍 AI Model İçgörüleri")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 En Önemli Özellikler**")
                        st.bar_chart(importance_df.set_index('feature')['importance'].head(8))
                    
                    with col2:
                        # Model performance metrics
                        st.markdown("**📊 Model Performansı**")
                    st.markdown(f"""
                        <div class="performance-metrics">
                            <div class="metric-row">
                                <span>Eğitim Skoru:</span>
                                <span>{training_results.get('train_score', 0):.3f}</span>
                            </div>
                            <div class="metric-row">
                                <span>Test Skoru:</span>
                                <span>{training_results.get('test_score', 0):.3f}</span>
                            </div>
                            <div class="metric-row">
                                <span>RMSE:</span>
                                <span>{training_results.get('rmse', 0):.3f}</span>
                            </div>
                            <div class="metric-row">
                                <span>MAE:</span>
                                <span>{training_results.get('mae', 0):.3f}</span>
                            </div>
                    </div>
                        """, unsafe_allow_html=True)
                
                # === RISK ASSESSMENT ===
                st.markdown("### ⚖️ Risk Değerlendirmesi")
                
                # Calculate risk metrics
                volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
                max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min()
                
                # Risk score based on volatility, prediction confidence, and market conditions
                risk_score = (volatility * 0.4) + ((1 - confidence) * 0.4) + (abs(ensemble_return/100) * 0.2)
                risk_level = "DÜŞÜK" if risk_score < 0.3 else "ORTA" if risk_score < 0.6 else "YÜKSEK"
                risk_color = "positive" if risk_level == "DÜŞÜK" else "neutral" if risk_level == "ORTA" else "negative"
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    st.markdown(f"""
                    <div class="metric-card-modern">
                        <div class="metric-title">Risk Seviyesi</div>
                        <div class="metric-value">{risk_level}</div>
                        <div class="metric-change {risk_color}">Skor: {risk_score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col2:
                    st.markdown(f"""
                    <div class="metric-card-modern">
                        <div class="metric-title">Volatilite</div>
                        <div class="metric-value">{volatility:.1%}</div>
                        <div class="metric-change neutral">Yıllık</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col3:
                    drawdown_color = "positive" if max_drawdown > -0.1 else "negative"
                    st.markdown(f"""
                    <div class="metric-card-modern">
                        <div class="metric-title">Maks Düşüş</div>
                        <div class="metric-value">{max_drawdown:.1%}</div>
                        <div class="metric-change {drawdown_color}">Tarihsel</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # === DISCLAIMER ===
                st.markdown("""
                <div class="info-box-modern warning">
                    <h4>⚠️ Yatırım Uyarısı</h4>
                    <p>AI tahminleri geçmiş veriler ve teknik indikatörlere dayanmaktadır. Geçmiş performans gelecek sonuçları garanti etmez. 
                    Yatırım kararları vermeden önce her zaman kendi araştırmanızı yapın ve bir finansal danışmana danışmayı düşünün.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Model eğitimi başarısız: {str(e)}")
    
    # Add custom CSS for new elements
    st.markdown("""
    <style>
    .scenario-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #e5e7eb;
        margin: 10px 0;
    }
    .scenario-card.optimistic { border-color: #10b981; }
    .scenario-card.pessimistic { border-color: #ef4444; }
    .scenario-card.neutral { border-color: #6b7280; }
    
    .scenario-price {
        font-size: 24px;
        font-weight: bold;
        color: #1f2937;
        margin: 10px 0;
    }
    .scenario-return {
        font-size: 18px;
        font-weight: 600;
        margin: 5px 0;
    }
    .scenario-prob {
        font-size: 14px;
        color: #6b7280;
    }
    
    .performance-metrics {
        background: #f9fafb;
        border-radius: 8px;
        padding: 15px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    </style>
                    """, unsafe_allow_html=True)

def show_stock_screener():
    """Hisse tarayıcı sayfası"""
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #262730; padding: 10px 20px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center;">
            <h1 style="margin: 0; font-size: 24px;">🔍 Hisse Tarayıcı</h1>
        </div>
        <span style="font-size: 16px; color: #a0a0a0;">Teknik kriterlere göre hisse taraması</span>
    </div>
    """, unsafe_allow_html=True)
    
    screener = StockScreener(BIST_SYMBOLS)
    

    
    time_intervals = {
        "1d": "1 Gün",
        "4h": "4 Saat", 
        "1h": "1 Saat",
        "30m": "30 Dakika",
        "15m": "15 Dakika",
        "5m": "5 Dakika"
    }
    
    # Tarama sekmeli yapısı
    tab1, tab2, tab4 = st.tabs(["🚀 BOĞA SİNYALLERİ", "⚡ TEKNİK TARAMALAR", "💰 DAY TRADE FIRSATLARI"])
    
    with tab1:
        # Boğa sinyali seçimi
        signal_types = {
            'OTT Buy Signal': '🔵 OTT Alım Sinyali',
            'VWAP Bull Signal': '📈 VWAP Boğa Sinyali',
            'Golden Cross': '🌟 Golden Cross',
            'MACD Bull Signal': '📊 MACD Boğa Sinyali',
            'RSI Recovery': '🔄 RSI Toparlanma',
            'Bollinger Breakout': '🎯 Bollinger Sıkışma',
            'Higher High + Higher Low': '📈 Yükselen Trend',
            'VWAP Reversal': '🔄 VWAP Geri Dönüş',
            'Volume Breakout': '💥 Hacim Patlaması',
            'Gap Up Signal': '⬆️ Gap Up Sinyali'
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<p style='margin-bottom:0; color:white;'>Sinyal Türü Seç</p>", unsafe_allow_html=True)
            selected_signal = st.selectbox("Sinyal Türü Seç", list(signal_types.keys()),
                                         format_func=lambda x: signal_types[x], key="signal_type", label_visibility="collapsed")
        
        with col2:
            st.markdown("<p style='margin-bottom:0; color:white;'>⏰ Zaman Dilimi</p>", unsafe_allow_html=True)
            selected_interval = st.selectbox(
                "⏰ Zaman Dilimi",
                list(time_intervals.keys()),
                format_func=lambda x: time_intervals[x],
                index=0,
                key="screener_interval",
                label_visibility="collapsed"
            )
        
        with col3:
            st.markdown("<p style='margin-bottom:0; color:white;'>&nbsp;</p>", unsafe_allow_html=True)
            scan_button = st.button("🔍 Sinyal Taraması Yap", type="primary", key="bull_signal_scan", use_container_width=True)

        with col4:
            st.markdown("<p style='margin-bottom:0; color:white;'>&nbsp;</p>", unsafe_allow_html=True)
            all_scan_button = st.button("🚀 Tüm Boğa Sinyallerini Tara", key="all_bull_signals", use_container_width=True)

        st.markdown("---_", unsafe_allow_html=True)

        if 'results' not in st.session_state:
            st.session_state.results = None

        if scan_button:
            with st.spinner(f"{signal_types[selected_signal]} sinyali aranıyor..."):
                # Seçili sinyale göre tarama fonksiyonu çağır
                if selected_signal == 'OTT Buy Signal':
                    results = screener.screen_by_ott_buy_signal(selected_interval)
                elif selected_signal == 'VWAP Bull Signal':
                    results = screener.screen_vwap_bull_signal(selected_interval)
                elif selected_signal == 'Golden Cross':
                    results = screener.screen_golden_cross(selected_interval)
                elif selected_signal == 'MACD Bull Signal':
                    results = screener.screen_macd_bull_signal(selected_interval)
                elif selected_signal == 'RSI Recovery':
                    results = screener.screen_rsi_recovery(selected_interval)
                elif selected_signal == 'Bollinger Breakout':
                    results = screener.screen_bollinger_breakout(selected_interval)
                elif selected_signal == 'Higher High + Higher Low':
                    results = screener.screen_higher_high_low(selected_interval)
                elif selected_signal == 'VWAP Reversal':
                    results = screener.screen_vwap_reversal(selected_interval)
                elif selected_signal == 'Volume Breakout':
                    results = screener.screen_volume_breakout(selected_interval)
                elif selected_signal == 'Gap Up Signal':
                    results = screener.screen_gap_up_signal(selected_interval)
                else:
                    results = []
                st.session_state.results = results

        if all_scan_button:
            with st.spinner("Tüm boğa sinyalleri taranıyor..."):
                all_results = screener.screen_all_bull_signals(selected_interval)
                st.session_state.results = all_results

        if st.session_state.results is not None:
            results = st.session_state.results
            if isinstance(results, list) and results:
                st.markdown(f"""
                <div class="info-box">
                    <h4>✅ {signal_types[selected_signal]} Sonuçları</h4>
                    <p>{len(results)} hisse bulundu</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sonuçları güçlü, orta, zayıf olarak grupla
                strong_signals = [r for r in results if r.get('strength') == 'Çok Güçlü']
                medium_signals = [r for r in results if r.get('strength') == 'Güçlü']
                weak_signals = [r for r in results if r.get('strength') == 'Orta']
                
                if strong_signals:
                    st.markdown("### 🟢 Çok Güçlü Sinyaller")
                    df_strong = pd.DataFrame(strong_signals)
                    st.dataframe(df_strong, use_container_width=True)
                
                if medium_signals:
                    st.markdown("### 🟡 Güçlü Sinyaller")
                    df_medium = pd.DataFrame(medium_signals)
                    st.dataframe(df_medium, use_container_width=True)
                
                if weak_signals:
                    st.markdown("### 🟠 Orta Sinyaller")
                    df_weak = pd.DataFrame(weak_signals)
                    st.dataframe(df_weak, use_container_width=True)

                # PDF İndirme Butonu
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.add_font('Roboto', '', 'assets/Roboto-Regular.ttf', uni=True)
                    pdf.add_font('Roboto-Bold', '', 'assets/Roboto-Bold.ttf', uni=True)
                    
                    # Başlık
                    pdf.set_font('Roboto-Bold', '', 16)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 15, 'HİSSE TARAMA RAPORU', 0, 1, 'C')
                    pdf.ln(5)
                    
                    # Tarih ve saat
                    from datetime import datetime
                    pdf.set_font('Roboto', '', 10)
                    pdf.set_text_color(127, 140, 141)
                    pdf.cell(0, 8, f'Rapor Tarihi: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
                    pdf.ln(10)
                    
                    # Sinyal türü başlığı
                    pdf.set_font('Roboto-Bold', '', 14)
                    pdf.set_text_color(52, 73, 94)
                    pdf.cell(0, 12, f'{signal_types.get(selected_signal, "Seçili Sinyal")} Sonuçları', 0, 1, 'L')
                    pdf.ln(5)
                    
                    # Tablo başlıkları
                    pdf.set_font('Roboto-Bold', '', 10)
                    pdf.set_fill_color(52, 152, 219)
                    pdf.set_text_color(255, 255, 255)
                    
                    # Başlık satırı
                    pdf.cell(40, 10, 'Sembol', 1, 0, 'C', True)
                    pdf.cell(80, 10, 'Şirket Adı', 1, 0, 'C', True)
                    pdf.cell(30, 10, 'Fiyat (₺)', 1, 0, 'C', True)
                    pdf.cell(30, 10, 'Sinyal', 1, 1, 'C', True)
                    
                    # Veri satırları
                    pdf.set_font('Roboto', '', 9)
                    pdf.set_text_color(44, 62, 80)
                    
                    for i, result in enumerate(results):
                        # Alternatif satır renkleri
                        if i % 2 == 0:
                            pdf.set_fill_color(236, 240, 241)
                        else:
                            pdf.set_fill_color(255, 255, 255)
                        
                        # Veri çıkarma
                        symbol = str(result.get('symbol', 'N/A'))[:8]
                        name = str(result.get('name', 'N/A'))[:25] + ('...' if len(str(result.get('name', ''))) > 25 else '')
                        price = f"{float(result.get('current_price', 0)):.2f}" if result.get('current_price') else 'N/A'
                        signal = str(result.get('signal', 'N/A'))[:8]
                        
                        pdf.cell(40, 8, symbol, 1, 0, 'C', True)
                        pdf.cell(80, 8, name, 1, 0, 'L', True)
                        pdf.cell(30, 8, price, 1, 0, 'C', True)
                        pdf.cell(30, 8, signal, 1, 1, 'C', True)
                    
                    # Özet bilgiler
                    pdf.ln(10)
                    pdf.set_font('Roboto-Bold', '', 12)
                    pdf.set_text_color(52, 73, 94)
                    pdf.cell(0, 10, 'RAPOR ÖZETİ', 0, 1, 'L')
                    pdf.ln(2)
                    
                    pdf.set_font('Roboto', '', 10)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 8, f'• Toplam bulunan hisse sayısı: {len(results)}', 0, 1, 'L')
                    pdf.cell(0, 8, f'• Tarama kriteri: {signal_types.get(selected_signal, "Bilinmeyen")}', 0, 1, 'L')
                    pdf.cell(0, 8, f'• Zaman dilimi: {selected_interval}', 0, 1, 'L')
                    
                    # Alt bilgi
                    pdf.ln(15)
                    pdf.set_font('Roboto', '', 8)
                    pdf.set_text_color(127, 140, 141)
                    pdf.cell(0, 6, 'Bu rapor otomatik olarak oluşturulmuştur. Yatırım kararlarınızda dikkatli olunuz.', 0, 1, 'C')
                    
                    pdf_output = pdf.output(dest='S')
                    b64 = base64.b64encode(pdf_output).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="tarama_sonuclari.pdf">📄 Modern PDF Raporu İndir</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"PDF oluşturulamadı: {str(e)}")

            elif isinstance(results, dict):
                # Her sinyal için sonuçları göster
                for signal_name, signal_results in results.items():
                    if signal_results:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin-top: 0; color: hsl(210, 40%, 98%);">{signal_types[signal_name]}</h3>
                            <p style="color: rgba(255,255,255,0.7);">{len(signal_results)} hisse bulundu</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        df = pd.DataFrame(signal_results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>{signal_types[signal_name]}</h4>
                            <p>Sinyal bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
                # PDF İndirme Butonu - Çoklu Sinyal
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.add_font('Roboto', '', 'assets/Roboto-Regular.ttf', uni=True)
                    pdf.add_font('Roboto-Bold', '', 'assets/Roboto-Bold.ttf', uni=True)
                    
                    # Başlık
                    pdf.set_font('Roboto-Bold', '', 16)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 15, 'TOPLU HİSSE TARAMA RAPORU', 0, 1, 'C')
                    pdf.ln(5)
                    
                    # Tarih ve saat
                    from datetime import datetime
                    pdf.set_font('Roboto', '', 10)
                    pdf.set_text_color(127, 140, 141)
                    pdf.cell(0, 8, f'Rapor Tarihi: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
                    pdf.ln(10)
                    
                    total_found = 0
                    for signal_name, signal_results in results.items():
                        if signal_results:
                            total_found += len(signal_results)
                            
                            # Sinyal başlığı
                            pdf.set_font('Roboto-Bold', '', 14)
                            pdf.set_text_color(52, 73, 94)
                            pdf.cell(0, 12, f'{signal_types[signal_name]} ({len(signal_results)} adet)', 0, 1, 'L')
                            pdf.ln(3)
                            
                            # Tablo başlıkları
                            pdf.set_font('Roboto-Bold', '', 9)
                            pdf.set_fill_color(52, 152, 219)
                            pdf.set_text_color(255, 255, 255)
                            
                            pdf.cell(35, 8, 'Sembol', 1, 0, 'C', True)
                            pdf.cell(70, 8, 'Şirket Adı', 1, 0, 'C', True)
                            pdf.cell(25, 8, 'Fiyat (₺)', 1, 0, 'C', True)
                            pdf.cell(25, 8, 'Sinyal', 1, 0, 'C', True)
                            pdf.cell(25, 8, 'Değişim%', 1, 1, 'C', True)
                            
                            # Veri satırları
                            pdf.set_font('Roboto', '', 8)
                            pdf.set_text_color(44, 62, 80)
                            
                            for i, result in enumerate(signal_results[:10]):  # İlk 10 sonuç
                                if i % 2 == 0:
                                    pdf.set_fill_color(236, 240, 241)
                                else:
                                    pdf.set_fill_color(255, 255, 255)
                                
                                symbol = str(result.get('symbol', 'N/A'))[:7]
                                name = str(result.get('name', 'N/A'))[:22] + ('...' if len(str(result.get('name', ''))) > 22 else '')
                                price = f"{float(result.get('current_price', 0)):.2f}" if result.get('current_price') else 'N/A'
                                signal = str(result.get('signal', 'N/A'))[:6]
                                change = f"{float(result.get('change_percent', 0)):.1f}" if result.get('change_percent') else 'N/A'
                                
                                pdf.cell(35, 7, symbol, 1, 0, 'C', True)
                                pdf.cell(70, 7, name, 1, 0, 'L', True)
                                pdf.cell(25, 7, price, 1, 0, 'C', True)
                                pdf.cell(25, 7, signal, 1, 0, 'C', True)
                                pdf.cell(25, 7, change, 1, 1, 'C', True)
                            
                            if len(signal_results) > 10:
                                pdf.set_font('Roboto', '', 8)
                                pdf.set_text_color(127, 140, 141)
                                pdf.cell(0, 6, f'... ve {len(signal_results) - 10} adet daha', 0, 1, 'C')
                            
                            pdf.ln(8)
                    
                    # Genel özet
                    pdf.set_font('Roboto-Bold', '', 12)
                    pdf.set_text_color(52, 73, 94)
                    pdf.cell(0, 10, 'GENEL RAPOR ÖZETİ', 0, 1, 'L')
                    pdf.ln(2)
                    
                    pdf.set_font('Roboto', '', 10)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 8, f'• Toplam bulunan hisse sayısı: {total_found}', 0, 1, 'L')
                    pdf.cell(0, 8, f'• Taranan sinyal türü sayısı: {len([s for s, r in results.items() if r])}', 0, 1, 'L')
                    pdf.cell(0, 8, f'• Zaman dilimi: {selected_interval}', 0, 1, 'L')
                    
                    # Alt bilgi
                    pdf.ln(10)
                    pdf.set_font('Roboto', '', 8)
                    pdf.set_text_color(127, 140, 141)
                    pdf.cell(0, 6, 'Bu rapor otomatik olarak oluşturulmuştur. Yatırım kararlarınızda dikkatli olunuz.', 0, 1, 'C')
                    
                    pdf_output = pdf.output(dest='S')
                    b64 = base64.b64encode(pdf_output).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="toplu_tarama_sonuclari.pdf">📊 Kapsamlı PDF Raporu İndir</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"PDF oluşturulamadı: {str(e)}")

            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Sonuç Bulunamadı</h4>
                    <p>{signal_types.get(selected_signal, 'Seçili')} kriteri karşılayan hisse bulunamadı</p>
                </div>
                """, unsafe_allow_html=True)

    
    with tab2:

        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card hover-glow">
                <h3 style="margin-top: 0; color: hsl(210, 40%, 98%);">⚡ RSI Taraması</h3>
                <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Göreceli güç endeksi bazlı filtreleme</p>
            """, unsafe_allow_html=True)
            
            rsi_min = st.slider("RSI Min", 0, 100, 30, key="rsi_min")
            rsi_max = st.slider("RSI Max", 0, 100, 70, key="rsi_max")
            
            if st.button("🔍 RSI Taraması Yap", key="rsi_scan"):
                with st.spinner("Hisseler taranıyor..."):
                    results = screener.screen_by_rsi(rsi_min, rsi_max, selected_interval)
                    if results:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="info-box">
                            <h4>✅ RSI Tarama Sonuçları</h4>
                            <p>{} hisse bulundu</p>
                        </div>
                        """.format(len(results)), unsafe_allow_html=True)
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Sonuç Bulunamadı</h4>
                            <p>Belirtilen RSI aralığında hisse bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card hover-glow">
                <h3 style="margin-top: 0; color: hsl(210, 40%, 98%);">📊 Hacim Artışı</h3>
                <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Ortalama hacmin üzerindeki hisseler</p>
            """, unsafe_allow_html=True)
            
            volume_multiplier = st.slider("Hacim Çarpanı", 1.0, 5.0, 1.5, 0.1, key="volume_mult")
            
            if st.button("📈 Hacim Taraması Yap", key="volume_scan"):
                with st.spinner("Hacim artışları aranıyor..."):
                    results = screener.screen_by_volume(volume_multiplier, selected_interval)
                    if results:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="info-box">
                            <h4>✅ Hacim Tarama Sonuçları</h4>
                            <p>{} hisse bulundu</p>
                        </div>
                        """.format(len(results)), unsafe_allow_html=True)
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Sonuç Bulunamadı</h4>
                            <p>Belirtilen hacim çarpanında hisse bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card hover-glow">
                <h3 style="margin-top: 0; color: hsl(210, 40%, 98%);">🚀 Fiyat Kırılımları</h3>
                <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Destek/direnç kırılımları</p>
            """, unsafe_allow_html=True)
            
            lookback = st.slider("Geriye Bakış (Gün)", 10, 50, 20, key="lookback_days")
            
            if st.button("⚡ Kırılım Taraması Yap", key="breakout_scan"):
                with st.spinner("Kırılımlar aranıyor..."):
                    results = screener.screen_by_price_breakout(lookback, selected_interval)
                    if results:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="info-box">
                            <h4>✅ Kırılım Tarama Sonuçları</h4>
                            <p>{} hisse bulundu</p>
                        </div>
                        """.format(len(results)), unsafe_allow_html=True)
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Sonuç Bulunamadı</h4>
                            <p>Belirtilen sürede kırılım bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)
    

    
    with tab4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin-top: 0; color: hsl(210, 40%, 98%);">🚀 Day Trade Fırsatları</h2>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Teknik göstergelerle dikkat çeken day trade fırsatları</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Refresh button for day trade opportunities
        refresh_daytrading = st.button("🔄 Fırsatları Tara", type="primary", key="refresh_daytrading")
        
        if refresh_daytrading or "daytrading_results" not in st.session_state:
            with st.spinner("🔍 Day trade fırsatları taranıyor..."):
                daytrading_opportunities = scan_daytrading_opportunities()
                st.session_state.daytrading_results = daytrading_opportunities
        
        if "daytrading_results" in st.session_state and st.session_state.daytrading_results:
            opportunities = st.session_state.daytrading_results

            st.markdown("#### Filtreleme Seçenekleri")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_score = st.slider("Minimum Puan", 0, 10, 5, key="min_score_slider_v2")
            with col2:
                min_volatility = st.slider("Minimum Volatilite (%)", 0.0, 15.0, 2.0, 0.1, key="min_volatility_slider_v2")
            with col3:
                min_volume_ratio = st.slider("Minimum Hacim Oranı", 0.0, 10.0, 1.5, 0.1, key="min_volume_ratio_slider_v2")

            filtered_ops = [
                op for op in opportunities 
                if op['score'] >= min_score and 
                   op['volatility'] >= min_volatility and 
                   op['volume_ratio'] >= min_volume_ratio
            ]

            st.markdown(f"**{len(filtered_ops)}** adet fırsat bulundu.")

            # Display opportunities in a more structured way
            for i in range(0, len(filtered_ops), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(filtered_ops):
                        with cols[j]:
                            opportunity = filtered_ops[i+j]
                            score_color = "#00ff88" if opportunity['score'] >= 8 else "#f39c12"
                            signal_color = "#00ff88" if opportunity['signal'] == "AL" else "#ff4757" if opportunity['signal'] == "SAT" else "#f39c12"
                            
                            st.markdown(f"""
                            <div class="metric-card hover-glow">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <h4 style="margin: 0; color: hsl(210, 40%, 98%);">{opportunity['symbol']}</h4>
                                    <span style="background: {score_color}; color: black; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{opportunity['score']}/10</span>
                                </div>
                                <p style="margin: 0.25rem 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">{opportunity['name']}</p>
                                <div style="margin: 0.5rem 0;">
                                    <span style="color: {signal_color}; font-weight: bold; font-size: 1.1rem;">{opportunity['signal']}</span>
                                    <span style="color: rgba(255,255,255,0.6); margin-left: 0.5rem;">₺{opportunity['price']:.2f}</span>
                                </div>
                                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7);">
                                    <div>📊 Volatilite: {opportunity['volatility']:.1f}%</div>
                                    <div>📈 Hacim: {opportunity['volume_ratio']:.1f}x</div>
                                    <div>⚡ RSI: {opportunity['rsi']:.0f}</div>
                                </div>
                                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.5);">
                                    {opportunity['reason']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            # Summary stats
            st.markdown("### 📈 Tarama Özeti")
            col1, col2, col3, col4 = st.columns(4)
            
            total_scanned = 50 # Assuming first 50 BIST symbols are scanned
            total_opportunities = len(opportunities)
            high_potential = len([op for op in opportunities if op.get('score', 0) >= 7])
            avg_score = sum(op['score'] for op in opportunities) / len(opportunities) if opportunities else 0
            
            with col1:
                st.metric("Taranan Hisse", total_scanned)
            with col2:
                st.metric("Toplam Fırsat", total_opportunities)
            with col3:
                st.metric("Yüksek Potansiyel", high_potential)
            with col4:
                st.metric("Ortalama Puan", f"{avg_score:.1f}/10")
        
        else:
            st.info("🔍 Day trade fırsatlarını görmek için 'Fırsatları Tara' butonuna tıklayın.")

def show_pattern_analysis():
    """Pattern analizi sayfası"""
    st.markdown("""
    <div class="page-header">
        <h1 style="display: inline-block; margin-right: 1rem;">🎯 Patern Analizi</h1>
        <span style="color: rgba(255,255,255,0.8); font-size: 1.1rem; display: inline-block; vertical-align: middle;">Gelişmiş formasyon tespiti ve sinyal analizi</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Ayarlar Paneli ---
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin-top: 0; color: hsl(210, 40%, 98%);">⚙️ Analiz Ayarları</h3>
        <p style="color: rgba(255,255,255,0.7);">Hisse, zaman dilimi ve veri periyodu seçerek analizi başlatın.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_symbol = st.selectbox(
            "📊 Hisse Senedi",
            options=sorted(list(BIST_SYMBOLS.keys())),
            format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
            key="pattern_stock_select_v2"
        )

    with col2:
        time_interval = st.selectbox(
            "⏰ Zaman Dilimi",
            options=["5m", "15m", "1h", "4h", "1d"],
            index=4,  # Varsayılan: 1d
            key="pattern_time_interval_v2"
        )

    with col3:
        if time_interval in ["5m", "15m"]:
            period_options = {"1d": "1 Gün", "5d": "5 Gün", "1mo": "1 Ay"}
            default_period = "5d"
        elif time_interval in ["1h", "4h"]:
            period_options = {"1mo": "1 Ay", "3mo": "3 Ay", "6mo": "6 Ay"}
            default_period = "3mo"
        else:  # 1d
            period_options = {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl", "5y": "5 Yıl"}
            default_period = "1y"

        time_period = st.selectbox(
            "📅 Veri Periyodu",
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            index=list(period_options.keys()).index(default_period),
            key="pattern_time_period_v2"
        )

    with st.expander("🔧 Gelişmiş Ayarlar"):
        col1, col2 = st.columns(2)
        with col1:
            lookback_period = st.slider("Formasyon Arama Periyodu (Bar Sayısı)", 5, 100, 30, key="pattern_lookback_v2")
        with col2:
            sensitivity = st.slider("Tespit Hassasiyeti", 0.5, 1.5, 1.0, 0.1, key="pattern_sensitivity_v2")

    if st.button("🔍 Analizi Başlat", type="primary", use_container_width=True):
        with st.spinner(f"{selected_symbol} için formasyonlar analiz ediliyor..."):
            fetcher = BISTDataFetcher()
            data = fetcher.get_stock_data(selected_symbol, period=time_period, interval=time_interval)

            if data is not None and not data.empty:
                st.success(f"{selected_symbol} için {len(data)} adet bar verisi başarıyla çekildi.")
                
                # --- Candlestick Patternleri ---
                st.markdown("### 🕯️ Candlestick Formasyonları")
                pattern_analyzer = PatternRecognition(data)
                latest_patterns = pattern_analyzer.get_latest_patterns(lookback=lookback_period)
                
                candlestick_names = {
                    'doji': '⭐ Doji', 'hammer': '🔨 Çekiç', 'shooting_star': '🌠 Kayan Yıldız',
                    'bullish_engulfing': '🟢 Yükseliş Saran', 'bearish_engulfing': '🔴 Düşüş Saran',
                    'morning_star': '🌅 Sabah Yıldızı', 'evening_star': '🌆 Akşam Yıldızı'
                }
                
                detected_candlesticks = {k: v for k, v in latest_patterns.items() if v is not None}
                if detected_candlesticks:
                    cols = st.columns(len(detected_candlesticks))
                    for i, (pattern, date) in enumerate(detected_candlesticks.items()):
                        with cols[i]:
                            st.metric(label=candlestick_names.get(pattern, pattern.replace('_', ' ').title()), 
                                      value="Tespit Edildi", 
                                      help=f"Tarih: {date.strftime('%Y-%m-%d')}" if date else "Tarih bulunamadı")
                else:
                    st.info("Belirtilen periyotta belirgin bir candlestick formasyonu bulunamadı.")

                # --- Chart Patternleri (Gelişmiş) ---
                st.markdown("### 📈 Grafik Formasyonları")
                try:
                    from modules.pattern_recognition_advanced import AdvancedPatternRecognition
                    advanced_analyzer = AdvancedPatternRecognition(data, sensitivity=sensitivity)
                    chart_patterns = advanced_analyzer.detect_all_patterns(lookback=lookback_period)
                    
                    chart_names = {
                        'flag': '🚩 Bayrak', 'pennant': '🔺 Flama', 'triangle': '📐 Üçgen',
                        'head_shoulders': '👤 OBO/TOBO', 'double_top': '⛰️ Çift Tepe', 'double_bottom': '🏔️ Çift Dip'
                    }

                    detected_charts = {k: v for k, v in chart_patterns.items() if v}
                    if detected_charts:
                        cols = st.columns(len(detected_charts))
                        for i, (pattern, details) in enumerate(detected_charts.items()):
                            with cols[i]:
                                st.metric(label=chart_names.get(pattern, pattern.replace('_', ' ').title()),
                                          value=f"{details['type']}",
                                          help=f"Periyot: {details['start_date']} - {details['end_date']}")
                    else:
                        st.info("Belirtilen periyotta belirgin bir grafik formasyonu bulunamadı.")

                except ImportError:
                    st.warning("Gelişmiş formasyon modülü bulunamadı.")
                except Exception as e:
                    st.error(f"Grafik formasyon analizi sırasında bir hata oluştu: {e}")

            else:
                st.error("Veri alınamadı. Lütfen farklı bir hisse veya zaman aralığı deneyin.")







if __name__ == "__main__":
    main()