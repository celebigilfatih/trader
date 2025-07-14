import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time

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
from modules.risk_calculator import RiskCalculator

# Navigation için
from streamlit_option_menu import option_menu

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="BIST Teknik Analiz Uygulaması",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_chart(df, analyzer, selected_indicators):
    """Modern Plotly grafik oluşturur"""
    
    # Alt grafikler oluştur (ana grafik + volume + RSI)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=['Price & Indicators', 'Volume', 'RSI'],
        row_width=[0.6, 0.2, 0.2],
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
        rsi_data = analyzer.indicators['rsi']
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data,
                name="RSI",
                line=dict(color='#ff9800', width=2)
            ),
            row=3, col=1
        )
        
        # RSI seviyeler
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Teknik indikatörleri ana grafiğe ekle
    for indicator, enabled in selected_indicators.items():
        if enabled and indicator in analyzer.indicators:
            indicator_data = analyzer.indicators[indicator]
            config = INDICATORS_CONFIG.get(indicator, {})
            
            if indicator.startswith('ema'):
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
                if isinstance(indicator_data, dict):
                    # Üst bant
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=indicator_data.get('upper', []),
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
                            y=indicator_data.get('lower', []),
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
                            y=indicator_data.get('middle', []),
                            name="BB Middle",
                            line=dict(color='#9e9e9e', width=1)
                        ),
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
        height=600
    )
    
    # X ekseni ayarları
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        showline=True,
        linecolor='rgba(255,255,255,0.2)'
    )
    
    # Y ekseni ayarları
    fig.update_yaxes(
        showgrid=True,
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
        
        /* Modern SaaS Dashboard Theme */
        .main {
            background-color: hsl(224, 71%, 4%);
            color: hsl(210, 40%, 98%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Sidebar */
        .css-1d391kg {
            background-color: hsl(224, 71%, 4%);
            border-right: 1px solid hsl(215, 28%, 17%);
            width: 280px !important;
        }
        
        /* Main content area */
        .main .block-container {
            padding: 2rem 2rem 2rem 2rem;
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
            background: hsl(224, 71%, 4%);
            border: 1px solid hsl(215, 28%, 17%);
            border-radius: 0.75rem;
            padding: 1.5rem;
            position: relative;
            transition: border-color 0.2s;
        }
        
        .kpi-card:hover, .metric-card:hover, .metric-card-modern:hover, .modern-card:hover, .chart-card:hover, .info-card:hover {
            border-color: hsl(215, 28%, 25%);
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
            background: hsl(224, 71%, 4%);
            border: 1px solid hsl(215, 28%, 17%);
            border-radius: 0.75rem;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .page-header h1, .page-header-modern h1 {
            font-size: 2rem;
            font-weight: 700;
            color: hsl(210, 40%, 98%);
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
            background: hsl(224, 71%, 4%);
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
            border: 1px solid hsl(215, 28%, 17%);
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 1rem 0;
            background: hsl(224, 71%, 4%);
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
            color: hsl(210, 40%, 98%);
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
            background: linear-gradient(135deg, hsl(224, 71%, 4%) 0%, hsl(215, 28%, 17%) 100%);
            border: 1px solid hsl(215, 28%, 17%);
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
            background: hsl(224, 71%, 4%);
            border: 1px solid hsl(215, 28%, 17%);
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
            border: 1px solid hsl(215, 28%, 17%);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        /* Modern Table */
        .dataframe {
            background: hsl(224, 71%, 4%);
            border: 1px solid hsl(215, 28%, 17%);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: hsl(210, 40%, 98%);
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
        <div style="margin-bottom: 1rem;">
            <div style="color: #8B8B8B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; padding-left: 0.5rem;">
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
        <div style="margin: 1.5rem 0 1rem 0;">
            <div style="color: #8B8B8B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; padding-left: 0.5rem;">
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
        
        # Sentiment Analysis
        if st.button("📰 Duygu Analizi", key="sentiment_btn", use_container_width=True,
                    type="primary" if current_menu == "sentiment" else "secondary"):
            st.session_state.selected_menu = "sentiment"
            st.rerun()
        
        # Tools Section
        st.markdown("""
        <div style="margin: 1.5rem 0 1rem 0;">
            <div style="color: #8B8B8B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; padding-left: 0.5rem;">
                Tools
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Management
        if st.button("⚡ Risk Yönetimi", key="risk_btn", use_container_width=True,
                    type="primary" if current_menu == "risk" else "secondary"):
            st.session_state.selected_menu = "risk"
            st.rerun()
    
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
    elif current_menu == "risk":
        show_risk_management()
    elif current_menu == "sentiment":
        show_sentiment_analysis()
    else:
        # Varsayılan olarak dashboard göster
        show_modern_dashboard()

def show_technical_analysis():
    """Teknik analiz sayfası - Modern Shadcn stil"""
    
    # Modern page header with controls
    st.markdown("""
    <div class="page-header">
        <h1>📈 Teknik Analiz</h1>
        <p>Gelişmiş teknik indikatörlerle gerçek zamanlı BIST hisse analizi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern header controls
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # Header kontrolleri - 3 sütunlu layout
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    
    with header_col1:
        st.markdown("**📊 Hisse Seçimi**")
        selected_symbol = st.selectbox(
            "Hisse",
            options=list(BIST_SYMBOLS.keys()),
            format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
            label_visibility="collapsed",
            key="header_symbol"
        )
    
    with header_col2:
        st.markdown("**⏰ Zaman Aralığı**")
        time_interval = st.selectbox(
            "Aralık",
            ["5m", "15m", "1h", "2h", "4h", "1d"],
            index=5,
            format_func=lambda x: {
                "5m": "5 Dakika", "15m": "15 Dakika", "1h": "1 Saat",
                "2h": "2 Saat", "4h": "4 Saat", "1d": "1 Gün"
            }[x],
            label_visibility="collapsed",
            key="header_interval"
        )
    
    with header_col3:
        st.markdown("**📅 Dönem**")
        
        if time_interval in ["5m", "15m"]:
            period_options = ["1d", "7d", "30d"]
            default_period = "7d"
        elif time_interval in ["1h", "2h", "4h"]:
            period_options = ["7d", "30d", "90d"] 
            default_period = "30d"
        else:
            period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
            default_period = "1y"
        
        time_period = st.selectbox(
            "Dönem",
            period_options,
            index=period_options.index(default_period),
            format_func=lambda x: {
                "1d": "1 Gün", "7d": "7 Gün", "30d": "30 Gün", "90d": "90 Gün",
                "1mo": "1 Ay", "3mo": "3 Ay", "6mo": "6 Ay", 
                "1y": "1 Yıl", "2y": "2 Yıl", "5y": "5 Yıl"
            }.get(x, x),
            label_visibility="collapsed",
            key="header_period"
        )
    
    st.markdown("---")
    
    # Simplified sidebar - only indicators
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px 0;">
            <h3 style="color: #ffffff; margin-bottom: 20px;">📈 İndikatörler</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_indicators = {}
        
        # Hareketli Ortalamalar - Düzenli layout
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 0.9rem;">📊 Hareketli Ortalamalar</h4>
        </div>
        """, unsafe_allow_html=True)
        
        ema_indicators = ['ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_50', 'ema_121']
        ema_cols = st.columns(2)  # 2 sütunlu layout için daha düzenli
        
        for i, indicator in enumerate(ema_indicators):
            if indicator in INDICATORS_CONFIG:
                config = INDICATORS_CONFIG[indicator]
                with ema_cols[i % 2]:
                    selected_indicators[indicator] = st.checkbox(
                        config["name"], 
                        value=config["default"],
                        key=f"check_{indicator}"
                    )
        
        # Ana İndikatörler - OTT, SuperTrend, VWAP, RSI, MACD
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 0.9rem;">📈 Ana İndikatörler</h4>
        </div>
        """, unsafe_allow_html=True)
        
        main_indicators = ['ott', 'supertrend', 'vwap', 'rsi', 'macd']
        main_cols = st.columns(2)  # 2 sütunlu layout
        
        for i, indicator in enumerate(main_indicators):
            if indicator in INDICATORS_CONFIG:
                config = INDICATORS_CONFIG[indicator]
                with main_cols[i % 2]:
                    selected_indicators[indicator] = st.checkbox(
                        config["name"],
                        value=config["default"],
                        key=f"check_{indicator}"
                    )
        
        # Diğer İndikatörler
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: #ffffff; margin-bottom: 0.5rem; font-size: 0.9rem;">📊 Diğer İndikatörler</h4>
        </div>
        """, unsafe_allow_html=True)
        
        other_indicators = ['bollinger', 'stoch', 'williams_r', 'cci']
        other_cols = st.columns(2)  # 2 sütunlu layout
        
        for i, indicator in enumerate(other_indicators):
            if indicator in INDICATORS_CONFIG:
                config = INDICATORS_CONFIG[indicator]
                with other_cols[i % 2]:
                    selected_indicators[indicator] = st.checkbox(
                        config["name"],
                        value=config["default"],
                        key=f"check_{indicator}"
                    )
        
        st.markdown("---")
        
        # Uyarılar
        st.markdown("**🚨 Uyarı Ayarları**")
        enable_alerts = st.checkbox("Uyarıları Aktif Et", value=True)
        
        if enable_alerts:
            alert_methods = st.multiselect(
                "Uyarı Yöntemi",
                ["Email", "Telegram", "Desktop"], 
                default=["Desktop"]
            )
    
    # Ana içerik
    try:
        with st.spinner("Veriler yükleniyor..."):
            fetcher = BISTDataFetcher()
            df = fetcher.get_stock_data(selected_symbol, period=time_period, interval=time_interval)
            
            if df is not None and not df.empty:
                analyzer = TechnicalAnalyzer(df)
                
                # İndikatörleri hesapla
                for indicator, enabled in selected_indicators.items():
                    if enabled:
                        analyzer.add_indicator(indicator)
                
                # Grafik
                fig = create_chart(df, analyzer, selected_indicators)
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal
                alert_system = AlertSystem()
                signal = alert_system.generate_signal(analyzer)
                
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
                st.markdown("### 🚀 Boğa Sinyalleri")
                
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
                            <div class="signal-icon">🚀</div>
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
                            <div class="signal-icon">🚀</div>
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
                            <div class="signal-icon">🚀</div>
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
                            <div class="signal-icon">🚀</div>
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
                
                # Market Info
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change = latest['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                
                # Modern metrics
                st.markdown("### 📊 Piyasa Bilgileri")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    change_class = "positive" if change >= 0 else "negative"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Mevcut Fiyat</div>
                        <div class="metric-value">₺{latest['Close']:.2f}</div>
                        <div class="metric-change {change_class}">
                            {'+' if change >= 0 else ''}{change:.2f} ({change_pct:+.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Günlük Yüksek</div>
                        <div class="metric-value">₺{latest['High']:.2f}</div>
                        <div class="metric-change neutral">24s Aralık</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Günlük Düşük</div>
                        <div class="metric-value">₺{latest['Low']:.2f}</div>
                        <div class="metric-change neutral">Destek Seviyesi</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    volume_change = ((latest['Volume'] - df['Volume'].tail(20).mean()) / df['Volume'].tail(20).mean()) * 100
                    volume_class = "positive" if volume_change >= 0 else "negative"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Hacim</div>
                        <div class="metric-value">{latest['Volume']:,.0f}</div>
                        <div class="metric-change {volume_class}">
                            {volume_change:+.1f}% ortalamaya göre
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Hareketli Ortalama Uzaklıkları
                ema_indicators = ['ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_50', 'ema_121']
                selected_emas = [ind for ind in ema_indicators if selected_indicators.get(ind, False)]
                
                if selected_emas:
                    st.markdown("### 📏 Hareketli Ortalama Uzaklıkları")
                    
                    current_price = latest['Close']
                    indicator_values = analyzer.get_latest_indicators()
                    
                    # EMA uzaklık kartları
                    ema_distance_cols = st.columns(min(len(selected_emas), 4))
                    
                    for i, indicator in enumerate(selected_emas):
                        if indicator in indicator_values:
                            ema_value = indicator_values[indicator]
                            distance = current_price - ema_value
                            distance_pct = (distance / ema_value) * 100
                            
                            config = INDICATORS_CONFIG.get(indicator, {})
                            distance_class = "positive" if distance >= 0 else "negative"
                            
                            with ema_distance_cols[i % len(ema_distance_cols)]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">{config.get('name', indicator)}</div>
                                    <div class="metric-value">₺{ema_value:.2f}</div>
                                    <div class="metric-change {distance_class}">
                                        {'+' if distance >= 0 else ''}{distance:.2f} ({distance_pct:+.1f}%)
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # İndikatör değerleri
                if any(selected_indicators.values()):
                    st.markdown("### 🔬 İndikatör Değerleri")
                    indicator_values = analyzer.get_latest_indicators()
                    
                    # Sadece EMA olmayan indikatörler için
                    non_ema_indicators = {k: v for k, v in selected_indicators.items() 
                                        if v and k not in ema_indicators}
                    
                    if non_ema_indicators:
                        # İndikatör kartları
                        indicator_cols = st.columns(min(len(non_ema_indicators), 4))
                        
                        col_idx = 0
                        current_price = latest['Close']
                        
                        for indicator, enabled in non_ema_indicators.items():
                            if enabled and indicator in indicator_values:
                                value = indicator_values[indicator]
                                config = INDICATORS_CONFIG.get(indicator, {})
                                
                                # İndikatör durumunu belirleme ve tooltip içeriği
                                status_class = "neutral"
                                status_text = "Nötr"
                                status_icon = "⚪"
                                tooltip_title = ""
                                tooltip_description = ""
                                tooltip_range = ""
                                
                                if indicator == 'rsi':
                                    tooltip_title = "RSI (Relative Strength Index)"
                                    tooltip_description = "14 günlük momentum osilatörü. Aşırı alım/satım seviyelerini gösterir."
                                    tooltip_range = "<strong>Seviyeler:</strong> 0-30 Aşırı Satım, 30-70 Normal, 70-100 Aşırı Alım"
                                    
                                    if value > 70:
                                        status_class = "negative"
                                        status_text = "Satış Baskısı Beklentisi"
                                        status_icon = "🔴"
                                    elif value < 30:
                                        status_class = "positive"
                                        status_text = "Alış Fırsatı Sinyali"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "neutral"
                                        status_text = "Dengeli Momentum"
                                        status_icon = "⚪"
                                
                                elif indicator == 'macd':
                                    tooltip_title = "MACD (Moving Average Convergence Divergence)"
                                    tooltip_description = "12-26 günlük hareketli ortalama farkı. Trend değişimi sinyalleri verir."
                                    tooltip_range = "<strong>Yorumlama:</strong> 0 üstü Yukarı Momentum, 0 altı Aşağı Momentum"
                                    
                                    if value > 0:
                                        status_class = "positive"
                                        status_text = "Yukarı Momentum"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "negative"
                                        status_text = "Aşağı Momentum"
                                        status_icon = "🔴"
                                
                                elif indicator == 'stoch':
                                    tooltip_title = "Stochastic Oscillator"
                                    tooltip_description = "14 günlük fiyat pozisyonunu ölçer. Kısa vadeli dönüş noktalarını gösterir."
                                    tooltip_range = "<strong>Seviyeler:</strong> 0-20 Aşırı Satım, 20-80 Normal, 80-100 Aşırı Alım"
                                    
                                    if value > 80:
                                        status_class = "negative"
                                        status_text = "Düzeltme Beklentisi"
                                        status_icon = "🔴"
                                    elif value < 20:
                                        status_class = "positive"
                                        status_text = "Toparlanma Beklentisi"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "neutral"
                                        status_text = "Kararlı Fiyat Bandı"
                                        status_icon = "⚪"
                                
                                elif indicator == 'williams_r':
                                    tooltip_title = "Williams %R"
                                    tooltip_description = "14 günlük ters momentum osilatörü. Kısa vadeli geri dönüşleri işaret eder."
                                    tooltip_range = "<strong>Seviyeler:</strong> -100/-80 Aşırı Satım, -80/-20 Normal, -20/0 Aşırı Alım"
                                    
                                    if value > -20:
                                        status_class = "negative"
                                        status_text = "Satış Sinyali Yakın"
                                        status_icon = "🔴"
                                    elif value < -80:
                                        status_class = "positive"
                                        status_text = "Alış Sinyali Yakın"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "neutral"
                                        status_text = "Trend Devam Ediyor"
                                        status_icon = "⚪"
                                
                                elif indicator == 'cci':
                                    tooltip_title = "CCI (Commodity Channel Index)"
                                    tooltip_description = "Fiyatın tipik seviyesinden sapmasını ölçer. Aşırı alım/satım koşullarını gösterir."
                                    tooltip_range = "<strong>Seviyeler:</strong> -100'un altı Aşırı Satım, -100/+100 Normal, +100'ün üstü Aşırı Alım"
                                    
                                    if value > 100:
                                        status_class = "negative"
                                        status_text = "Geri Çekilme Beklenir"
                                        status_icon = "🔴"
                                    elif value < -100:
                                        status_class = "positive"
                                        status_text = "Yükseliş Beklenir"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "neutral"
                                        status_text = "Doğal Fiyat Seviyesi"
                                        status_icon = "⚪"
                                
                                elif indicator in ['ott', 'supertrend', 'vwap']:
                                    if indicator == 'ott':
                                        tooltip_title = "OTT (Optimized Trend Tracker)"
                                        tooltip_description = "Trend takip indikatörü. Dinamik destek/direnç seviyesi sağlar."
                                        tooltip_range = "<strong>Pozisyon:</strong> Fiyat üstünde = Alış Sinyali, Fiyat altında = Satış Sinyali"
                                    elif indicator == 'supertrend':
                                        tooltip_title = "SuperTrend"
                                        tooltip_description = "ATR bazlı trend takip indikatörü. Net alış/satış sinyalleri verir."
                                        tooltip_range = "<strong>Pozisyon:</strong> Fiyat üstünde = Alış Trendi, Fiyat altında = Satış Trendi"
                                    else:  # vwap
                                        tooltip_title = "VWAP (Volume Weighted Average Price)"
                                        tooltip_description = "Hacim ağırlıklı ortalama fiyat. Kurumsal işlem seviyesini gösterir."
                                        tooltip_range = "<strong>Pozisyon:</strong> Fiyat üstünde = Güçlü Pozisyon, Fiyat altında = Zayıf Pozisyon"
                                    
                                    if current_price > value:
                                        status_class = "positive"
                                        status_text = "Alış Bölgesi"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "negative"
                                        status_text = "Satış Bölgesi"
                                        status_icon = "🔴"
                                
                                elif indicator == 'bollinger':
                                    tooltip_title = "Bollinger Bands (Orta Band)"
                                    tooltip_description = "20 günlük hareketli ortalama + volatilite bantları. Fiyat bandlarını gösterir."
                                    tooltip_range = "<strong>Pozisyon:</strong> Orta bant trend merkezi, Üst band = kuvvet, Alt band = zayıflık"
                                    
                                    # Bollinger Bands için orta band ile karşılaştırma
                                    if abs(current_price - value) / value < 0.02:  # %2 tolerance
                                        status_class = "neutral"
                                        status_text = "Trend Merkezi"
                                        status_icon = "⚪"
                                    elif current_price > value:
                                        status_class = "positive"
                                        status_text = "Güçlü Bölge"
                                        status_icon = "🟢"
                                    else:
                                        status_class = "negative"
                                        status_text = "Zayıf Bölge"
                                        status_icon = "🔴"
                                
                                with indicator_cols[col_idx % len(indicator_cols)]:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-info-icon">i</div>
                                        <div class="metric-tooltip">
                                            <div class="metric-tooltip-title">{tooltip_title}</div>
                                            <div class="metric-tooltip-description">{tooltip_description}</div>
                                            <div class="metric-tooltip-range">{tooltip_range}</div>
                                        </div>
                                        <div class="metric-title">{config.get('name', indicator)}</div>
                                        <div class="metric-value">{value:.2f}</div>
                                        <div class="metric-change {status_class}">
                                            <span>{status_icon}</span>
                                            <span>{status_text}</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                col_idx += 1
                
            else:
                st.markdown("""
                <div class="error-box">
                    <h4>⚠️ Veri Hatası</h4>
                    <p>Seçilen hisse için veri yüklenemedi.</p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Hata</h4>
            <p>{str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

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
                options=list(BIST_SYMBOLS.keys()),
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
        # Adjust period based on interval for better performance
        if time_interval in ["5m", "15m"]:
            period = "1mo"  # 1 month for short intervals
        elif time_interval in ["1h", "2h"]:
            period = "3mo"  # 3 months for hourly intervals
        elif time_interval == "4h":
            period = "6mo"  # 6 months for 4-hour intervals
        else:
            period = "1y"   # 1 year for daily intervals
            
        df = fetcher.get_stock_data(selected_symbol, period=period, interval=time_interval)
        
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
                trend_icon = "📈" if week_change > 0 else "📉"
                change_class = "positive" if week_change > 0 else "negative"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <div class="kpi-title">
                            <span>📊</span> Haftalık Performans
                        </div>
                        <div class="kpi-trend">{trend_icon}</div>
                    </div>
                    <div class="kpi-value">{abs(week_change):.1f}%</div>
                    <div class="kpi-change {change_class}">
                        <span>{'↗' if week_change > 0 else '↘'}</span>
                        <span>{'+' if week_change > 0 else ''}{week_change:.2f}% Son haftadan beri</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
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
            
            with col3:
                range_pct = ((latest['High'] - latest['Low']) / latest['Low']) * 100
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <div class="kpi-title">
                            <span>📊</span> Günlük Aralık
                        </div>
                        <div class="kpi-trend">📊</div>
                    </div>
                    <div class="kpi-value">{range_pct:.1f}%</div>
                    <div class="kpi-change neutral">
                        <span>↕</span>
                        <span>₺{latest['Low']:.2f} - ₺{latest['High']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                price_class = "positive" if change > 0 else "negative"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <div class="kpi-title">
                            <span>💰</span> Mevcut Fiyat
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
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Charts Section
            st.markdown("""
            <div class="charts-grid">
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Dynamic chart title based on interval
                interval_names = {
                    "5m": "5 Dakika",
                    "15m": "15 Dakika", 
                    "1h": "1 Saat",
                    "2h": "2 Saat",
                    "4h": "4 Saat",
                    "1d": "Günlük"
                }
                
                period_names = {
                    "1mo": "son ay",
                    "3mo": "son 3 ay",
                    "6mo": "son 6 ay", 
                    "1y": "son yıl"
                }
                
                st.markdown(f"""
                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">Fiyat Hareketi - {interval_names.get(time_interval, time_interval)}</div>
                        <div class="chart-subtitle">Fiyat hareketi gösteriliyor: {period_names.get(period, period)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Technical analysis
                analyzer = TechnicalAnalyzer(df)
                analyzer.add_indicator('ema_21')
                analyzer.add_indicator('ema_50')
                
                # Create simplified chart
                fig = create_chart(df, analyzer, {'ema_21': True, 'ema_50': True})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">Aylık Performans</div>
                        <div class="chart-subtitle">{:+.1f}% son aydan beri</div>
                    </div>
                </div>
                """.format(month_change), unsafe_allow_html=True)
                
                # Volume chart
                volume_data = df['Volume'].tail(30)
                st.bar_chart(volume_data)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
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
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_ai_predictions():
    """AI tahminleri sayfası - Modern Shadcn stil"""
    st.markdown("""
    <div class="page-header-modern">
        <h1>🤖 AI Predictions</h1>
        <p>Machine learning powered price forecasting with confidence scores</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern card for stock selection
    st.markdown("""
    <div class="modern-card">
        <h3>Stock Selection</h3>
        <p>Choose a stock for AI-powered price prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock",
            options=list(BIST_SYMBOLS.keys()),
            format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
            key="ai_stock_select",
            label_visibility="collapsed"
        )
    
    with col2:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("AI model running..."):
            # Veri çek
            fetcher = BISTDataFetcher()
            data = fetcher.get_stock_data(selected_symbol, period="1y", interval="1d")
            
            if data is not None:
                # Teknik analiz
                analyzer = TechnicalAnalyzer(data)
                for indicator in ['rsi', 'ema_5', 'ema_8', 'ema_13', 'ema_21', 'vwap']:
                    analyzer.add_indicator(indicator)
                
                # ML tahmin
                ml_predictor = SimpleMLPredictor()
                prediction = ml_predictor.predict_next_day(data, analyzer.indicators)
                
                if 'error' not in prediction:
                    # Modern metrics grid
                    st.markdown("### 🎯 Prediction Results")
                    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">Current Price</div>
                            <div class="metric-value">₺{prediction['current_price']:.2f}</div>
                            <div class="metric-change neutral">Live Price</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        change_class = "positive" if prediction['predicted_return'] > 0 else "negative"
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">Predicted Price</div>
                            <div class="metric-value">₺{prediction['predicted_price']:.2f}</div>
                            <div class="metric-change {change_class}">{prediction['predicted_return']:+.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_class = "positive" if prediction['confidence'] > 0.7 else "negative" if prediction['confidence'] < 0.5 else "neutral"
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">Confidence</div>
                            <div class="metric-value">{prediction['confidence']:.0%}</div>
                            <div class="metric-change {confidence_class}">AI Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        signal_map = {'BUY': ('🚀', 'Buy Signal'), 'SELL': ('📉', 'Sell Signal'), 'HOLD': ('⏳', 'Hold')}
                        icon, text = signal_map.get(prediction['signal'], ('⏳', 'Hold'))
                        st.markdown(f"""
                        <div class="metric-card-modern">
                            <div class="metric-title">AI Signal</div>
                            <div class="metric-value">{icon}</div>
                            <div class="metric-change neutral">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feature importance
                    importance_df = ml_predictor.get_feature_importance()
                    if not importance_df.empty:
                        st.markdown("""
                        <div class="modern-card">
                            <h3>📊 Feature Importance</h3>
                            <p>Which factors the AI model considers most important</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.bar_chart(importance_df.set_index('feature')['importance'].head(8))
                else:
                    st.markdown(f"""
                    <div class="info-box-modern error">
                        <h4>❌ Prediction Error</h4>
                        <p>{prediction['error']}</p>
                    </div>
                    """, unsafe_allow_html=True)

def show_stock_screener():
    """Hisse tarayıcı sayfası"""
    st.markdown("""
    <div class="page-header">
        <h1>🔍 Hisse Tarayıcı</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin: 0;">Teknik kriterlere göre hisse taraması</p>
    </div>
    """, unsafe_allow_html=True)
    
    screener = StockScreener(BIST_SYMBOLS)
    
    # Zaman dilimi seçimi
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin-top: 0; color: #f39c12;">⏰ Zaman Dilimi</h3>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Tarama için zaman dilimi seçin</p>
    </div>
    """, unsafe_allow_html=True)
    
    time_intervals = {
        "1d": "1 Gün",
        "4h": "4 Saat", 
        "1h": "1 Saat",
        "30m": "30 Dakika",
        "15m": "15 Dakika",
        "5m": "5 Dakika"
    }
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        selected_interval = st.selectbox(
            "Zaman Dilimi", 
            list(time_intervals.keys()),
            format_func=lambda x: time_intervals[x],
            index=0,
            key="screener_interval"
        )
    
    with col2:
        st.markdown(f"""
        <div style="margin-top: 1.5rem; padding: 0.5rem; background: rgba(79, 172, 254, 0.1); border-radius: 8px; border: 1px solid rgba(79, 172, 254, 0.3);">
            <small style="color: #4facfe;">Seçili: {time_intervals[selected_interval]}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Tarama sekmeli yapısı
    tab1, tab2, tab3 = st.tabs(["🚀 Boğa Sinyalleri", "⚡ Teknik Taramalar", "📊 Genel Taramalar"])
    
    with tab1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin-top: 0; color: #4ecdc4;">🚀 Boğa Sinyalleri Taraması</h2>
            <p style="color: rgba(255,255,255,0.7);">Ana uygulamadaki boğa sinyallerini BIST hisselerinde tara</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Boğa sinyali seçimi
        signal_types = {
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_signal = st.selectbox("Sinyal Türü Seç", list(signal_types.keys()),
                                         format_func=lambda x: signal_types[x], key="signal_type")
        
        with col2:
            if st.button("🔍 Sinyal Taraması Yap", type="primary", key="bull_signal_scan"):
                with st.spinner(f"{signal_types[selected_signal]} sinyali aranıyor..."):
                    # Seçili sinyale göre tarama fonksiyonu çağır
                    if selected_signal == 'VWAP Bull Signal':
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
                    
                    if results:
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
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>⚠️ Sonuç Bulunamadı</h4>
                            <p>{signal_types[selected_signal]} kriteri karşılayan hisse bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Tüm sinyalleri tara butonu
        if st.button("🚀 Tüm Boğa Sinyallerini Tara", type="secondary", key="all_bull_signals"):
            with st.spinner("Tüm boğa sinyalleri taranıyor..."):
                all_results = screener.screen_all_bull_signals(selected_interval)
                
                # Her sinyal için sonuçları göster
                for signal_name, signal_results in all_results.items():
                    if signal_results:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin-top: 0; color: #45b7d1;">{signal_types[signal_name]}</h3>
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
    
    with tab2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin-top: 0; color: #4ecdc4;">📋 Teknik Tarama Kriterleri</h2>
            <p style="color: rgba(255,255,255,0.7);">Hisse senetlerini filtrelemek için kriterlerinizi seçin</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card hover-glow">
                <h3 style="margin-top: 0; color: #ff6b6b;">⚡ RSI Taraması</h3>
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
                <h3 style="margin-top: 0; color: #f39c12;">📊 Hacim Artışı</h3>
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
                <h3 style="margin-top: 0; color: #45b7d1;">🚀 Fiyat Kırılımları</h3>
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
    
    with tab3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin-top: 0; color: #4ecdc4;">🎯 Çoklu Kriter Taraması</h2>
            <p style="color: rgba(255,255,255,0.7);">Birden fazla kriteri birleştirerek tarama yapın</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 RSI Kriterleri")
            use_rsi = st.checkbox("RSI Filtresi Kullan", key="use_rsi_filter")
            if use_rsi:
                rsi_min_multi = st.slider("RSI Min", 0, 100, 30, key="rsi_min_multi")
                rsi_max_multi = st.slider("RSI Max", 0, 100, 70, key="rsi_max_multi")
        
        with col2:
            st.markdown("##### 📈 Fiyat Kriterleri")
            use_ema = st.checkbox("EMA Üstünde Fiyat", key="use_ema_filter")
            
            st.markdown("##### 📊 Hacim Kriterleri")
            use_volume = st.checkbox("Hacim Filtresi Kullan", key="use_volume_filter")
            if use_volume:
                min_volume_ratio = st.slider("Min Hacim Oranı", 1.0, 5.0, 1.2, 0.1, key="min_vol_ratio")
        
        if st.button("🔍 Çoklu Kriter Taraması Yap", type="primary", key="multi_criteria_scan"):
            criteria = {}
            
            if use_rsi:
                criteria['rsi_min'] = rsi_min_multi
                criteria['rsi_max'] = rsi_max_multi
            
            if use_ema:
                criteria['price_above_ema'] = True
            
            if use_volume:
                criteria['min_volume_ratio'] = min_volume_ratio
            
            if criteria:
                with st.spinner("Çoklu kriter taraması yapılıyor..."):
                    results = screener.screen_multi_criteria(criteria, selected_interval)
                    if results:
                        st.markdown("""
                        <div class="info-box">
                            <h4>✅ Çoklu Kriter Tarama Sonuçları</h4>
                            <p>{} hisse bulundu</p>
                        </div>
                        """.format(len(results)), unsafe_allow_html=True)
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Sonuç Bulunamadı</h4>
                            <p>Belirtilen kriterleri karşılayan hisse bulunamadı</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Lütfen en az bir kriter seçin.")

def show_pattern_analysis():
    """Pattern analizi sayfası"""
    st.markdown("""
    <div class="page-header">
        <h1>👁️ Mum Formasyonu Analizi</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin: 0;">Candlestick pattern tespiti ve sinyal analizi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hisse seçimi modern kart içinde
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin-top: 0; color: #4ecdc4;">📊 Hisse Seçimi</h3>
        <p style="color: rgba(255,255,255,0.7);">Analiz edilecek hisseyi seçin</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_symbol = st.selectbox(
        "Hisse Seçin",
        options=list(BIST_SYMBOLS.keys()),
        format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
        key="pattern_stock_select"
    )
    
    if st.button("🔍 Pattern Analizi Yap", type="primary"):
        with st.spinner("Formasyonlar analiz ediliyor..."):
            # Veri çek
            fetcher = BISTDataFetcher()
            data = fetcher.get_stock_data(selected_symbol, period="1y", interval="1d")
            
            if data is not None:
                # Pattern recognition
                pattern_analyzer = PatternRecognition(data)
                patterns = pattern_analyzer.analyze_all_patterns()
                latest_patterns = pattern_analyzer.get_latest_patterns()
                signals = pattern_analyzer.get_pattern_signals()
                
                # Sonuçları modern kartlarda göster
                st.markdown("""
                <div class="metric-card" style="margin-top: 2rem;">
                    <h2 style="margin-top: 0; color: #45b7d1;">🕯️ Tespit Edilen Formasyonlar</h2>
                    <p style="color: rgba(255,255,255,0.7);">Son işlem gününde tespit edilen candlestick patternleri</p>
                </div>
                """, unsafe_allow_html=True)
                
                pattern_cols = st.columns(4)
                pattern_names = {
                    'doji': '⭐ Doji',
                    'hammer': '🔨 Çekiç',
                    'shooting_star': '⭐ Kayan Yıldız',
                    'bullish_engulfing': '🟢 Yükseliş Saran',
                    'bearish_engulfing': '🔴 Düşüş Saran',
                    'morning_star': '🌅 Sabah Yıldızı',
                    'evening_star': '🌆 Akşam Yıldızı'
                }
                
                for i, (pattern, detected) in enumerate(latest_patterns.items()):
                    with pattern_cols[i % 4]:
                        if detected:
                            st.markdown(f"""
                            <div class="metric-card hover-glow" style="border: 2px solid #00ff88;">
                                <h4 style="margin: 0; color: white;">{pattern_names.get(pattern, pattern)}</h4>
                                <div style="text-align: center; margin: 1rem 0;">
                                    <span class="status-positive">✅ TESPİT EDİLDİ</span>
                                </div>
                                <p style="margin: 0; color: #00ff88; font-weight: bold; text-align: center;">AKTIF</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin: 0; color: white;">{pattern_names.get(pattern, pattern)}</h4>
                                <div style="text-align: center; margin: 1rem 0;">
                                    <span class="status-neutral">❌ TESPİT EDİLMEDİ</span>
                                </div>
                                <p style="margin: 0; color: rgba(255,255,255,0.5); text-align: center;">PASİF</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Sinyaller
                if signals:
                    st.markdown("""
                    <div class="metric-card" style="margin-top: 2rem;">
                        <h2 style="margin-top: 0; color: #ff6b6b;">📈 Pattern Sinyalleri</h2>
                        <p style="color: rgba(255,255,255,0.7);">Tespit edilen patternlerden çıkarılan işlem sinyalleri</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    signal_cols = st.columns(len(signals) if len(signals) <= 4 else 4)
                    for i, (pattern, signal) in enumerate(signals.items()):
                        with signal_cols[i % 4]:
                            signal_color = "#00ff88" if signal == "BUY" else "#ff4757"
                            signal_text = "ALIM" if signal == "BUY" else "SATIM"
                            signal_icon = "🚀" if signal == "BUY" else "📉"
                            
                            st.markdown(f"""
                            <div class="metric-card hover-glow">
                                <h4 style="margin: 0; color: white;">{pattern_names.get(pattern, pattern)}</h4>
                                <h2 style="margin: 0.5rem 0; color: {signal_color};">{signal_icon} {signal_text}</h2>
                                <span class="status-badge" style="background: {signal_color};">{signal}</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        <h4>ℹ️ Bilgi</h4>
                        <p>Şu anda aktif pattern sinyali bulunmamaktadır.</p>
                    </div>
                    """, unsafe_allow_html=True)

def show_risk_management():
    """Risk yönetimi sayfası"""
    st.markdown("""
    <div class="page-header">
        <h1>🛡️ Risk Yönetimi</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin: 0;">Pozisyon büyüklüğü ve risk hesaplama araçları</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hesap bilgileri modern kart içinde
    st.markdown("""
    <div class="metric-card">
        <h2 style="margin-top: 0; color: #4ecdc4;">💰 Hesap Bilgileri</h2>
        <p style="color: rgba(255,255,255,0.7);">Risk hesaplaması için gerekli bilgileri girin</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #45b7d1;">📊 Hesap Ayarları</h3>
        """, unsafe_allow_html=True)
        
        account_balance = st.number_input("Hesap Bakiyesi (₺)", value=100000, min_value=1000)
        risk_percentage = st.slider("Risk Yüzdesi (%)", 1.0, 10.0, 2.0, 0.5)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #f39c12;">⚡ İşlem Parametreleri</h3>
        """, unsafe_allow_html=True)
        
        entry_price = st.number_input("Giriş Fiyatı (₺)", value=10.0, min_value=0.1)
        stop_loss_price = st.number_input("Stop Loss Fiyatı (₺)", value=9.0, min_value=0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk hesaplayıcı
    risk_calc = RiskCalculator(account_balance)
    
    if st.button("📊 Risk Analizi Yap", type="primary"):
        # Pozisyon büyüklüğü
        position_calc = risk_calc.calculate_position_size(entry_price, stop_loss_price, risk_percentage)
        
        if 'error' not in position_calc:
            st.markdown("""
            <div class="metric-card" style="margin-top: 2rem;">
                <h2 style="margin-top: 0; color: #00ff88;">📈 Pozisyon Analizi</h2>
                <p style="color: rgba(255,255,255,0.7);">Hesaplanan pozisyon detayları</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card hover-glow">
                    <h4 style="margin: 0; color: white;">📊 Alınacak Hisse</h4>
                    <h2 style="margin: 0.5rem 0; color: #4ecdc4;">{position_calc['shares']:,}</h2>
                    <p style="margin: 0; color: rgba(255,255,255,0.7);">Adet</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card hover-glow">
                    <h4 style="margin: 0; color: white;">💰 Yatırım Tutarı</h4>
                    <h2 style="margin: 0.5rem 0; color: #45b7d1;">₺{position_calc['total_investment']:,.0f}</h2>
                    <p style="margin: 0; color: rgba(255,255,255,0.7);">Toplam</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card hover-glow">
                    <h4 style="margin: 0; color: white;">⚠️ Risk Tutarı</h4>
                    <h2 style="margin: 0.5rem 0; color: #ff4757;">₺{position_calc['risk_amount']:,.0f}</h2>
                    <p style="margin: 0; color: rgba(255,255,255,0.7);">Maksimum kayıp</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                portfolio_color = "#00ff88" if position_calc['portfolio_percentage'] <= 20 else "#f39c12" if position_calc['portfolio_percentage'] <= 50 else "#ff4757"
                st.markdown(f"""
                <div class="metric-card hover-glow">
                    <h4 style="margin: 0; color: white;">📈 Portföy Oranı</h4>
                    <h2 style="margin: 0.5rem 0; color: {portfolio_color};">{position_calc['portfolio_percentage']:.1f}%</h2>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {min(position_calc['portfolio_percentage'], 100):.1f}%; background: {portfolio_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Hedef fiyatlar
            targets = risk_calc.calculate_target_prices(entry_price, stop_loss_price)
            
            st.markdown("""
            <div class="metric-card" style="margin-top: 2rem;">
                <h2 style="margin-top: 0; color: #f9ca24;">🎯 Hedef Fiyatlar</h2>
                <p style="color: rgba(255,255,255,0.7);">Risk/Reward oranına göre hesaplanan hedefler</p>
            </div>
            """, unsafe_allow_html=True)
            
            target_data = []
            for target_name, data in targets.items():
                target_data.append({
                    'Hedef': target_name,
                    'Fiyat': f"₺{data['price']:.2f}",
                    'Kar': f"₺{data['profit_amount']:.2f}",
                    'Kar %': f"{data['profit_percentage']:.1f}%",
                    'Risk/Reward': f"1:{data['risk_reward_ratio']}"
                })
            
            target_df = pd.DataFrame(target_data)
            st.dataframe(target_df, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Hesaplama Hatası</h4>
                <p>{position_calc['error']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_sentiment_analysis():
    """Sentiment analizi sayfası"""
    st.markdown("""
    <div class="page-header">
        <h1>💭 Duygu Analizi</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin: 0;">Haber ve sosyal medya sentiment analizi</p>
    </div>
    """, unsafe_allow_html=True)
    
    sentiment_analyzer = SentimentAnalyzer()
    
    # Hisse seçimi modern kart içinde
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin-top: 0; color: #4ecdc4;">📊 Hisse Seçimi</h3>
        <p style="color: rgba(255,255,255,0.7);">Sentiment analizi yapılacak hisseyi seçin</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_symbol = st.selectbox(
        "Hisse Seçin",
        options=list(BIST_SYMBOLS.keys()),
        format_func=lambda x: f"{x} - {BIST_SYMBOLS[x]}",
        key="sentiment_stock_select"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card hover-glow">
            <h3 style="margin-top: 0; color: #ff6b6b;">📰 Hisse Sentiment Analizi</h3>
            <p style="color: rgba(255,255,255,0.7);">Belirli hisse için sentiment analizi</p>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Hisse Sentiment Analizi", type="primary", key="stock_sentiment"):
            with st.spinner("Sentiment analiz ediliyor..."):
                sentiment = sentiment_analyzer.get_basic_sentiment_score(selected_symbol)
                social_sentiment = sentiment_analyzer.analyze_social_media_sentiment(selected_symbol)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Sonuçları modern kartlarda göster
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h2 style="margin-top: 0; color: #45b7d1;">📈 {selected_symbol} Sentiment Analizi</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_color = "#00ff88" if sentiment['sentiment_score'] > 0.1 else "#ff4757" if sentiment['sentiment_score'] < -0.1 else "#f39c12"
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">📰 Haber Sentiment</h4>
                        <h2 style="margin: 0.5rem 0; color: {sentiment_color};">{sentiment['sentiment_score']:.2f}</h2>
                        <span class="status-badge" style="background: {sentiment_color};">{sentiment['sentiment_label']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    social_color = "#00ff88" if social_sentiment['social_sentiment'] > 0.1 else "#ff4757" if social_sentiment['social_sentiment'] < -0.1 else "#f39c12"
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">🐦 Sosyal Medya</h4>
                        <h2 style="margin: 0.5rem 0; color: {social_color};">{social_sentiment['social_sentiment']:.2f}</h2>
                        <span class="status-badge" style="background: {social_color};">{social_sentiment['social_label']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">📊 İstatistikler</h4>
                        <p style="margin: 0.5rem 0; color: #4ecdc4;">Bahsedilme: {social_sentiment['mention_count']:,}</p>
                        <span class="status-badge status-neutral">{social_sentiment['trending_status']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Haber başlıkları
                news = sentiment_analyzer.get_news_headlines(selected_symbol)
                if news:
                    st.markdown("""
                    <div class="metric-card" style="margin-top: 2rem;">
                        <h3 style="margin-top: 0; color: #f9ca24;">📰 Son Haberler</h3>
                        <p style="color: rgba(255,255,255,0.7);">Son haberlerin sentiment skorları</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for item in news:
                        sentiment_color = "#00ff88" if item['sentiment'] > 0.1 else "#ff4757" if item['sentiment'] < -0.1 else "#f39c12"
                        st.markdown(f"""
                        <div class="info-box">
                            <h5 style="margin: 0; color: white;">{item['date']}</h5>
                            <p style="margin: 0.5rem 0;">{item['headline']}</p>
                            <span class="status-badge" style="background: {sentiment_color};">{item['sentiment']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card hover-glow">
            <h3 style="margin-top: 0; color: #45b7d1;">🌍 Piyasa Sentiment Analizi</h3>
            <p style="color: rgba(255,255,255,0.7);">Genel piyasa sentiment durumu</p>
        """, unsafe_allow_html=True)
        
        if st.button("🌍 Piyasa Sentiment Analizi", key="market_sentiment"):
            with st.spinner("Piyasa sentiment analiz ediliyor..."):
                market_sentiment = sentiment_analyzer.get_market_sentiment(list(BIST_SYMBOLS.keys())[:10])
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h2 style="margin-top: 0; color: #6c5ce7;">🌐 Piyasa Genel Sentiment</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    market_color = "#00ff88" if market_sentiment['market_sentiment'] > 0.1 else "#ff4757" if market_sentiment['market_sentiment'] < -0.1 else "#f39c12"
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">🌍 Genel Sentiment</h4>
                        <h2 style="margin: 0.5rem 0; color: {market_color};">{market_sentiment['market_sentiment']:.2f}</h2>
                        <span class="status-badge" style="background: {market_color};">{market_sentiment['market_label']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">🟢 Pozitif Hisse</h4>
                        <h2 style="margin: 0.5rem 0; color: #00ff88;">{market_sentiment['positive_stocks']}</h2>
                        <p style="margin: 0; color: rgba(255,255,255,0.7);">Adet</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">🔴 Negatif Hisse</h4>
                        <h2 style="margin: 0.5rem 0; color: #ff4757;">{market_sentiment['negative_stocks']}</h2>
                        <p style="margin: 0; color: rgba(255,255,255,0.7);">Adet</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card hover-glow">
                        <h4 style="margin: 0; color: white;">🟡 Nötr Hisse</h4>
                        <h2 style="margin: 0.5rem 0; color: #f39c12;">{market_sentiment['neutral_stocks']}</h2>
                        <p style="margin: 0; color: rgba(255,255,255,0.7);">Adet</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 