import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

class PortfolioManager:
    """Portföy yönetimi ve takip sistemi"""
    
    def __init__(self, portfolio_file: str = "portfolio_data.json"):
        self.portfolio_file = portfolio_file
        self.holdings = {}
        self.transactions = []
        self.portfolio_history = []
        self.load_portfolio()
    
    def load_portfolio(self) -> None:
        """Portföy verilerini dosyadan yükler"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.holdings = data.get('holdings', {})
                    self.transactions = data.get('transactions', [])
                    self.portfolio_history = data.get('portfolio_history', [])
            except Exception as e:
                print(f"Portföy verisi yüklenirken hata: {e}")
                self.holdings = {}
                self.transactions = []
                self.portfolio_history = []
    
    def save_portfolio(self) -> None:
        """Portföy verilerini dosyaya kaydeder"""
        data = {
            'holdings': self.holdings,
            'transactions': self.transactions,
            'portfolio_history': self.portfolio_history
        }
        try:
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Portföy verisi kaydedilirken hata: {e}")
    
    def add_transaction(self, symbol: str, transaction_type: str, quantity: float, 
                       price: float, date: str = None, commission: float = 0) -> None:
        """Yeni işlem ekler (BUY/SELL)"""
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        transaction = {
            'symbol': symbol,
            'type': transaction_type.upper(),
            'quantity': quantity,
            'price': price,
            'date': date,
            'commission': commission,
            'total_cost': quantity * price + commission
        }
        
        self.transactions.append(transaction)
        
        # Holdings güncelle
        if symbol not in self.holdings:
            self.holdings[symbol] = {
                'quantity': 0,
                'avg_cost': 0,
                'total_cost': 0,
                'first_buy_date': date
            }
        
        if transaction_type.upper() == 'BUY':
            # Alım işlemi
            old_quantity = self.holdings[symbol]['quantity']
            old_total_cost = self.holdings[symbol]['total_cost']
            
            new_quantity = old_quantity + quantity
            new_total_cost = old_total_cost + transaction['total_cost']
            
            self.holdings[symbol]['quantity'] = new_quantity
            self.holdings[symbol]['total_cost'] = new_total_cost
            self.holdings[symbol]['avg_cost'] = new_total_cost / new_quantity if new_quantity > 0 else 0
            
        elif transaction_type.upper() == 'SELL':
            # Satım işlemi
            if self.holdings[symbol]['quantity'] >= quantity:
                # Ortalama maliyet aynı kalır, sadece miktar azalır
                self.holdings[symbol]['quantity'] -= quantity
                
                # Eğer tüm pozisyon kapatıldıysa
                if self.holdings[symbol]['quantity'] == 0:
                    self.holdings[symbol]['avg_cost'] = 0
                    self.holdings[symbol]['total_cost'] = 0
            else:
                raise ValueError(f"Yetersiz hisse miktarı. Mevcut: {self.holdings[symbol]['quantity']}, Satış: {quantity}")
        
        self.save_portfolio()
    
    def get_current_portfolio(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """Mevcut portföy durumunu döner"""
        portfolio_data = []
        
        for symbol, holding in self.holdings.items():
            if holding['quantity'] > 0:
                current_price = current_prices.get(symbol, 0)
                current_value = holding['quantity'] * current_price
                total_cost = holding['total_cost']
                
                pnl = current_value - total_cost
                pnl_percent = (pnl / total_cost * 100) if total_cost > 0 else 0
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Quantity': holding['quantity'],
                    'Avg_Cost': holding['avg_cost'],
                    'Current_Price': current_price,
                    'Total_Cost': total_cost,
                    'Current_Value': current_value,
                    'PnL': pnl,
                    'PnL_Percent': pnl_percent,
                    'Weight': 0  # Daha sonra hesaplanacak
                })
        
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            total_value = df['Current_Value'].sum()
            df['Weight'] = (df['Current_Value'] / total_value * 100) if total_value > 0 else 0
            return df
        
        return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Portföy metriklerini hesaplar"""
        portfolio_df = self.get_current_portfolio(current_prices)
        
        if portfolio_df.empty:
            return {
                'total_value': 0,
                'total_cost': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'num_positions': 0,
                'largest_position': None,
                'best_performer': None,
                'worst_performer': None
            }
        
        total_value = portfolio_df['Current_Value'].sum()
        total_cost = portfolio_df['Total_Cost'].sum()
        total_pnl = portfolio_df['PnL'].sum()
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # En büyük pozisyon
        largest_position = portfolio_df.loc[portfolio_df['Weight'].idxmax(), 'Symbol']
        
        # En iyi ve en kötü performans
        best_performer = portfolio_df.loc[portfolio_df['PnL_Percent'].idxmax(), 'Symbol']
        worst_performer = portfolio_df.loc[portfolio_df['PnL_Percent'].idxmin(), 'Symbol']
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'num_positions': len(portfolio_df),
            'largest_position': largest_position,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'average_gain': portfolio_df['PnL_Percent'].mean(),
            'positions_in_profit': len(portfolio_df[portfolio_df['PnL'] > 0]),
            'positions_in_loss': len(portfolio_df[portfolio_df['PnL'] < 0])
        }
    
    def calculate_risk_metrics(self, price_history: Dict[str, pd.DataFrame], 
                              current_prices: Dict[str, float]) -> Dict:
        """Risk metriklerini hesaplar"""
        portfolio_df = self.get_current_portfolio(current_prices)
        
        if portfolio_df.empty:
            return {}
        
        # Portföy getirileri hesapla
        portfolio_returns = []
        symbols = portfolio_df['Symbol'].tolist()
        weights = portfolio_df['Weight'].values / 100
        
        # Her sembol için getiri serisi al
        return_series = {}
        for symbol in symbols:
            if symbol in price_history:
                prices = price_history[symbol]['Close']
                returns = prices.pct_change().dropna()
                return_series[symbol] = returns
        
        if not return_series:
            return {}
        
        # Ortak tarih aralığını bul
        common_dates = None
        for symbol, returns in return_series.items():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)
        
        if len(common_dates) < 30:  # En az 30 gün veri
            return {}
        
        # Portföy getirilerini hesapla
        portfolio_returns = pd.Series(0, index=common_dates)
        for i, symbol in enumerate(symbols):
            if symbol in return_series:
                symbol_returns = return_series[symbol].loc[common_dates]
                portfolio_returns += weights[i] * symbol_returns
        
        # Risk metrikleri
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # VaR (Value at Risk) %95 güven aralığı
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Beta (BIST 100'e göre) - burada basitleştirildi
        beta = 1.0  # Gerçek uygulamada BIST 100 verisi ile hesaplanmalı
        
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95 * 100,
            'var_99': var_99 * 100,
            'max_drawdown': max_drawdown * 100,
            'beta': beta,
            'correlation_matrix': self._calculate_correlation_matrix(return_series, symbols),
            'diversification_ratio': self._calculate_diversification_ratio(portfolio_df)
        }
    
    def _calculate_correlation_matrix(self, return_series: Dict, symbols: List[str]) -> pd.DataFrame:
        """Korelasyon matrisini hesaplar"""
        correlation_data = {}
        
        for symbol in symbols:
            if symbol in return_series:
                correlation_data[symbol] = return_series[symbol]
        
        if correlation_data:
            correlation_df = pd.DataFrame(correlation_data)
            return correlation_df.corr()
        
        return pd.DataFrame()
    
    def _calculate_diversification_ratio(self, portfolio_df: pd.DataFrame) -> float:
        """Diversifikasyon oranını hesaplar"""
        if len(portfolio_df) <= 1:
            return 0.0
        
        # Herfindahl-Hirschman Index (HHI) kullanarak diversifikasyon ölçer
        weights = portfolio_df['Weight'].values / 100
        hhi = np.sum(weights ** 2)
        
        # Diversifikasyon oranı (0-1 arası, 1 mükemmel diversifikasyon)
        max_diversification = 1 / len(portfolio_df)
        diversification_ratio = (1/hhi - 1) / (1/max_diversification - 1) if len(portfolio_df) > 1 else 0
        
        return min(1.0, max(0.0, diversification_ratio))
    
    def get_sector_allocation(self, sector_mapping: Dict[str, str]) -> Dict[str, float]:
        """Sektör dağılımını hesaplar"""
        portfolio_df = self.get_current_portfolio({})  # Current prices verilmeyecek
        
        if portfolio_df.empty:
            return {}
        
        sector_values = {}
        total_value = portfolio_df['Current_Value'].sum()
        
        for _, row in portfolio_df.iterrows():
            symbol = row['Symbol']
            sector = sector_mapping.get(symbol, 'Unknown')
            
            if sector not in sector_values:
                sector_values[sector] = 0
            
            sector_values[sector] += row['Current_Value']
        
        # Yüzde olarak dönüştür
        sector_percentages = {sector: (value / total_value * 100) 
                            for sector, value in sector_values.items()}
        
        return sector_percentages
    
    def suggest_rebalancing(self, target_weights: Dict[str, float], 
                          current_prices: Dict[str, float]) -> Dict:
        """Portföy yeniden dengeleme önerileri"""
        portfolio_df = self.get_current_portfolio(current_prices)
        
        if portfolio_df.empty:
            return {}
        
        current_weights = dict(zip(portfolio_df['Symbol'], portfolio_df['Weight']))
        rebalancing_suggestions = {}
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            difference = target_weight - current_weight
            
            if abs(difference) > 1:  # %1'den fazla fark varsa
                if symbol in current_prices:
                    total_value = portfolio_df['Current_Value'].sum()
                    target_value = total_value * target_weight / 100
                    current_value = total_value * current_weight / 100
                    
                    value_difference = target_value - current_value
                    quantity_change = value_difference / current_prices[symbol]
                    
                    rebalancing_suggestions[symbol] = {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_difference': difference,
                        'value_difference': value_difference,
                        'quantity_change': quantity_change,
                        'action': 'BUY' if quantity_change > 0 else 'SELL'
                    }
        
        return rebalancing_suggestions
    
    def get_transaction_history(self, symbol: str = None, 
                               start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """İşlem geçmişini filtreli olarak döner"""
        transactions = self.transactions.copy()
        
        if symbol:
            transactions = [t for t in transactions if t['symbol'] == symbol]
        
        if start_date:
            transactions = [t for t in transactions if t['date'] >= start_date]
            
        if end_date:
            transactions = [t for t in transactions if t['date'] <= end_date]
        
        if transactions:
            return pd.DataFrame(transactions)
        
        return pd.DataFrame()
    
    def calculate_position_sizing(self, symbol: str, risk_per_trade: float, 
                                stop_loss_percent: float, account_balance: float) -> Dict:
        """Pozisyon büyüklüğü hesaplar"""
        
        # Risk miktarı (TL cinsinden)
        risk_amount = account_balance * (risk_per_trade / 100)
        
        # Stop loss mesafesi
        stop_loss_distance = stop_loss_percent / 100
        
        # Pozisyon büyüklüğü
        position_size = risk_amount / stop_loss_distance
        
        return {
            'risk_amount': risk_amount,
            'stop_loss_distance': stop_loss_distance * 100,
            'position_size': position_size,
            'max_shares': int(position_size),
            'risk_per_share': stop_loss_distance
        }
    
    def update_portfolio_history(self, current_prices: Dict[str, float]) -> None:
        """Portföy geçmişini günceller"""
        metrics = self.calculate_portfolio_metrics(current_prices)
        
        history_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_value': metrics['total_value'],
            'total_pnl': metrics['total_pnl'],
            'total_pnl_percent': metrics['total_pnl_percent'],
            'num_positions': metrics['num_positions']
        }
        
        # Aynı günün verisi varsa güncelle
        today = history_entry['date']
        existing_entry = next((entry for entry in self.portfolio_history 
                              if entry['date'] == today), None)
        
        if existing_entry:
            existing_entry.update(history_entry)
        else:
            self.portfolio_history.append(history_entry)
        
        # Son 365 gün tutulsun
        if len(self.portfolio_history) > 365:
            self.portfolio_history = self.portfolio_history[-365:]
        
        self.save_portfolio()
    
    def get_portfolio_performance_chart_data(self) -> pd.DataFrame:
        """Portföy performans grafiği için veri hazırlar"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df 