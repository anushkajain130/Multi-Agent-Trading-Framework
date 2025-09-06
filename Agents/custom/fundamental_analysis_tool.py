def analyze_fundamental_indicators(symbol, lookback_years=5):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from scipy import stats
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        quarterly_financials = stock.quarterly_financials
        
        # Profitability Ratios
        profitability_metrics = {
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "roic": info.get("returnOnCapital")
        }
        
        # Valuation Ratios
        valuation_metrics = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda")
        }
        
        # Liquidity Ratios
        liquidity_metrics = {
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "cash_ratio": None  # Calculate if data available
        }
        
        # Debt Ratios
        debt_metrics = {
            "debt_to_equity": info.get("debtToEquity"),
            "total_debt": info.get("totalDebt"),
            "long_term_debt": balance_sheet.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in balance_sheet.index else None,
            "interest_coverage": None  # Calculate if data available
        }
        
        # Growth Metrics
        growth_metrics = {
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            "dividend_growth": None  # Calculate if applicable
        }
        
        # Dividend Metrics
        dividend_metrics = {
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "dividend_rate": info.get("dividendRate"),
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield")
        }
        
        # Efficiency Metrics
        efficiency_metrics = {
            "asset_turnover": info.get("assetTurnover") if "assetTurnover" in info else None,
            "inventory_turnover": info.get("inventoryTurnover") if "inventoryTurnover" in info else None,
            "receivables_turnover": None  # Calculate if data available
        }
        
        # Calculate DCF-based intrinsic value
        try:
            fcf = cash_flow.loc["Free Cash Flow"].mean() if "Free Cash Flow" in cash_flow.index else None
            if fcf and fcf > 0:
                growth_rate = 0.05
                discount_rate = 0.10
                years = 5
                
                future_cash_flows = []
                for i in range(1, years + 1):
                    fcf_future = fcf * (1 + growth_rate) ** i
                    discounted_fcf = fcf_future / (1 + discount_rate) ** i
                    future_cash_flows.append(discounted_fcf)
                
                terminal_value = (future_cash_flows[-1] * (1 + growth_rate)) / (discount_rate - growth_rate)
                discounted_terminal_value = terminal_value / (1 + discount_rate) ** years
                intrinsic_value = sum(future_cash_flows) + discounted_terminal_value
                
                shares_outstanding = info.get("sharesOutstanding", 0)
                intrinsic_value_per_share = intrinsic_value / shares_outstanding if shares_outstanding > 0 else None
            else:
                intrinsic_value_per_share = None
        except:
            intrinsic_value_per_share = None
        
        # Get analyst recommendations
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            try:
                # Try to get the latest recommendation grade
                latest_recommendation = recommendations.iloc[-1].get("To Grade", None)
                if latest_recommendation is None:
                    # Try alternative column names that might exist
                    possible_columns = ["Recommendation", "Rating", "Action"]
                    for col in possible_columns:
                        if col in recommendations.columns:
                            latest_recommendation = recommendations.iloc[-1][col]
                            break
                if latest_recommendation is None:
                    latest_recommendation = "No specific grade available"
            except Exception as e:
                print(f"Warning: Could not parse recommendation for {symbol}: {str(e)}")
                latest_recommendation = "Not available"
            total_recommendations = len(recommendations)
        else:
            latest_recommendation = "No recommendations available"
            total_recommendations = 0
        
        # Compile all metrics
        fundamental_indicators = {
            "symbol": symbol,
            "analysis_date": pd.Timestamp.now().strftime('%Y-%m-%d'),
            "company_info": {
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice")
            },
            "profitability_metrics": profitability_metrics,
            "valuation_metrics": valuation_metrics,
            "liquidity_metrics": liquidity_metrics,
            "debt_metrics": debt_metrics,
            "growth_metrics": growth_metrics,
            "dividend_metrics": dividend_metrics,
            "efficiency_metrics": efficiency_metrics,
            "valuation_analysis": {
                "intrinsic_value": intrinsic_value_per_share,
                "current_price": info.get("currentPrice"),
                "value_difference": (info.get("currentPrice") - intrinsic_value_per_share) if intrinsic_value_per_share else None,
                "dcf_assumptions": {
                    "growth_rate": growth_rate if 'growth_rate' in locals() else None,
                    "discount_rate": discount_rate if 'discount_rate' in locals() else None,
                    "forecast_years": years if 'years' in locals() else None
                }
            },
            "analyst_recommendations": {
                "latest": latest_recommendation,
                "total_count": total_recommendations
            }
        }
        
        # Determine overall financial health
        health_score = 0
        total_metrics = 0
        
        # Profitability assessment
        if profitability_metrics["roe"]:
            health_score += 1 if profitability_metrics["roe"] > 0.15 else 0
            total_metrics += 1
        if profitability_metrics["operating_margin"]:
            health_score += 1 if profitability_metrics["operating_margin"] > 0.15 else 0
            total_metrics += 1
            
        # Liquidity assessment
        if liquidity_metrics["current_ratio"]:
            health_score += 1 if liquidity_metrics["current_ratio"] > 1.5 else 0
            total_metrics += 1
            
        # Debt assessment
        if debt_metrics["debt_to_equity"]:
            health_score += 1 if debt_metrics["debt_to_equity"] < 1.0 else 0
            total_metrics += 1
            
        # Growth assessment
        if growth_metrics["revenue_growth"]:
            health_score += 1 if growth_metrics["revenue_growth"] > 0.05 else 0
            total_metrics += 1
            
        # Calculate financial health score
        if total_metrics > 0:
            health_percentage = (health_score / total_metrics) * 100
            if health_percentage >= 80:
                financial_health = "Strong"
            elif health_percentage >= 60:
                financial_health = "Good"
            elif health_percentage >= 40:
                financial_health = "Moderate"
            else:
                financial_health = "Weak"
        else:
            financial_health = "Insufficient Data"
            
        fundamental_indicators["overall_health"] = {
            "status": financial_health,
            "score": health_score,
            "total_metrics": total_metrics,
            "score_percentage": (health_score / total_metrics * 100) if total_metrics > 0 else None
        }
        
        print(f"Fundamental analysis completed for {symbol}")
        return fundamental_indicators
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

if __name__ == "__main__":
    symbol = "AAPL"
    indicators = analyze_fundamental_indicators(symbol)
    
    if indicators:
        print(f"\nFundamental Analysis for {symbol}")
        print(f"Overall Financial Health: {indicators['overall_health']['status']}")
        print(f"Current Price: ${indicators['company_info']['current_price']:.2f}")
        print(f"P/E Ratio: {indicators['valuation_metrics']['pe_ratio']:.2f}")
        print(f"ROE: {indicators['profitability_metrics']['roe']*100:.2f}%")
        print(f"Debt to Equity: {indicators['debt_metrics']['debt_to_equity']:.2f}")
