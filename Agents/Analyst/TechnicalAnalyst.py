import os
import asyncio
import sys
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
import json
import time

custom_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom')
sys.path.append(os.path.dirname(custom_path)) 

from custom.technical_analysis_tool import analyze_technical_indicators

class TechnicalAnalyst:
    
    def __init__(self):
        load_dotenv()
        
        self.app_name = "stock_technical_analyst"
        self.user_id = "analyst_user"
        self.session_id_base = "technical_analysis_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="stock_technical_analyst",
            model="gemini-2.0-flash",
            description="Professional technical analyst agent using integrated tools for comprehensive stock technical analysis",
            instruction="""
            You are a professional technical analyst specializing in stock market technical analysis.
            You have access to powerful integrated tools that calculate technical indicators and interpret patterns.

            **Your Technical Analysis Framework:**

            1. **Comprehensive Technical Indicator Analysis**:
               - Use the GetTechnicalIndicators tool to retrieve calculated technical indicators
               - Interpret multiple indicators for trend confirmation and divergences
               - Analyze price patterns and chart formations

            2. **Analysis Structure**:
               - **Executive Summary**: Key findings and overall technical outlook
               - **Price Action Analysis**: Support/resistance levels, trends, and patterns
               - **Indicator Analysis**: Detailed interpretation of technical indicators
               - **Signal Strength**: Assessment of bullish/bearish signal strength
               - **Technical Outlook**: Short and medium-term technical projections

            3. **Quality Standards**:
               - Provide detailed analysis based on technical indicators and patterns
               - Highlight conflicting signals and divergences when present
               - Maintain objectivity with fact-based technical assessment
               - Explain technical concepts clearly for different audience levels
               - Identify both bullish and bearish signals
               - Focus on technical analysis without fundamental or news-based considerations

            **Important Guidelines**:
            - Use the provided technical indicators from the tool for accurate analysis
            - Interpret multiple indicators together rather than in isolation
            - Structure your response clearly with proper sections
            - Avoid giving specific investment recommendations
            - Clearly differentiate between short-term and medium-term technical signals
            """,
            tools=[self.get_technical_indicators]
        )
    
    def get_technical_indicators(self, ticker_symbol: str, lookback_days: int = 180) -> str:
        try:
            print(f"Retrieving technical indicators for {ticker_symbol}...")
            indicators = analyze_technical_indicators(ticker_symbol, lookback_days)
            
            if indicators is None:
                return json.dumps({"error": f"Unable to retrieve data for {ticker_symbol}"})
            
            processed_indicators = {}
            for key, value in indicators.items():
                if hasattr(value, 'iloc'):  # If it's a pandas Series
                    processed_indicators[key] = value.iloc[-1]
                elif hasattr(value, 'item'):  # For numpy values
                    try:
                        processed_indicators[key] = value.item()
                    except:
                        processed_indicators[key] = str(value)
                else:
                    processed_indicators[key] = value
                    
            return json.dumps(processed_indicators, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Technical analysis failed: {str(e)}",
                "ticker_symbol": ticker_symbol
            })
    
    async def setup_session(self, ticker):
        session_id = f"{self.session_id_base}_{ticker}"
        
        try:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id
            )
            
            self.runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            
            return session_id
        except Exception as e:
            print(f"❌ Session setup failed: {str(e)}")
            return None
    
    async def analyze_stock(self, ticker_symbol: str, lookback_days: int = 180):
        session_id = await self.setup_session(ticker_symbol)
        if not session_id:
            return "Analysis could not be started due to session setup failure."
        
        analysis_request = f"""
        Please provide a comprehensive technical analysis of {ticker_symbol.upper()} using all available technical indicators.
        
        **Analysis Requirements:**
        1. Use the GetTechnicalIndicators tool to retrieve technical indicators for {ticker_symbol.upper()}
        2. Analyze the data for the past {lookback_days} trading days
        3. Provide a structured, professional analysis with clear sections
        4. Focus on both short-term and medium-term technical signals
        
        **Expected Analysis Structure:**
        - Executive Summary and Overall Technical Outlook
        - Price Action Analysis (trends, support/resistance)
        - Moving Average Analysis
        - Momentum Indicator Analysis (RSI, MACD, etc.)
        - Volatility Assessment
        - Technical Signal Strength
        - Conclusion and Technical Outlook
        
        Please analyze the data objectively and highlight both bullish and bearish signals where present.
        """
        
        content = types.Content(
            role='user', 
            parts=[types.Part(text=analysis_request)]
        )

        try:
            events = self.runner.run_async(
                user_id=self.user_id, 
                session_id=session_id, 
                new_message=content
            )
            
            async for event in events:
                if hasattr(event, 'tool_call_started'):
                    print(f"🔧 Tool started: {event.tool_name}")
                elif hasattr(event, 'tool_call_completed'):
                    print(f"✅ Tool completed: {event.tool_name}")
                elif event.is_final_response():
                    return event.content.parts[0].text
            
            return "Analysis completed but no final response was received."
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return f"Analysis failed with error: {str(e)}"
    
    async def get_analysis_async(self, ticker_symbol, lookback_days=180):
        print(f"🚀 Starting technical analysis for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            analysis_result = await self.analyze_stock(ticker_symbol, lookback_days)
            return analysis_result
            
        except Exception as e:
            error_msg = f"❌ Analysis process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_technical_analysis(self, ticker_symbol, lookback_days=180):
        try:
            return asyncio.run(self.get_analysis_async(ticker_symbol, lookback_days))
        except Exception as e:
            return f"Analysis execution failed: {str(e)}"


if __name__ == "__main__":
    analyst = TechnicalAnalyst()
    
    ticker = "GOOG"
    print(f"🎯 Analyzing {ticker}")
    
    result = analyst.get_technical_analysis(ticker)
    
    print(f"\nTechnical Analysis result for {ticker}:\n")
    print(result)