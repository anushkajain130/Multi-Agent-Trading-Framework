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

from custom.fundamental_analysis_tool import analyze_fundamental_indicators
from custom.sec_filling_tool import comprehensive_sec_analysis_tool

class FundamentalAnalyst:
    def __init__(self):
        load_dotenv()
        
        self.app_name = "stock_fundamental_analyst"
        self.user_id = "analyst_user"
        self.session_id_base = "fundamental_analysis_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="stock_fundamental_analyst",
            model="gemini-2.0-flash",
            description="Professional fundamental analyst agent using integrated tools for comprehensive stock fundamental analysis",
            instruction="""
            You are a professional fundamental analyst specializing in stock market fundamental analysis.
            You have access to powerful integrated tools that calculate fundamental indicators and interpret financial metrics.

            **Your Fundamental Analysis Framework:**

            1. **Comprehensive Financial Analysis**:
               - Analyze profitability metrics and ratios
               - Evaluate growth metrics and trends
               - Assess liquidity and solvency ratios
               - Review efficiency metrics
               - Calculate valuation metrics
               - Monitor dividend metrics if applicable

            2. **Analysis Structure**:
               - **Executive Summary**: Key findings and overall financial health
               - **Profitability Analysis**: ROE, ROA, Margins
               - **Growth Analysis**: Revenue, Earnings, Market Position
               - **Financial Health**: Debt, Liquidity, Efficiency
               - **Valuation Assessment**: Fair value and comparative metrics
               - **Risk Analysis**: Key financial risks and concerns

            3. **Quality Standards**:
               - Provide detailed analysis based on financial metrics
               - Compare against industry benchmarks when relevant
               - Highlight both strengths and weaknesses
               - Maintain objectivity with fact-based assessment
               - Focus on fundamental factors
               - Consider both historical performance and future outlook

            **Important Guidelines**:
            - Use the provided fundamental indicators from the tool
            - Consider multiple metrics together for comprehensive analysis
            - Structure your response clearly with proper sections
            - Avoid giving specific investment recommendations
            - Clearly identify both positive and negative factors
            """,
            tools=[self.get_fundamental_indicators, self.comprehensive_sec_analysis]
        )
    
    def get_fundamental_indicators(self, ticker_symbol: str) -> str:
        try:
            print(f"Retrieving fundamental indicators for {ticker_symbol}...")
            indicators = analyze_fundamental_indicators(ticker_symbol)
            
            if indicators is None:
                return json.dumps({"error": f"Unable to retrieve data for {ticker_symbol}"})
            
            # Process any pandas/numpy values to make them JSON serializable
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
                "error": f"Fundamental analysis failed: {str(e)}",
                "ticker_symbol": ticker_symbol
            })
    
    def comprehensive_sec_analysis(self, ticker_symbol: str) -> str:
        try:
            print(f"Retrieving comprehensive SEC filings analysis for {ticker_symbol}...")
            analysis = comprehensive_sec_analysis_tool(ticker_symbol)
            
            if analysis is None:
                return json.dumps({"error": f"Unable to retrieve SEC filings for {ticker_symbol}"})
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"SEC filings analysis failed: {str(e)}",
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
    
    async def analyze_stock(self, ticker_symbol: str):
        session_id = await self.setup_session(ticker_symbol)
        if not session_id:
            return "Analysis could not be started due to session setup failure."
        
        analysis_request = f"""
        Please provide a comprehensive fundamental analysis of {ticker_symbol.upper()} using all available fundamental indicators, SEC fillings.
        
        **Analysis Requirements:**
        1. Use the GetFundamentalIndicators tool to retrieve fundamental data for {ticker_symbol.upper()}
        2. Analyze all available financial metrics and ratios
        3. Provide a structured, professional analysis with clear sections
        4. Focus on both historical performance and future outlook
        
        **Expected Analysis Structure:**
        - Executive Summary and Overall Financial Health
        - Profitability Analysis
        - Growth Analysis
        - Financial Health Assessment
        - Valuation Analysis
        - Risk Assessment
        
        Please analyze the data objectively and highlight both strengths and weaknesses.
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
    
    async def get_analysis_async(self, ticker_symbol):
        print(f"🚀 Starting fundamental analysis for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            analysis_result = await self.analyze_stock(ticker_symbol)
            return analysis_result
            
        except Exception as e:
            error_msg = f"❌ Analysis process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_fundamental_analysis(self, ticker_symbol):
        try:
            return asyncio.run(self.get_analysis_async(ticker_symbol))
        except Exception as e:
            return f"Analysis execution failed: {str(e)}"


if __name__ == "__main__":
    analyst = FundamentalAnalyst()
    
    ticker = "AAPL"
    print(f"🎯 Analyzing {ticker}")
    
    result = analyst.get_fundamental_analysis(ticker)
    
    print(f"\nFundamental Analysis result for {ticker}:\n")
    print(result)