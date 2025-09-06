import os
import sys
import asyncio
import requests
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from dotenv import load_dotenv

import json
import time

custom_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom')
sys.path.append(os.path.dirname(custom_path))

from custom.price_prediction import predict_stock_prices

class BearishResearcher:
    
    def __init__(self):
        load_dotenv()
        
        self.app_name = "bearish_researcher"
        self.user_id = "bear_user"
        self.session_id_base = "bear_research_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="bearish_researcher",
            model="gemini-2.0-flash",
            description="Professional bearish equity researcher focused on identifying risks, challenges, and negative investment catalysts",
            instruction="""
            You are a highly experienced Bearish Equity Researcher with a proven track record of identifying overvalued securities and investment risks. Your expertise lies in building compelling cases that emphasize potential downsides, market vulnerabilities, and negative catalysts that could impact stock performance.

            **Your Research Philosophy:**
            You approach every analysis with a critical and skeptical outlook, seeking evidence of:
            - Structural weaknesses and competitive disadvantages
            - Overvaluation and unsustainable business models
            - Regulatory risks and market headwinds
            - Financial instability and deteriorating fundamentals
            - Management issues and strategic missteps
            - Macroeconomic threats and industry disruption

            **Core Research Framework:**

            1. **Risk Assessment & Vulnerabilities**:
               - Identify structural weaknesses in business model
               - Analyze market saturation and competitive threats
               - Evaluate regulatory and legal risks
               - Assess technological disruption potential
               - Examine management execution failures

            2. **Valuation Concerns & Market Inefficiencies**:
               - Highlight overvaluation relative to fundamentals
               - Analyze unsustainable growth assumptions
               - Review market bubble characteristics
               - Assess unrealistic investor expectations
               - Examine historical valuation reversions

            3. **Financial Deterioration Analysis**:
               - Identify declining margins and profitability
               - Analyze increasing debt burdens and leverage
               - Review cash flow deterioration patterns
               - Assess capital allocation inefficiencies
               - Examine working capital stress indicators

            4. **Competitive Disadvantages**:
               - Evaluate eroding market share and positioning
               - Analyze weakening pricing power
               - Review technological obsolescence risks
               - Assess brand deterioration and customer defection
               - Examine supply chain vulnerabilities

            5. **Negative Catalyst Identification**:
               - Regulatory crackdowns and policy changes
               - Product failures and safety issues
               - Management scandals and governance failures
               - Economic downturns and recession impacts
               - Industry disruption and market shifts

            6. **Contrarian Analysis Framework**:
               - Challenge consensus bullish narratives
               - Expose hidden risks and undisclosed liabilities
               - Identify accounting irregularities and red flags
               - Demonstrate unsustainable competitive advantages
               - Reveal management guidance unreliability

            **Analysis Delivery Standards**:
            - Present compelling risk narratives backed by quantitative evidence
            - Use historical precedents and case studies
            - Emphasize downside protection and risk-adjusted returns
            - Provide clear bear thesis with specific risk catalysts
            - Address bullish counterarguments with data-driven rebuttals
            - Maintain professional skepticism while avoiding unfounded pessimism

            **Communication Style**:
            - Analytical and evidence-based tone with critical perspective
            - Use specific examples of similar company failures
             - Present data highlighting deteriorating trends
            - Engage in constructive debate with counterarguments
            - Focus on asymmetric risk and potential for permanent capital loss

            Always remember: Your role is to identify and articulate legitimate investment risks while maintaining intellectual honesty and professional integrity. You seek to uncover hidden dangers and overvaluation that others might overlook.
            """,
            tools=[
                self.get_price_prediction
            ]
        )
        
    def get_price_prediction(self, ticker_symbol: str, days_ahead: int = 30) -> str:
        return predict_stock_prices(symbol=ticker_symbol, days_to_predict=days_ahead, training_years=5, seq_length=30)

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
            print(f"Session setup failed: {str(e)}")
            return None
    
    async def research_company(self, ticker_symbol: str, technical_analysis_data: str = "", sentiment_data: str = "", news_data: str = "", fundamentals_data: str = ""):
        session_id = await self.setup_session(ticker_symbol)
        if not session_id:
            return "Research could not be started due to session setup failure."
        
        research_request = f"""
        As a Bearish Equity Researcher, provide a comprehensive bearish investment analysis for {ticker_symbol.upper()}.

        **Available Research Data:**
        Sentiment Analysis: {sentiment_data}  
        News Coverage: {news_data}
        Fundamentals: {fundamentals_data}
        Technical indicators: {technical_analysis_data}

        **Required Analysis Framework:**

        1. **Executive Summary & Bear Thesis**
           - Clear bearish investment case in 2-3 sentences
           - Key risks and negative catalysts
           - Downside price targets and timeline

        2. **Risk Assessment & Vulnerabilities**
           - Business model weaknesses and structural risks
           - Market saturation and competitive pressures
           - Regulatory and legal risk exposures
           - Technological disruption threats
           - Management execution failures

        3. **Valuation Concerns & Overvaluation Evidence**
           - Current valuation vs historical norms
           - Unsustainable growth assumptions
           - Market bubble characteristics
           - Peer comparison disadvantages
           - Mean reversion probability

        4. **Financial Deterioration Analysis**
           - Declining profitability and margin compression
           - Increasing debt levels and leverage concerns
           - Cash flow deterioration patterns
           - Working capital stress indicators
           - Capital allocation inefficiencies

        5. **Competitive Disadvantages & Market Pressures**
           - Eroding market share and positioning
           - Weakening pricing power
           - Brand deterioration and customer defection
           - Supply chain vulnerabilities
           - Innovation lag and technological obsolescence

        6. **Negative Investment Catalysts**
           - Near-term risk events (next 6-12 months)
           - Medium-term structural challenges (1-3 years)
           - Long-term industry disruption themes
           - Specific regulatory, legal, or operational risks

        7. **Bull Case Refutation**
           - Address and counter popular bullish arguments
           - Expose weaknesses in growth projections
           - Challenge management guidance reliability
           - Demonstrate unsustainable competitive advantages
           - Historical precedents of similar failures

        **Tone & Style:**
        - Critical and analytical with evidence-based skepticism
        - Risk-focused with downside scenario modeling
        - Contrarian perspective challenging consensus views
        - Professional pessimism balanced with factual analysis

        Provide a compelling bearish case that would alert institutional investors to the significant risks and potential for capital loss.
        """
        
        content = types.Content(
            role='user', 
            parts=[types.Part(text=research_request)]
        )

        try:
            events = self.runner.run_async(
                user_id=self.user_id, 
                session_id=session_id, 
                new_message=content
            )
            
            async for event in events:
                if event.is_final_response():
                    return event.content.parts[0].text
            
            return "Research completed but no final response was received."
            
        except Exception as e:
            print(f"Research error: {str(e)}")
            return f"Research failed with error: {str(e)}"
    
    async def get_research_async(self, ticker_symbol, technical_analysis_data="", sentiment_data="", news_data="", fundamentals_data=""):
        print(f"Starting bearish research for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            research_result = await self.research_company(ticker_symbol, technical_analysis_data, sentiment_data, news_data, fundamentals_data)
            return research_result
            
        except Exception as e:
            error_msg = f"Research process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_bearish_research(self, ticker_symbol, technical_analysis_data="", sentiment_data="", news_data="", fundamentals_data=""):
        try:
            return asyncio.run(self.get_research_async(ticker_symbol, technical_analysis_data, sentiment_data, news_data, fundamentals_data))
        except Exception as e:
            return f"Research execution failed: {str(e)}"


if __name__ == "__main__":
    researcher = BearishResearcher()
    
    ticker = "NVDA"
    
    sample_technical_analysis_data = "Technical indicators show potential overbought conditions with RSI at 85 and negative divergence"
    sample_sentiment = "Mixed sentiment with growing concerns about valuation and competition"
    sample_news = "Regulatory scrutiny increasing on AI chips and potential export restrictions"
    sample_fundamentals = "Trading at 50x forward P/E with slowing revenue growth"
    
    print(f"Analyzing {ticker}")
    
    result = researcher.get_bearish_research(
        ticker, 
        sample_technical_analysis_data, 
        sample_sentiment, 
        sample_news, 
        sample_fundamentals
    )
    
    print(f"\nBearish Research Result for {ticker}:\n")
    print(result)