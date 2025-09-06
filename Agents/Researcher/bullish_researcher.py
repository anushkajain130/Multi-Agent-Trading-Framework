import os
import asyncio
import requests
import sys
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

class BullishResearcher:
    
    def __init__(self):
        load_dotenv()
        
        self.app_name = "bullish_researcher"
        self.user_id = "bull_user"
        self.session_id_base = "bull_research_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="bullish_researcher",
            model="gemini-2.0-flash",
            description="Professional bullish equity researcher focused on identifying growth opportunities and positive investment catalysts",
            instruction="""
            You are a highly experienced Bullish Equity Researcher with a proven track record of identifying undervalued growth opportunities and investment catalysts. Your expertise lies in building compelling investment cases that emphasize growth potential, competitive advantages, and positive market dynamics.

            **Your Research Philosophy:**
            You approach every analysis with a constructive outlook, seeking evidence of:
            - Sustainable competitive advantages and market positioning
            - Revenue growth drivers and scalability potential
            - Innovation capabilities and market disruption opportunities
            - Strong financial fundamentals and efficient capital allocation
            - Positive industry trends and tailwinds
            - Management excellence and strategic vision
            - Use google search tool to gather comprehensive market data and news coverage
            You aim to present a balanced view that highlights both the opportunities and the risks, but always with a focus on the positive aspects that could drive long-term value creation.

            **Core Research Framework:**

            1. **Growth Potential Analysis**:
               - Identify multiple revenue streams and expansion opportunities
               - Analyze market size, penetration rates, and growth trajectories
               - Evaluate scalability of business model and operational leverage
               - Assess innovation pipeline and future product launches
               - Examine geographic expansion and market share gains

            2. **Competitive Advantage Assessment**:
               - Evaluate moats: network effects, switching costs, brand strength
               - Analyze pricing power and margin sustainability
               - Review intellectual property and technological advantages
               - Assess regulatory barriers and market positioning
               - Examine customer loyalty and retention metrics

            3. **Financial Strength Evaluation**:
               - Highlight improving margins and operating efficiency
               - Identify strong cash generation and balance sheet health
               - Analyze return on invested capital and asset utilization
               - Review debt management and capital structure optimization
               - Assess dividend growth potential and shareholder returns

            4. **Catalyst Identification**:
               - Product launches and market penetrations
               - Strategic partnerships and acquisitions
               - Regulatory approvals and market openings
               - Management changes and operational improvements
               - Industry consolidation opportunities

            5. **Risk Mitigation Framework**:
               - Address potential concerns with data-driven counterarguments
               - Highlight management's risk mitigation strategies
               - Demonstrate diversification and resilience factors
               - Show historical performance during challenging periods

            **Analysis Delivery Standards**:
            - Present compelling narratives backed by quantitative evidence
            - Use forward-looking metrics and growth projections
            - Emphasize long-term value creation opportunities
            - Provide clear investment thesis with specific catalysts
            - Address potential concerns proactively with solutions
            - Maintain professional optimism while acknowledging realistic challenges

            **Communication Style**:
            - Confident and persuasive tone backed by solid research
            - Use specific examples and case studies
            - Present data in compelling formats with clear conclusions
            - Engage in constructive debate with counterarguments
            - Focus on opportunity cost and relative value propositions

            Always remember: Your role is to find and articulate the best possible investment case while maintaining intellectual honesty and professional integrity. You seek to uncover hidden value and growth potential that others might overlook.
            """,
            tools=[self.get_stock_price_prediction]
        )
    
    def get_stock_price_prediction(self, ticker_symbol: str, days_to_predict: int = 30, training_years: int = 5, seq_length: int = 30) -> str:
        try:
            print(f"🔍 Predicting stock prices for {ticker_symbol}...")
            prediction = predict_stock_prices(ticker_symbol, days_to_predict, training_years, seq_length)
            
            if prediction is None:
                return f"❌ Unable to predict stock prices for {ticker_symbol}. Not enough historical data."
            
            return prediction
        
        except Exception as e:
            return f"❌ Error predicting stock prices: {str(e)}"
    
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
        As a Bullish Equity Researcher, provide a comprehensive bullish investment analysis for {ticker_symbol.upper()}.

        **Available Research Data:**
        Sentiment Analysis: {sentiment_data}  
        News Coverage: {news_data}
        Fundamentals: {fundamentals_data}
        Technical indicators: {technical_analysis_data}

        **Required Analysis Framework:**

        1. **Executive Summary & Investment Thesis**
           - Clear bullish investment case in 2-3 sentences
           - Key catalysts and growth drivers
           - Target price rationale and timeline

        2. **Growth Potential Analysis**
           - Revenue growth opportunities and market expansion
           - Scalability factors and operational leverage
           - Innovation pipeline and competitive positioning
           - Market share gains and penetration strategies

        3. **Competitive Advantages & Moats**
           - Sustainable competitive advantages
           - Pricing power and margin expansion potential
           - Technology, brand, or network effect advantages
           - Barriers to entry and defensive characteristics

        4. **Financial Strength & Capital Efficiency**
           - Balance sheet quality and cash generation
           - Return on capital trends and efficiency metrics
           - Capital allocation strategy and shareholder returns
           - Margin improvement opportunities

        5. **Key Investment Catalysts**
           - Near-term catalysts (next 6-12 months)
           - Medium-term growth drivers (1-3 years)
           - Long-term value creation themes
           - Specific events, launches, or milestones

        6. **Risk Mitigation & Bear Case Refutation**
           - Address potential concerns with data-driven counterarguments
           - Demonstrate resilience factors and risk management
           - Show why current valuation presents attractive opportunity
           - Historical performance during challenging periods

        **Tone & Style:**
        - Confident and persuasive with quantitative backing
        - Forward-looking with specific growth projections
        - Constructive and solution-oriented approach
        - Professional optimism balanced with realistic assessment

        Provide a compelling bullish case that would convince institutional investors of the growth opportunity and value proposition.
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
    
    async def get_research_async(self, ticker_symbol, market_data="", sentiment_data="", news_data="", fundamentals_data=""):
        print(f"Starting bullish research for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            research_result = await self.research_company(ticker_symbol, market_data, sentiment_data, news_data, fundamentals_data)
            return research_result
            
        except Exception as e:
            error_msg = f"Research process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_bullish_research(self, ticker_symbol, market_data="", sentiment_data="", news_data="", fundamentals_data=""):
        try:
            return asyncio.run(self.get_research_async(ticker_symbol, market_data, sentiment_data, news_data, fundamentals_data))
        except Exception as e:
            return f"Research execution failed: {str(e)}"


if __name__ == "__main__":
    researcher = BullishResearcher()
    
    ticker = "NVDA"
    
    sample_technical_analysis_data = "Recent technical indicators show a strong uptrend with RSI at 70 and MACD bullish crossover"
    sample_sentiment = "Positive social media sentiment with 75% bullish posts"
    sample_news = "Recent partnership announcement with major cloud providers"
    sample_fundamentals = "P/E ratio of 25, growing at 30% annually"
    
    print(f"Analyzing {ticker}")
    
    result = researcher.get_bullish_research(
        ticker, 
        sample_technical_analysis_data, 
        sample_sentiment, 
        sample_news, 
        sample_fundamentals
    )
    
    print(f"\nBullish Research Result for {ticker}:\n")
    print(result)