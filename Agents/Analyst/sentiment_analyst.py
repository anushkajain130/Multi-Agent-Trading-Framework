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

from custom.sentiment_analysis_tool import analyze_stock_sentiment

class SentimentAnalyst:
    def __init__(self):
        load_dotenv()
        
        self.app_name = "stock_sentiment_analyst"
        self.user_id = "analyst_user"
        self.session_id_base = "sentiment_analysis_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="stock_sentiment_analyst",
            model="gemini-2.0-flash",
            description="Professional sentiment analyst agent using integrated tools for comprehensive market sentiment analysis",
            instruction="""
            You are a professional sentiment analyst specializing in market sentiment analysis.
            You have access to powerful integrated tools that analyze sentiment indicators and interpret market mood.

            **Your Sentiment Analysis Framework:**

            1. **Comprehensive Sentiment Analysis**:
               - Analyze news sentiment and media coverage
               - Evaluate social media sentiment trends
               - Assess technical sentiment indicators
               - Monitor sentiment momentum and shifts
               - Calculate aggregate sentiment metrics

            2. **Analysis Structure**:
               - **Executive Summary**: Key findings and overall sentiment
               - **News Analysis**: Media coverage and news sentiment
               - **Social Media Analysis**: Social platform sentiment
               - **Technical Sentiment**: Price and volume-based sentiment
               - **Sentiment Trends**: Changes and momentum in sentiment
               - **Risk Assessment**: Sentiment-based market risks

            3. **Quality Standards**:
               - Provide detailed analysis based on sentiment metrics
               - Compare current sentiment against historical patterns
               - Highlight significant sentiment shifts
               - Maintain objectivity in assessment
               - Consider multiple data sources
               - Identify potential sentiment catalysts

            **Important Guidelines**:
            - Use the provided sentiment indicators from the tool
            - Consider multiple sentiment sources together
            - Structure your response clearly with proper sections
            - Avoid giving specific investment recommendations
            - Clearly identify sentiment shifts and trends
            
            Highly important : properly indicate the specific news and social media posts that are driving the sentiment indicators.
            """,
            tools=[self.get_sentiment_indicators]
        )
    
    def get_sentiment_indicators(self, ticker_symbol: str, lookback_days : int) -> str:
        try:
            print(f"Retrieving sentiment indicators for {ticker_symbol}...")
            indicators = analyze_stock_sentiment(ticker_symbol, lookback_days)
            
            if indicators is None:
                return json.dumps({"error": f"Unable to retrieve sentiment data for {ticker_symbol}"})
            
            return json.dumps(indicators, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Sentiment analysis failed: {str(e)}",
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
        Please provide a comprehensive sentiment analysis of {ticker_symbol.upper()} using all available sentiment indicators.
        
        **Analysis Requirements:**
        1. Use the GetSentimentIndicators tool to retrieve sentiment data for {ticker_symbol.upper()}
        2. Analyze all available sentiment metrics and trends
        3. Provide a structured, professional analysis with clear sections
        4. Focus on both current sentiment and recent shifts
        
        **Expected Analysis Structure:**
        - Executive Summary and Overall Sentiment
        - News Sentiment Analysis
        - Social Media Sentiment
        - Technical Sentiment Indicators
        - Sentiment Trends and Momentum
        - Risk Assessment
        
        Please analyze the data objectively and highlight both positive and negative sentiment factors.
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
        print(f"🚀 Starting sentiment analysis for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            analysis_result = await self.analyze_stock(ticker_symbol)
            return analysis_result
            
        except Exception as e:
            error_msg = f"❌ Analysis process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_sentiment_analysis(self, ticker_symbol):
        try:
            return asyncio.run(self.get_analysis_async(ticker_symbol))
        except Exception as e:
            return f"Analysis execution failed: {str(e)}"


if __name__ == "__main__":
    analyst = SentimentAnalyst()
    
    ticker = "GOOG"
    print(f"🎯 Analyzing {ticker}")
    
    result = analyst.get_sentiment_analysis(ticker)
    
    print(f"\nSentiment Analysis result for {ticker}:\n")
    print(result)