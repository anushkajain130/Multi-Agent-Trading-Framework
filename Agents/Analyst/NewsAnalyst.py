import os
import asyncio
import requests
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
import json
import time

class NewsAnalyst:
    
    def __init__(self):
        load_dotenv()
        
        self.app_name = "company_news_analyst"
        self.user_id = "analyst_user"
        self.session_id_base = "news_analysis_session"
        
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()
    
    def _create_agent(self):
        return Agent(
            name="company_news_analyst",
            model="gemini-2.0-flash",
            description="Professional news analyst agent using long-running tools for comprehensive company analysis",
            instruction="""
            You are a professional financial news analyst specializing in comprehensive company analysis. 
            You have access to powerful long-running tools that can perform extensive research operations.

            **Your Enhanced Analysis Framework:**

            1. **Comprehensive Multi-Source Analysis**: Use your long-running tools strategically:
               - Use TavilyComprehensiveSearch for extensive news coverage across multiple financial sources
               - Allow sufficient time for long-running operations to complete

            2. **Analysis Structure**:
               - **Executive Summary**: Key findings and overall sentiment
               - **Recent News Analysis**: Comprehensive coverage from Tavily search results
               - **Market Context**: Financial metrics and market positioning
               - **Key Developments**: Major announcements, earnings, partnerships
               - **Conclusion**: Balanced assessment with key takeaways

            3. **Quality Standards**:
               - Provide detailed analysis based on comprehensive data gathering
               - Cross-reference information from multiple sources
               - Maintain objectivity and professional tone
               - Cite specific sources and dates when available
               - Highlight both positive and negative developments
               - Focus on factual reporting without investment advice

            **Important Guidelines**:
            - Use the comprehensive data from these tools to provide thorough analysis
            - Combine insights from all available tools for complete coverage
            - Structure your response clearly with proper sections and citations
            """,
            tools=[self.search_tavily_comprehensive]
        )
    
    def search_tavily_comprehensive(self, company_ticker: str) -> str:
        try:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                return json.dumps({"error": "Tavily API key not found"})
            
            base_url = "https://api.tavily.com/search"
            
            search_queries = [
                f"{company_ticker} earnings report recent",
                f"{company_ticker} financial news latest",
                f"{company_ticker} analyst coverage upgrade downgrade",
                f"{company_ticker} stock news market analysis",
                f"{company_ticker} company news developments"
            ]
            
            all_results = []
            
            for query in search_queries:
                time.sleep(1)  # Rate limiting
                
                payload = {
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 8,
                    "include_answer": True,
                    "include_raw_content": True,
                    "include_images": False,
                    "include_domains": [
                        "reuters.com", "bloomberg.com", "cnbc.com", 
                        "wsj.com", "ft.com", "marketwatch.com",
                        "seekingalpha.com", "yahoo.com"
                    ]
                }
                
                try:
                    response = requests.post(base_url, json=payload, timeout=30)
                    if response.status_code == 200:
                        search_data = response.json()
                        all_results.append({
                            "query": query,
                            "results": search_data.get("results", []),
                            "answer": search_data.get("answer", "")
                        })
                    else:
                        all_results.append({
                            "query": query,
                            "error": f"HTTP {response.status_code}: {response.text}"
                        })
                except requests.RequestException as e:
                    all_results.append({
                        "query": query,
                        "error": f"Request failed: {str(e)}"
                    })
            
            compiled_results = {
                "company_ticker": company_ticker.upper(),
                "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_queries": len(search_queries),
                "search_results": all_results,
                "summary": f"Comprehensive news search completed for {company_ticker.upper()} with {len(search_queries)} different query types"
            }
            
            return json.dumps(compiled_results, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Comprehensive search failed: {str(e)}",
                "company_ticker": company_ticker.upper()
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
    
    async def analyze_company(self, ticker_symbol: str):
        session_id = await self.setup_session(ticker_symbol)
        if not session_id:
            return "Analysis could not be started due to session setup failure."
        
        analysis_request = f"""
        Please provide a comprehensive analysis of {ticker_symbol.upper()} using all available tools including the long-running comprehensive search and financial analysis tools.
        
        **Analysis Requirements:**
        1. Use TavilyComprehensiveSearch to gather extensive news coverage from multiple financial sources
        3. Provide a structured, professional analysis with clear sections
        4. Focus on the last 30 days with emphasis on the most recent 7 days
        
        **Expected Analysis Structure:**
        - Executive Summary
        - Recent News Analysis (from comprehensive search)
        - Market Context and Financial Metrics
        - Key Developments and Announcements
        - Conclusion and Key Takeaways
        
        Please be patient as the long-running tools gather comprehensive data - this may take some time but will provide thorough analysis.
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
        print(f"🚀 Starting comprehensive analysis for: {ticker_symbol.upper()}")
        print("=" * 80)
        
        try:
            analysis_result = await self.analyze_company(ticker_symbol)
            return analysis_result
            
        except Exception as e:
            error_msg = f"❌ Analysis process failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_news_analysis(self, ticker_symbol):
        try:
            return asyncio.run(self.get_analysis_async(ticker_symbol))
        except Exception as e:
            return f"Analysis execution failed: {str(e)}"


if __name__ == "__main__":
    analyst = NewsAnalyst()
    
    ticker = "GOOG"
    print(f"🎯 Analyzing {ticker}")
    
    result = analyst.get_news_analysis(ticker)
    
    print(f"\nAnalysis result for {ticker}:\n")
    print(result)