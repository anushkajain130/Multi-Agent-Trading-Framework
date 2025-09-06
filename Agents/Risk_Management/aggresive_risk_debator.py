import os
import sys
import asyncio
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import json

# Custom tool import
# from custom.risk_model import analyze_stock_risk
# custom_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom')
# sys.path.append(custom_path)
# print(custom_path)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from custom.risk_model import analyze_stock_risk  # Must accept ticker_symbol

class AggressiveRiskDebator:
    def __init__(self):
        load_dotenv()
        self.app_name = "aggressive_risk_debator"
        self.user_id = "debator_user"
        self.session_id_base = "risk_debate_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="aggressive_risk_debator",
            model="gemini-2.0-flash",
            description="Agent specializing in aggressively debating for high-risk, high-reward investment strategies using advanced risk modeling tools.",
            instruction="""
You are the Aggressive Risk Debator. Your mission is to champion bold, high-reward investment strategies, using data-driven arguments and advanced risk modeling.
- **Directly challenge** conservative and neutral viewpoints, refuting their points with persuasive logic and evidence.
- **Emphasize upside, innovation, and competitive advantages** even when risks are high.
- **Use the RiskModel tool** to quantify and justify high-risk opportunities, referencing the specific ticker symbol.
- **Never hallucinate arguments** if no opposing responses are present—focus on your own case.
- **Maintain a debate style**: address counterpoints, highlight missed opportunities, and assert why risk-taking is optimal for outperformance.
- **Output in a conversational, persuasive tone.**

**Debate Framework:**
1. Summarize the trader's plan for the given ticker and your bold thesis.
2. Directly rebut conservative and neutral arguments (if present).
3. Use market, sentiment, news, and fundamentals data to support your stance.
4. Run the RiskModel tool for quantitative backup, referencing the ticker.
5. End with a call-to-action or summary of why risk-taking is justified.
""",
            tools=[self.get_risk_model_opportunity]
        )

    def get_risk_model_opportunity(self, ticker_symbol: str) -> str:
        """
        Calls the risk modeling tool to evaluate the high-risk, high-reward opportunity for a specific ticker.
        """
        try:
            print(f"Running risk model tool for {ticker_symbol} aggressive opportunity evaluation...")
            result = analyze_stock_risk(ticker_symbol)
            
            if result is None:
                return json.dumps({"error": f"Unable to retrieve data for {ticker_symbol}"})
            
            processed_result = {}
            for key, value in result.items():
                if hasattr(value, 'iloc'):  # If it's a pandas Series
                    processed_result[key] = value.iloc[-1]
                elif hasattr(value, 'item'):  # For numpy values
                    try:
                        processed_result[key] = value.item()
                    except:
                        processed_result[key] = str(value)
                else:
                    processed_result[key] = value
                    
            return json.dumps(processed_result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Technical analysis failed: {str(e)}",
                "ticker_symbol": ticker_symbol
            })

    async def setup_session(self, session_key):
        session_id = f"{self.session_id_base}_{session_key}"
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

    async def debate(self, ticker_symbol, sentiment_report, news_report, fundamentals_report,
                    safe_response="", neutral_response="", history="", risky_history="", safe_history="", neutral_history="", count=0):
        session_key = hash((ticker_symbol)) % (10 ** 8)
        session_id = await self.setup_session(session_key)
        if not session_id:
            return "Debate could not be started due to session setup failure."

        # Compose the debate prompt
        debate_prompt = f"""
**Your role:** 
- Directly challenge the conservative and neutral analysts.
- Use the RiskModel tool for quantitative risk-reward analysis for {ticker_symbol}.
- Incorporate the following reports:
    - Social Media Sentiment: {sentiment_report}
    - World Affairs/News: {news_report}
    - Company Fundamentals: {fundamentals_report}
- Here is the current conversation history: {history}
- Conservative Analyst's last argument: {safe_response}
- Neutral Analyst's last argument: {neutral_response}

If there are no responses from the other analysts, do not hallucinate—focus on your own case.

**Debate structure:** 
1. Summarize your high-reward thesis for {ticker_symbol}.
2. Rebut opposing points (if any).
3. Use data and the RiskModel tool.
4. Finish with a persuasive conclusion.
"""

        # Prepare market data for the tool
        market_data = {
            "sentiment_report": sentiment_report,
            "news_report": news_report,
            "fundamentals_report": fundamentals_report
        }

        # Compose content for the agent
        content = types.Content(
            role='user',
            parts=[
                types.Part(text=debate_prompt),
            ]
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
            return "Debate completed but no final response was received."
        except Exception as e:
            print(f"❌ Debate error: {str(e)}")
            return f"Debate failed with error: {str(e)}"

    async def get_debate_async(self, ticker_symbol, sentiment_report, news_report, fundamentals_report,
                               safe_response="", neutral_response="", history="", risky_history="", safe_history="", neutral_history="", count=0):
        print(f"🚀 Starting aggressive risk debate for {ticker_symbol}:\n")
        print("=" * 80)
        try:
            debate_result = await self.debate(
                ticker_symbol, sentiment_report, news_report, fundamentals_report,
                safe_response, neutral_response, history, risky_history, safe_history, neutral_history, count
            )
            return debate_result
        except Exception as e:
            error_msg = f"❌ Debate process failed: {str(e)}"
            print(error_msg)
            return error_msg

    def get_aggressive_debate(self, ticker_symbol, sentiment_report, news_report, fundamentals_report,
                              safe_response="", neutral_response="", history="", risky_history="", safe_history="", neutral_history="", count=0):
        try:
            return asyncio.run(self.get_debate_async(
             ticker_symbol, sentiment_report, news_report, fundamentals_report,
                safe_response, neutral_response, history, risky_history, safe_history, neutral_history, count
            ))
        except Exception as e:
            return f"Debate execution failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    debator = AggressiveRiskDebator()

    trader_plan = "Buy 1000 shares ahead of earnings."
    ticker_symbol = "AAPL"
    sentiment_report = "Social media is bullish on AAPL; some warn of overvaluation."
    news_report = "AAPL Corp launching new AI product; global markets uncertain."
    fundamentals_report = "AAPL has high revenue growth, but negative cash flow."

    print(f"🎯 Debating plan for {ticker_symbol}: {trader_plan}")

    result = debator.get_aggressive_debate(
        ticker_symbol, sentiment_report, news_report, fundamentals_report
    )

    print(f"\nAggressive Risk Debate result:\n{result}")