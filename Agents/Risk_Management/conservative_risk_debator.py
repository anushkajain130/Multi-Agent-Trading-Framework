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

class SafeRiskAnalyst:
    def __init__(self):
        load_dotenv()
        self.app_name = "safe_risk_analyst"
        self.user_id = "conservative_user"
        self.session_id_base = "safe_debate_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="safe_risk_analyst",
            model="gemini-2.0-flash",
            description="Agent advocating for low-risk investment strategies, defending asset preservation and long-term stability.",
            instruction="""
As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth.

You will receive full debate context during each session, including analyst arguments, market data, and sentiment indicators.

Respond precisely, grounded in facts, and advocate for the safest course of action.
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
                     current_risky_response="", current_neutral_response="", history="", count=0):
        session_key = hash((ticker_symbol)) % (10 ** 8)
        session_id = await self.setup_session(session_key)
        if not session_id:
            return "Debate could not be started due to session setup failure."

        # Compose the debate prompt
        debate_prompt = f"""
As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. 

Your task is to actively counter the arguments of the Risky and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} 
Here is the last response from the risky analyst: {current_risky_response} 
Here is the last response from the neutral analyst: {current_neutral_response} 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting.
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
                               current_risky_response="", current_neutral_response="", history="", count=0):
        print(f"🛡️ Starting conservative risk debate for {ticker_symbol}:\n")
        print("=" * 80)
        try:
            debate_result = await self.debate(
                ticker_symbol, sentiment_report, news_report, fundamentals_report,
                current_risky_response, current_neutral_response, history, count
            )
            return debate_result
        except Exception as e:
            error_msg = f"❌ Debate process failed: {str(e)}"
            print(error_msg)
            return error_msg

    def get_conservative_debate(self, ticker_symbol, sentiment_report, news_report, fundamentals_report,
                                 current_risky_response="", current_neutral_response="", history="", count=0):
        try:
            return asyncio.run(self.get_debate_async(
                ticker_symbol, sentiment_report, news_report, fundamentals_report,
                current_risky_response, current_neutral_response, history, count
            ))
        except Exception as e:
            return f"Debate execution failed: {str(e)}"

if __name__ == "__main__":
    safe_analyst = SafeRiskAnalyst()

    ticker_symbol = "AAPL"
    sentiment_report = "Investors are excited about AI launches, but worry about inflated valuation."
    news_report = "Geopolitical instability in Europe may affect global markets."
    fundamentals_report = "Cash flow is negative despite solid revenue; debt levels are slightly rising."
    current_risky_response = "The AI launch is a massive game-changer; we should double down ahead of earnings."
    current_neutral_response = "It's a promising move, but we need to watch volatility."
    history = "Debate so far: risky analyst went bold, neutral raised some caution."

    result = safe_analyst.get_conservative_debate(
        ticker_symbol,
        sentiment_report,
        news_report,
        fundamentals_report,
        current_risky_response,
        current_neutral_response,
        history
    )

    print("\nSafe/Conservative Debate Response:\n", result)
