import os
import sys
import asyncio
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import json

# Add custom path for risk_model tool
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from custom.risk_model import analyze_stock_risk  # Must accept ticker_symbol

class NeutralRiskAnalyst:
    def __init__(self):
        load_dotenv()
        self.app_name = "neutral_risk_analyst"
        self.user_id = "neutral_user"
        self.session_id_base = "neutral_debate_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="neutral_risk_analyst",
            model="gemini-2.0-flash",
            description="Agent focused on balanced investment perspectives, analyzing both risk and reward for a sustainable strategy.",
            instruction="""
As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan.

You will receive full debate context during each session, including analyst arguments, market data, and sentiment indicators.

Respond with fairness, question extreme views, and propose sustainable, diversified strategies that blend growth and safety.
""",
            tools=[self.get_risk_model_opportunity]
        )

    def get_risk_model_opportunity(self, ticker_symbol: str) -> str:
        try:
            print(f"Running risk model tool for {ticker_symbol} balanced opportunity evaluation...")
            result = analyze_stock_risk(ticker_symbol)

            if result is None:
                return json.dumps({"error": f"Unable to retrieve data for {ticker_symbol}"})

            processed_result = {}
            for key, value in result.items():
                if hasattr(value, 'iloc'):
                    processed_result[key] = value.iloc[-1]
                elif hasattr(value, 'item'):
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
                     current_risky_response="", current_safe_response="", history="", count=0):
        session_key = hash((ticker_symbol)) % (10 ** 8)
        session_id = await self.setup_session(session_key)
        if not session_id:
            return "Debate could not be started due to session setup failure."

        debate_prompt = f"""
As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.

Your task is to challenge both the Risky and Safe Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Here is the current conversation history: {history} 
Here is the last response from the risky analyst: {current_risky_response} 
Here is the last response from the safe analyst: {current_safe_response} 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage actively by analyzing both sides critically, addressing weaknesses in the risky and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Output conversationally as if you are speaking without any special formatting.
"""

        content = types.Content(
            role='user',
            parts=[types.Part(text=debate_prompt)]
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
                               current_risky_response="", current_safe_response="", history="", count=0):
        print(f"⚖️ Starting neutral risk debate for {ticker_symbol}:\n")
        print("=" * 80)
        try:
            debate_result = await self.debate(
                ticker_symbol, sentiment_report, news_report, fundamentals_report,
                current_risky_response, current_safe_response, history, count
            )
            return debate_result
        except Exception as e:
            error_msg = f"❌ Debate process failed: {str(e)}"
            print(error_msg)
            return error_msg

    def get_neutral_debate(self, ticker_symbol, sentiment_report, news_report, fundamentals_report,
                            current_risky_response="", current_safe_response="", history="", count=0):
        try:
            return asyncio.run(self.get_debate_async(
                ticker_symbol, sentiment_report, news_report, fundamentals_report,
                current_risky_response, current_safe_response, history, count
            ))
        except Exception as e:
            return f"Debate execution failed: {str(e)}"

if __name__ == "__main__":
    neutral_analyst = NeutralRiskAnalyst()

    ticker_symbol = "AAPL"
    sentiment_report = "Investors are excited about AI launches, but worry about inflated valuation."
    news_report = "Geopolitical instability in Europe may affect global markets."
    fundamentals_report = "Cash flow is negative despite solid revenue; debt levels are slightly rising."
    current_risky_response = "The AI launch is a massive game-changer; we should double down ahead of earnings."
    current_safe_response = "This AI hype may not translate into profits; too risky amid global instability."
    history = "Risky analyst went bold; safe analyst raised valuation and geopolitical concerns."

    result = neutral_analyst.get_neutral_debate(
        ticker_symbol,
        sentiment_report,
        news_report,
        fundamentals_report,
        current_risky_response,
        current_safe_response,
        history
    )

    print("\nNeutral Debate Response:\n", result)
