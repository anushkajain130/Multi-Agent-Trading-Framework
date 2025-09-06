import os
import sys
import asyncio
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .aggresive_risk_debator import AggressiveRiskDebator
from .conservative_risk_debator import SafeRiskAnalyst
from .neutral_risk_debator import NeutralRiskAnalyst

class RiskSynthesizer:
    def __init__(self):
        load_dotenv()
        self.app_name = "risk_synthesizer"
        self.user_id = "synth_user"
        self.session_id_base = "risk_synthesis_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="risk_synthesizer",
            model="gemini-2.0-flash",
            description="Agent that synthesizes arguments from aggressive, neutral, and safe risk analysts to generate a cohesive risk summary.",
            instruction="""
You are the Risk Synthesizer. Your job is to read the arguments from three analysts:
- The Aggressive Analyst, who pushes for high-reward, high-risk strategies.
- The Neutral Analyst, who promotes a balanced and moderate investment plan.
- The Safe Analyst, who advocates for conservative and low-risk strategies.

Task:
1. Concisely summarize each analyst’s viewpoint.
2. Highlight key points of agreement and disagreement.
3. Identify blind spots or biases in any of the positions.
4. Provide a final strategic synthesis that integrates their inputs or offers a reasoned conclusion favoring one stance.

Avoid formatting, and write in a professional, human conversational tone.
"""
        )

    async def setup_session(self, session_key):
        session_id = f"{self.session_id_base}_{session_key}"
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

    async def synthesize(
        self,
        ticker_symbol,
        sentiment_report,
        news_report,
        fundamentals_report,
        history=""
    ):
        aggressive = AggressiveRiskDebator()
        safe = SafeRiskAnalyst()
        neutral = NeutralRiskAnalyst()

        aggressive_plan = aggressive.get_aggressive_debate(
            ticker_symbol, sentiment_report, news_report, fundamentals_report
        )
        neutral_plan = neutral.get_neutral_debate(
            ticker_symbol, sentiment_report, news_report, fundamentals_report, aggressive_plan
        )
        safe_plan = safe.get_conservative_debate(
            ticker_symbol, sentiment_report, news_report, fundamentals_report, aggressive_plan, neutral_plan
        )

        session_key = hash((ticker_symbol, aggressive_plan, neutral_plan, safe_plan)) % (10 ** 8)
        session_id = await self.setup_session(session_key)

        prompt = f"""
You are summarizing an internal debate between investment risk analysts about a trading plan for {ticker_symbol}.



Aggressive Analyst Plan:
{aggressive_plan}

Neutral Analyst Plan:
{neutral_plan}

Safe Analyst Plan:
{safe_plan}

Debate History:
{history}

Your job is to:
1. Briefly summarize each analyst's position.
2. Compare and contrast them.
3. Identify any flaws or oversights.
4. Suggest a synthesized or final actionable recommendation.
"""

        content = types.Content(
            role='user',
            parts=[types.Part(text=prompt)]
        )

        events = self.runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=content
        )
        async for event in events:
            if event.is_final_response():
                return event.content.parts[0].text
        return "Synthesis completed but no final response was received."

    def get_synthesis(
        self,
        ticker_symbol,
        sentiment_report,
        news_report,
        fundamentals_report,
        history=""
    ):
        return asyncio.run(self.synthesize(
            ticker_symbol,
            sentiment_report,
            news_report,
            fundamentals_report,
            history
        ))

if __name__ == "__main__":
    ticker = "AAPL"
    sentiment_report = "Investors excited about AI launch; slight concern over valuations."
    news_report = "U.S.-China tensions rise; global markets uncertain."
    fundamentals_report = "High revenue growth, negative free cash flow."

    synth = RiskSynthesizer()
    synthesis_result = synth.get_synthesis(
        ticker_symbol=ticker,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        history="Aggressive led, Neutral raised balance, Safe highlighted risks."
    )

    print("\n📌 Final Synthesized Recommendation:\n", synthesis_result)