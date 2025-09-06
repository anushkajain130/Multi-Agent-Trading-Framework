import asyncio
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

class ReflectionAgent:
    def __init__(self):
        load_dotenv()
        self.app_name = "reflection_agent"
        self.user_id = "reflection_user"
        self.session_id_base = "reflection_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="reflection_agent",
            model="gemini-2.0-flash",
            description="Agent that checks for hallucinations in trader recommendations",
            instruction="""
You are a reflection agent. Your job is to critically evaluate the trader's recommendation for hallucinations or unsupported claims. 
Given the trader's response and the provided research, market, sentiment, news, fundamentals, and risk reports, check if the trader's rationale is fully supported by the evidence. 
If you find any part of the trader's response that is not grounded in the provided reports, clearly state what is hallucinated or unsupported. 
If the response is fully supported, reply with 'NO HALLUCINATION'. Otherwise, reply with 'HALLUCINATION DETECTED' and explain briefly.
"""
        )

    async def setup_session(self, company_name):
        session_id = f"{self.session_id_base}_{company_name}"
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

    async def reflect_async(
        self,
        company_of_interest,
        trader_decision,
        research_report,
        market_report,
        sentiment_report,
        news_report,
        fundamentals_report,
        risk_report
    ):
        user_prompt = (
            f"Company: {company_of_interest}\n"
            f"Trader Decision: {trader_decision}\n"
            f"Research Report: {research_report}\n"
            f"Market Report: {market_report}\n"
            f"Sentiment Report: {sentiment_report}\n"
            f"News Report: {news_report}\n"
            f"Fundamentals Report: {fundamentals_report}\n"
            f"Risk Report: {risk_report}\n\n"
            "Does the trader's response contain any hallucinated or unsupported claims? If so, reply with 'HALLUCINATION DETECTED' and explain. If not, reply with 'NO HALLUCINATION'."
        )
        session_id = await self.setup_session(company_of_interest)
        content = types.Content(role='user', parts=[types.Part(text=user_prompt)])
        events = self.runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=content
        )
        async for event in events:
            if event.is_final_response():
                return {
                    "reflection": event.content.parts[0].text
                }
        return {
            "reflection": ""
        }

    def reflect(
        self,
        company_of_interest,
        trader_decision,
        research_report,
        market_report,
        sentiment_report,
        news_report,
        fundamentals_report,
        risk_report
    ):
        return asyncio.run(self.reflect_async(
            company_of_interest,
            trader_decision,
            research_report,
            market_report,
            sentiment_report,
            news_report,
            fundamentals_report,
            risk_report
        ))

if __name__ == "__main__":
    agent = ReflectionAgent()
    company = "NVDA"
    trader_decision = "Based on strong AI chip demand and positive market momentum, I recommend BUY. BUY"
    research_report = "NVIDIA is a leading AI chip manufacturer with strong market position."
    market_report = "Uptrend, strong volume, RSI at 68."
    sentiment_report = "Mostly positive, some caution on valuation."
    news_report = "Announced new AI chip partnership."
    fundamentals_report = "P/E 30, strong revenue growth, high margins."
    risk_report = "High competition in AI space, regulatory risks."

    result = agent.reflect(
        company,
        trader_decision,
        research_report,
        market_report,
        sentiment_report,
        news_report,
        fundamentals_report,
        risk_report
    )
    print(f"Reflection for {company}: {result['reflection']}")
