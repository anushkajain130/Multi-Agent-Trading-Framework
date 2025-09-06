import asyncio
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

class TraderAgent:
    def __init__(self):
        load_dotenv()
        self.app_name = "trader_agent"
        self.user_id = "trader_user"
        self.session_id_base = "trader_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="trader_agent",
            model="gemini-2.0-flash",
            description="Trading agent analyzing market data to make investment decisions",
            instruction="""
You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. Your response must include a clear rationale and end with only one of these: BUY, HOLD, or SELL.
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

    async def trade_decision_async(
        self,
        company_of_interest,
        research_report,
        market_report,
        sentiment_report,
        news_report,
        risk_report,
        fundamentals_report
    ):
        user_prompt = (
            f"Company: {company_of_interest}\n"
            f"Research Report: {research_report}\n"
            f"Market Report: {market_report}\n"
            f"Sentiment Report: {sentiment_report}\n"
            f"News Report: {news_report}\n"
            f"Fundamentals Report: {fundamentals_report}\n"
            f"Risk Report: {risk_report}\n\n"
            "Based on the above, provide a clear rationale and end your response with only one of these: BUY, HOLD, or SELL."
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
                    "decision": event.content.parts[0].text
                }
        return {
            "decision": ""
        }

    def trade_decision(
        self,
        company_of_interest,
        research_report,
        market_report,
        sentiment_report,
        news_report,
        fundamentals_report,
        risk_report
    ):
        return asyncio.run(self.trade_decision_async(
            company_of_interest,
            research_report,
            market_report,
            sentiment_report,
            news_report,
            fundamentals_report,
            risk_report
        ))

if __name__ == "__main__":
    agent = TraderAgent()
    company = "NVDA"
    research_report = "NVIDIA is a leading AI chip manufacturer with strong market position."
    market_report = "Uptrend, strong volume, RSI at 68."
    sentiment_report = "Mostly positive, some caution on valuation."
    news_report = "Announced new AI chip partnership."
    fundamentals_report = "P/E 30, strong revenue growth, high margins."
    risk_report = "High competition in AI space, regulatory risks."

    result = agent.trade_decision(
        company,
        research_report,
        market_report,
        sentiment_report,
        news_report,    
        fundamentals_report,
        risk_report
    )
    print(f"Decision for {company}: {result['decision']}")    
