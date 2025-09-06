import asyncio
from concurrent.futures import ThreadPoolExecutor
from Researcher.bullish_researcher import BullishResearcher
from Researcher.bearish_researcher import BearishResearcher
from Analyst.TechnicalAnalyst import TechnicalAnalyst
from Analyst.sentiment_analyst import SentimentAnalyst
from Analyst.NewsAnalyst import NewsAnalyst
from Analyst.fundamental_analyst import FundamentalAnalyst
from Risk_Management.synthesizer import RiskSynthesizer
from Trader.agent import TraderAgent
from Reflection.agent import ReflectionAgent
from researcher_debate import run_debate_simulation, DebateState

def run_parallel_analysts(ticker):
    with ThreadPoolExecutor() as executor:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bullish_future = loop.run_in_executor(executor, BullishResearcher().get_bullish_research, ticker)
        bearish_future = loop.run_in_executor(executor, BearishResearcher().get_bearish_research, ticker)
        technical_future = loop.run_in_executor(executor, TechnicalAnalyst().get_technical_analysis, ticker)
        sentiment_future = loop.run_in_executor(executor, SentimentAnalyst().get_sentiment_analysis, ticker)
        news_future = loop.run_in_executor(executor, NewsAnalyst().get_news_analysis, ticker)
        fundamentals_future = loop.run_in_executor(executor, FundamentalAnalyst().get_fundamental_analysis, ticker)
        results = loop.run_until_complete(asyncio.gather(
            bullish_future, bearish_future, technical_future, sentiment_future, news_future, fundamentals_future
        ))
        loop.close()
    return {
        "bullish_report": results[0],
        "bearish_report": results[1],
        "technical_report": results[2],
        "sentiment_report": results[3],
        "news_report": results[4],
        "fundamentals_report": results[5]
    }

def workflow(ticker):
    analyst_reports = run_parallel_analysts(ticker)
    debate_state = run_debate_simulation(
        ticker,
        analyst_reports["technical_report"],
        analyst_reports["sentiment_report"],
        analyst_reports["news_report"],
        analyst_reports["fundamentals_report"],
        max_rounds=4
    )
    debate_synthesis = debate_state.synthesis

    risk_synth = RiskSynthesizer()
    risk_report = risk_synth.synthesize(
        ticker,
        analyst_reports["sentiment_report"],
        analyst_reports["news_report"],
        analyst_reports["fundamentals_report"],
        history=None
    )

    trader = TraderAgent()
    reflection = ReflectionAgent()
    risk_report = None 
    trader_decision = trader.trade_decision(
        ticker,
        debate_synthesis,
        analyst_reports["market_report"] if "market_report" in analyst_reports else analyst_reports["technical_report"],
        analyst_reports["sentiment_report"],
        analyst_reports["news_report"],
        analyst_reports["fundamentals_report"],
        risk_report
    )

    reflection_result = reflection.reflect(
        ticker,
        trader_decision["decision"],
        debate_synthesis,
        analyst_reports["market_report"] if "market_report" in analyst_reports else analyst_reports["technical_report"],
        analyst_reports["sentiment_report"],
        analyst_reports["news_report"],
        analyst_reports["fundamentals_report"],
        risk_report
    )

    retry_count = 0
    while "HALLUCINATION" in reflection_result["reflection"].upper() and retry_count < 2:
        risk_report = None  # Placeholder for risk report if needed
        trader_decision = trader.trade_decision(
            ticker,
            debate_synthesis,
            analyst_reports["market_report"] if "market_report" in analyst_reports else analyst_reports["technical_report"],
            analyst_reports["sentiment_report"],
            analyst_reports["news_report"],
            analyst_reports["fundamentals_report"],
            risk_report
        )
        reflection_result = reflection.reflect(
            ticker,
            trader_decision["decision"],
            debate_synthesis,
            analyst_reports["market_report"] if "market_report" in analyst_reports else analyst_reports["technical_report"],
            analyst_reports["sentiment_report"],
            analyst_reports["news_report"],
            analyst_reports["fundamentals_report"],
            risk_report
        )
        retry_count += 1

    return {
        "analyst_reports": analyst_reports,
        "debate_synthesis": debate_synthesis,
        "risk_report": risk_report,
        "trader_decision": trader_decision["decision"],
        "reflection": reflection_result["reflection"]
    }

if __name__ == "__main__":
    ticker = "NVDA"
    result = workflow(ticker)
    print("Analyst Reports:")
    for k, v in result["analyst_reports"].items():
        print(f"{k}: {v}\n")
    print("Debate Synthesis:\n", result["debate_synthesis"], "\n")
    print("Risk Report:\n", result["risk_report"], "\n")
    print("Trader Decision:\n", result["trader_decision"], "\n")
    print("Reflection:\n", result["reflection"])