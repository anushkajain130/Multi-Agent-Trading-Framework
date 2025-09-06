import asyncio
from Researcher.bullish_researcher import BullishResearcher
from Researcher.bearish_researcher import BearishResearcher
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

class DebateState:
    def __init__(self, ticker, technical, sentiment, news, fundamentals, history=None, turn="bull", round=0, max_rounds=4, judge_verdict=None, synthesis=None):
        self.ticker = ticker
        self.technical = technical
        self.sentiment = sentiment
        self.news = news
        self.fundamentals = fundamentals
        self.history = history if history is not None else []
        self.turn = turn
        self.round = round
        self.max_rounds = max_rounds
        self.judge_verdict = judge_verdict
        self.synthesis = synthesis

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_last_message(self):
        if self.history:
            return self.history[-1]["content"]
        return ""

    def as_dict(self):
        return {
            "ticker": self.ticker,
            "technical": self.technical,
            "sentiment": self.sentiment,
            "news": self.news,
            "fundamentals": self.fundamentals,
            "history": self.history,
            "turn": self.turn,
            "round": self.round,
            "max_rounds": self.max_rounds,
            "judge_verdict": self.judge_verdict,
            "synthesis": self.synthesis
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            ticker=d["ticker"],
            technical=d["technical"],
            sentiment=d["sentiment"],
            news=d["news"],
            fundamentals=d["fundamentals"],
            history=d.get("history", []),
            turn=d.get("turn", "bull"),
            round=d.get("round", 0),
            max_rounds=d.get("max_rounds", 4),
            judge_verdict=d.get("judge_verdict", None),
            synthesis=d.get("synthesis", None)
        )

class DebateJudge:
    def __init__(self):
        load_dotenv()
        self.app_name = "debate_judge"
        self.user_id = "judge_user"
        self.session_id_base = "debate_judge_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="debate_judge",
            model="gemini-2.0-flash",
            description="Objective debate judge for bull vs bear investment debates",
            instruction="""
You are a neutral and objective investment debate judge. Your task is to review the full transcript of a bull vs bear debate about a stock and decide which side presented a stronger, more evidence-based, and convincing argument. 
If neither side is convincing enough, respond with 'NO' as your verdict and request another round of debate.
Only respond with "bull", "bear", or "tie" to the debate arguments, don't provide any additional commentary or analysis. 
"""
        )

    async def judge_debate(self, ticker, debate_history):
        session_id = f"{self.session_id_base}_{ticker}"
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
        transcript = ""
        for i, msg in enumerate(debate_history, 1):
            transcript += f"Turn {i} [{msg['role'].upper()}]:\n{msg['content']}\n\n"
        prompt = f"""
You are the judge for the following investment debate about {ticker.upper()}.
Here is the full transcript:

{transcript}

Please provide:
1. Your verdict: (Bull, Bear, Tie, or NO if neither side is convincing)
2. A concise justification for your decision.
"""
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        events = self.runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=content
        )
        async for event in events:
            if event.is_final_response():
                return event.content.parts[0].text
        return "No verdict returned."

class DebateSynthesizer:
    def __init__(self):
        load_dotenv()
        self.app_name = "debate_synthesizer"
        self.user_id = "synth_user"
        self.session_id_base = "debate_synth_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="debate_synthesizer",
            model="gemini-2.0-flash",
            description="Synthesizer for summarizing and concluding the debate outcome",
            instruction="""
You are a financial debate synthesizer. Your job is to review the full transcript and the judge's verdict of a bull vs bear investment debate, and provide a final, concise, and actionable summary for an investor. 
Summarize the strongest points from both sides, clearly state the judge's verdict, and give a balanced, practical takeaway for decision-making.
"""
        )

    async def synthesize(self, ticker, debate_history, judge_verdict):
        session_id = f"{self.session_id_base}_{ticker}"
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
        transcript = ""
        for i, msg in enumerate(debate_history, 1):
            transcript += f"Turn {i} [{msg['role'].upper()}]:\n{msg['content']}\n\n"
        prompt = f"""
You are the synthesizer for the following investment debate about {ticker.upper()}.
Here is the full transcript:

{transcript}

Judge's Verdict: {judge_verdict}

Please provide:
1. A concise summary of the strongest bull and bear arguments.
2. The judge's verdict.
3. A balanced, actionable takeaway for an investor.
"""
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        events = self.runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=content
        )
        async for event in events:
            if event.is_final_response():
                return event.content.parts[0].text
        return "No synthesis returned."

def bull_node(state_dict):
    bull = BullishResearcher()
    state = DebateState.from_dict(state_dict)
    result = bull.get_bullish_research(
        state.ticker,
        state.technical,
        state.sentiment,
        state.news,
        state.fundamentals
    )
    state.add_message("bull", result)
    state.turn = "bear"
    return state.as_dict()

def bear_node(state_dict):
    bear = BearishResearcher()
    state = DebateState.from_dict(state_dict)
    result = bear.get_bearish_research(
        state.ticker,
        state.technical,
        state.sentiment,
        state.news,
        state.fundamentals
    )
    state.add_message("bear", result)
    state.turn = "judge"
    state.round += 1
    return state.as_dict()

async def judge_node(state_dict):
    judge = DebateJudge()
    state = DebateState.from_dict(state_dict)
    verdict = await judge.judge_debate(state.ticker, state.history)
    state.add_message("judge", verdict)
    state.judge_verdict = verdict
    return state.as_dict()

async def synth_node(state_dict):
    synthesizer = DebateSynthesizer()
    state = DebateState.from_dict(state_dict)
    synthesis = await synthesizer.synthesize(state.ticker, state.history, state.judge_verdict)
    state.synthesis = synthesis
    return state.as_dict()

def debate_end(state_dict):
    return state_dict

def debate_router(state_dict):
    state = DebateState.from_dict(state_dict)
    if state.judge_verdict is not None:
        verdict = state.judge_verdict.lower()
        if verdict.startswith("no"):
            state.turn = "bull"
            return "bull"
        if any(x in verdict for x in ["bull", "bear", "tie"]):
            return "synth"
    if state.round >= state.max_rounds:
        return "synth"
    return state.turn

async def run_debate_simulation_async(
    ticker,
    technical,
    sentiment,
    news,
    fundamentals,
    max_rounds=6
):
    state = DebateState(ticker, technical, sentiment, news, fundamentals)
    state.max_rounds = max_rounds

    graph = StateGraph(dict)
    graph.add_node("bull", bull_node)
    graph.add_node("bear", bear_node)
    graph.add_node("judge", judge_node)
    graph.add_node("synth", synth_node)
    graph.add_node("end", debate_end)
    graph.set_entry_point("bull")
    graph.add_conditional_edges(
        "bull", lambda s: "bear", {"bear": "bear"}
    )
    graph.add_conditional_edges(
        "bear", lambda s: "judge", {"judge": "judge"}
    )
    graph.add_conditional_edges(
        "judge", debate_router, {"bull": "bull", "synth": "synth"}
    )
    graph.add_conditional_edges(
        "synth", lambda s: "end", {"end": "end"}
    )
    graph.add_edge("end", END)
    debate_graph = graph.compile()
    final_state_dict = await debate_graph.ainvoke(state.as_dict())
    return DebateState.from_dict(final_state_dict)

def run_debate_simulation(
    ticker,
    technical,
    sentiment,
    news,
    fundamentals,
    max_rounds=6
):
    return asyncio.run(run_debate_simulation_async(
        ticker, technical, sentiment, news, fundamentals, max_rounds
    ))

def debate_results(final_state):
    print(f"Debate Simulation for {final_state.ticker}:\n")
    for i, msg in enumerate(final_state.history, 1):
        print(f"Turn {i} [{msg['role'].upper()}]:\n{msg['content']}\n{'-'*60}")
    print("\nSYNTHESIS:\n")
    return final_state.synthesis

if __name__ == "__main__":
    ticker = "NVDA"
    technical = "Recent technical indicators show a strong uptrend with RSI at 70 and MACD bullish crossover"
    sentiment = "Positive social media sentiment with 75% bullish posts"
    news = "Recent partnership announcement with major cloud providers"
    fundamentals = "P/E ratio of 25, growing at 30% annually"
    max_rounds = 4

    final_state = run_debate_simulation(
        ticker, technical, sentiment, news, fundamentals, max_rounds
    )
    results = debate_results(final_state)
    print(results)