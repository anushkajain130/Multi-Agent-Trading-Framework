---

# AlphaForge

### Modular Multi-Agent AI Research & Trading Workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hackathon](https://img.shields.io/badge/Google-ADK_Hackathon-red.svg)](https://devpost.com)

AlphaForge is a sophisticated multi-agent orchestration system designed for the Google Agent Development Kit Hackathon. It leverages the Agent Development Kit and Google Cloud to transform raw market data into explainable, automated financial decisions through a "Council of Agents" approach.

---

## Key Features

| Agent Type | Functionality |
| :--- | :--- |
| Parallel Analysts | Simultaneous Technical, Sentiment, News, and Fundamental data processing. |
| Debate Engine | LangGraph-powered simulation between Bullish and Bearish researchers. |
| Risk Synthesis | Tri-layer debate (Aggressive, Neutral, Safe) to determine optimal capital exposure. |
| The Trader | Final actionable decision-making (BUY/HOLD/SELL) based on synthesized logic. |
| Reflection Layer | Self-correcting feedback loop that audits decisions for hallucinations or errors. |

---

## Workflow Architecture

```mermaid
flowchart TD
    A[Start: Input Ticker] --> B{Parallel Analysts}
    
    subgraph Analysts
    B --> B1[Technical]
    B --> B2[Sentiment]
    B --> B3[News]
    B --> B4[Fundamental]
    end

    B1 & B2 & B3 & B4 --> C[Researcher Debate: Bull vs Bear]
    
    C --> D[Debate Judge]
    D --> E[Debate Synthesizer]
    E --> F[Risk Management Synthesizer]
    F --> G[Trader Agent]
    
    G --> H{Reflection Agent}
    H -- Hallucination Detected --> G
    H -- Verified --> I[Final Output: Actionable Plan]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#00ff00,stroke:#333,stroke-width:2px
    style H fill:#ff9900,stroke:#333,stroke-width:2px
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AlphaForge.git
cd AlphaForge
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a .env file in the root directory and add your credentials:
```env
GOOGLE_CLOUD_PROJECT="your-project-id"
ADK_API_KEY="your-api-key"
```

---

## Usage

### Execute Workflow via CLI
Run the full automated pipeline for a specific ticker:
```bash
python Agents/workflow.py
```

### Launch Dashboard (UI)
Interact with the agents through a clean Streamlit interface:
```bash
streamlit run app.py
```

---

## Project Structure

```text
AlphaForge/
├── Agents/
│   ├── Researcher/      # Bull/Bear debate logic
│   ├── Analyst/         # Market data scrapers & processors
│   ├── Risk_Management/ # Strategy & safety synthesizers
│   ├── Trader/          # Decision execution logic
│   ├── Reflection/      # Verification & hallucination checks
│   ├── custom/          # Custom ADK tools
│   └── workflow.py      # Main orchestration entry point
├── app.py               # Streamlit UI
├── requirements.txt     # Dependency list
└── README.md            # Documentation
```

---

## Customization
- New Analysts: Easily plug in new data sources in the Agents/ directory.
- Logic Tuning: Modify the Risk_Management debate prompts to fit your profile.
- Scaling: Designed for easy deployment to Google Cloud Run or App Engine.

---

## License
This project is licensed under the MIT License.

---

**Built for the Google Agent Development Kit Hackathon.**
