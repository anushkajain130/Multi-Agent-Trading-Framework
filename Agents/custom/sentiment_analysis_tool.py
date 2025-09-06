import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

load_dotenv()

def is_rate_limited(response):
    return response.status_code == 429

@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    time.sleep(random.uniform(2, 6))
    response = requests.get(url, headers=headers)
    return response

def getNewsData(query, start_date, end_date):
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    while page < 2:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query}"
            f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_request(url, headers)
            soup = BeautifulSoup(response.content, "html.parser")
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                break

            for el in results_on_page:
                try:
                    link = el.find("a")["href"]
                    title = el.select_one("div.MBeuO").get_text()
                    snippet = el.select_one(".GI74Re").get_text()
                    date = el.select_one(".LfVVr").get_text()
                    source = el.select_one(".NUnG9d span").get_text()
                    news_results.append({
                        "link": link,
                        "title": title,
                        "snippet": snippet,
                        "date": date,
                        "source": source,
                    })
                except Exception as e:
                    continue

            page += 1

        except Exception as e:
            break

    return news_results

def get_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content[:500] if content else ""
    except:
        return ""

def analyze_stock_sentiment(symbol, lookback_days=10):
    news_api_key       = os.getenv("NEWS_API_KEY")
    alpha_vantage_key  = os.getenv("ALPHA_VANTAGE_KEY")
    finnhub_key        = os.getenv("FINHUB_API_KEY")
    tavily_api_key     = os.getenv("TAVILY_API_KEY")

    stock = yf.Ticker(symbol)
    hist  = stock.history(period=f"{lookback_days}d")
    hist['returns'] = hist['Close'].pct_change()
    tech = {
        "price_momentum": hist['returns'].mean(),
        "volatility":     hist['returns'].std(),
        "price_trend":    "Bullish" if hist['returns'].mean() > 0 else "Bearish"
    }

    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    to_date   = datetime.now().strftime('%Y-%m-%d')

    params_news = {
        "q": symbol,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": news_api_key,
        "pageSize": 5
    }
    articles_newsapi = requests.get("https://newsapi.org/v2/everything", params=params_news).json().get("articles", [])
    newsapi_scores = []
    newsapi_items  = []
    for a in articles_newsapi:
        text  = (a.get("title","") or "") + " " + (a.get("description","") or "")
        content = get_article_content(a.get("url", "")) if a.get("url") else ""
        full_text = text + " " + content
        if full_text.strip():
            score = TextBlob(full_text).sentiment.polarity
            newsapi_scores.append(score)
            newsapi_items.append({
                "title":       a.get("title"),
                "description": a.get("description"),
                "url":         a.get("url"),
                "published_at":a.get("publishedAt"),
                "content":     content,
                "sentiment":   score
            })
    newsapi = {
        "average_sentiment": np.mean(newsapi_scores) if newsapi_scores else None,
        "sentiment_std":     np.std(newsapi_scores) if newsapi_scores else None,
        "article_count":     len(newsapi_scores),
        "articles":          newsapi_items
    }

    av_response = requests.get("https://www.alphavantage.co/query", params={
        "function": "NEWS_SENTIMENT",
        "tickers":  symbol,
        "apikey":   alpha_vantage_key
    }).json().get("feed", [])
    av_scores = []
    av_items  = []
    for item in av_response[:5]:
        if isinstance(item, dict):
            for t in item.get("ticker_sentiment", []):
                if t.get("ticker") == symbol:
                    score = float(t.get("ticker_sentiment_score", 0))
                    content = get_article_content(item.get("url", "")) if item.get("url") else ""
                    av_scores.append(score)
                    av_items.append({
                        "title":     item.get("title"),
                        "url":       item.get("url"),
                        "time":      item.get("time_published"),
                        "content":   content,
                        "sentiment": score
                    })
    alpha_vantage = {
        "average_sentiment": np.mean(av_scores) if av_scores else None,
        "sentiment_std":     np.std(av_scores) if av_scores else None,
        "article_count":     len(av_scores),
        "articles":          av_items
    }

    fh_response = requests.get("https://finnhub.io/api/v1/company-news", params={
        "symbol": symbol,
        "from":   from_date,
        "to":     to_date,
        "token":  finnhub_key
    }).json()
    fh_scores = []
    fh_items  = []
    if isinstance(fh_response, list):
        for i in fh_response[:5]:
            if isinstance(i, dict):
                text  = (i.get("headline","") or "") + " " + (i.get("summary","") or "")
                content = get_article_content(i.get("url", "")) if i.get("url") else ""
                full_text = text + " " + content
                if full_text.strip():
                    score = TextBlob(full_text).sentiment.polarity
                    fh_scores.append(score)
                    fh_items.append({
                        "headline": i.get("headline"),
                        "summary":  i.get("summary"),
                        "url":      i.get("url"),
                        "datetime": i.get("datetime"),
                        "content":  content,
                        "sentiment":score
                    })
    finnhub = {
        "average_sentiment": np.mean(fh_scores) if fh_scores else None,
        "sentiment_std":     np.std(fh_scores) if fh_scores else None,
        "article_count":     len(fh_scores),
        "articles":          fh_items
    }

    try:
        google_news_data = getNewsData(f"{symbol} financial news", from_date, to_date)
        google_scores = []
        google_items = []
        for item in google_news_data[:5]:
            text = item.get("title", "") + " " + item.get("snippet", "")
            if text.strip():
                score = TextBlob(text).sentiment.polarity
                google_scores.append(score)
                google_items.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "url": item.get("link"),
                    "date": item.get("date"),
                    "source": item.get("source"),
                    "sentiment": score
                })
    except Exception as e:
        google_scores = []
        google_items = []

    google_news = {
        "average_sentiment": np.mean(google_scores) if google_scores else None,
        "sentiment_std":     np.std(google_scores) if google_scores else None,
        "article_count":     len(google_scores),
        "articles":          google_items
    }

    if tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=tavily_api_key)
            tavily_results = tavily_client.search(query=f"{symbol} financial news", topic="general", time_range="week", max_results=5)
            tv_scores = []
            tv_items  = []
            for r in tavily_results.get("results", []):
                title = r.get("title") or r.get("raw_content","")[:60]
                url   = r.get("url")
                pub   = r.get("published_at", None)
                text  = r.get("raw_content","") or ""
                if text.strip():
                    score = TextBlob(text).sentiment.polarity
                    tv_scores.append(score)
                    tv_items.append({
                        "title":       title,
                        "url":         url,
                        "published_at":pub,
                        "content":     text[:500],
                        "sentiment":   score
                    })
        except Exception as e:
            tv_scores = []
            tv_items = []
    else:
        tv_scores = []
        tv_items = []
    
    tavily = {
        "average_sentiment": np.mean(tv_scores) if tv_scores else None,
        "sentiment_std":     np.std(tv_scores) if tv_scores else None,
        "article_count":     len(tv_scores),
        "articles":          tv_items
    }

    all_scores = [
        tech["price_momentum"],
        newsapi["average_sentiment"],
        alpha_vantage["average_sentiment"],
        finnhub["average_sentiment"],
        tavily["average_sentiment"],
        google_news["average_sentiment"]
    ]
    valid_scores = [s for s in all_scores if s is not None]
    aggregate_score = np.mean(valid_scores) if valid_scores else None

    return {
        "technical":     tech,
        "newsapi":       newsapi,
        "alpha_vantage": alpha_vantage,
        "finnhub":       finnhub,
        "google_news":   google_news,
        "tavily":        tavily,
        "aggregate_score": aggregate_score
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analysis_tool.py <ticker_symbol> [lookback_days]")
        sys.exit(1)
    ticker_symbol = sys.argv[1]
    lookback_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    result = analyze_stock_sentiment(ticker_symbol, lookback_days)
    print(json.dumps(result, indent=2))