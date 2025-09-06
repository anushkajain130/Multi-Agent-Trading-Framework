import os
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()



def google_news_research_tool(query: str, days_back: int = 7, num_results: int = 10) -> str:
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    
    try:
        encoded_query = quote_plus(query)
        
        if days_back <= 1:
            time_filter = "qdr:d"
        elif days_back <= 7:
            time_filter = "qdr:w"
        elif days_back <= 30:
            time_filter = "qdr:m"
        else:
            time_filter = "qdr:y"
        
        url = f"https://www.google.com/search?q={encoded_query}&tbm=nws&tbs={time_filter}&num={num_results}"
        
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_results = []
        news_containers = soup.find_all('div', class_='SoaBEf')
        
        for container in news_containers[:num_results]:
            try:
                title_element = container.find('div', class_='MBeuO')
                title = title_element.get_text() if title_element else "No title"
                
                link_element = container.find('a')
                link = link_element.get('href') if link_element else "No link"
                
                snippet_element = container.find('div', class_='GI74Re')
                snippet = snippet_element.get_text() if snippet_element else "No summary available"
                
                source_element = container.find('span', class_='NUnG9d')
                source = source_element.get_text() if source_element else "Unknown source"
                
                date_element = container.find('span', class_='LfVVr')
                date = date_element.get_text() if date_element else "Unknown date"
                
                news_results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                    'source': source,
                    'date': date
                })
                
            except Exception as e:
                continue
        
        if not news_results:
            return f"No news results found for query: {query}"
        
        formatted_results = f"Google News Research Results for: '{query}' (Last {days_back} days)\n"
        formatted_results += "=" * 60 + "\n\n"
        
        for i, result in enumerate(news_results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   Source: {result['source']} | Date: {result['date']}\n"
            formatted_results += f"   URL: {result['link']}\n"
            formatted_results += f"   Summary: {result['snippet']}\n\n"
        
        return formatted_results
        
    except requests.RequestException as e:
        return f"Error performing Google News search: {str(e)}"
    except Exception as e:
        return f"Unexpected error during Google News research: {str(e)}"



if __name__ == "__main__":
    test_query = "NVDA stock analysis"
    
    print("\n\nTesting Google News Research Tool:")
    print("-" * 50)
    news_result = google_news_research_tool("NVDA earnings", days_back=30, num_results=5)
    print(news_result)
    
