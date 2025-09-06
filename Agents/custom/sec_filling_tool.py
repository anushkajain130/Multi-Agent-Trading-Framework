import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin

def comprehensive_sec_analysis_tool(ticker: str, filing_type: str = "10-K", include_content: bool = True) -> str:
    """
    Comprehensive SEC filing analysis tool that retrieves filing information and content
    
    Args:
        ticker (str): Stock ticker symbol
        filing_type (str): Type of SEC filing (10-K, 10-Q, 8-K, etc.)
        include_content (bool): Whether to extract actual filing content
    
    Returns:
        str: Complete SEC filing analysis including metadata and content
    """
    try:
        headers = {'User-Agent': 'Company Research Tool (contact@example.com)'}
        
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': filing_type,
            'dateb': '',
            'count': '3'
        }
        
        response = requests.get(url, params=params, headers=headers)
        time.sleep(0.1)
        
        if response.status_code != 200:
            return f"Could not retrieve SEC filings for {ticker}. Status: {response.status_code}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        company_name_elem = soup.find('span', class_='companyName')
        company_name = company_name_elem.text.strip() if company_name_elem else "Unknown Company"
        
        cik_match = re.search(r'CIK#:\s*(\d+)', company_name)
        cik = cik_match.group(1) if cik_match else "Unknown"
        company_clean = company_name.split(' CIK#:')[0]
        
        filings_table = soup.find('table', class_='tableFile2')
        if not filings_table:
            return f"No {filing_type} filings found for {ticker.upper()}"
        
        filings = filings_table.find_all('tr')[1:]
        if not filings:
            return f"No filing data available for {ticker.upper()}"
        
        results = f"COMPREHENSIVE SEC FILING ANALYSIS\n"
        results += f"Ticker: {ticker.upper()}\n"
        results += f"Company: {company_clean}\n"
        results += f"CIK: {cik}\n"
        results += f"Filing Type: {filing_type}\n"
        results += "=" * 80 + "\n\n"
        
        for i, filing in enumerate(filings[:3], 1):
            cells = filing.find_all('td')
            if len(cells) >= 5:
                filing_desc = cells[0].text.strip()
                doc_link_elem = cells[1].find('a', id='documentsbutton')
                filing_date = cells[3].text.strip()
                file_number = cells[4].text.strip()
                
                results += f"FILING #{i}:\n"
                results += f"Description: {filing_desc}\n"
                results += f"Filing Date: {filing_date}\n"
                results += f"File Number: {file_number}\n"
                
                if doc_link_elem and doc_link_elem.get('href') and include_content and i == 1:
                    doc_link = urljoin("https://www.sec.gov", doc_link_elem['href'])
                    results += f"Documents URL: {doc_link}\n\n"
                    
                    content = extract_filing_content(doc_link, filing_type, headers)
                    results += content + "\n"
                else:
                    results += "\n"
                
                results += "-" * 60 + "\n\n"
        
        return results
        
    except Exception as e:
        return f"Error in comprehensive SEC analysis: {str(e)}"

def extract_filing_content(doc_page_url: str, filing_type: str, headers: dict) -> str:
    """Extract and analyze content from SEC filing"""
    try:
        time.sleep(0.1)
        doc_page_response = requests.get(doc_page_url, headers=headers)
        
        if doc_page_response.status_code != 200:
            return "CONTENT EXTRACTION: Failed to access documents page"
        
        doc_soup = BeautifulSoup(doc_page_response.content, 'html.parser')
        filing_links = doc_soup.find_all('a', href=True)
        
        main_filing_url = None
        for link in filing_links:
            href = link['href']
            if href.endswith(('.htm', '.html')):
                if any(keyword in href.lower() for keyword in [filing_type.lower(), 'filing', 'document']):
                    main_filing_url = urljoin("https://www.sec.gov", href)
                    break
        
        if not main_filing_url and filing_links:
            for link in filing_links:
                href = link['href']
                if href.endswith(('.htm', '.html')):
                    main_filing_url = urljoin("https://www.sec.gov", href)
                    break
        
        if not main_filing_url:
            return "CONTENT EXTRACTION: Could not locate main filing document"
        
        time.sleep(0.1)
        filing_response = requests.get(main_filing_url, headers=headers)
        
        if filing_response.status_code != 200:
            return "CONTENT EXTRACTION: Failed to access filing content"
        
        filing_soup = BeautifulSoup(filing_response.content, 'html.parser')
        content_result = "FILING CONTENT ANALYSIS:\n"
        content_result += f"Document URL: {main_filing_url}\n\n"
        
        if filing_type.upper() == "10-K":
            sections = extract_10k_key_sections(filing_soup)
            if sections:
                for section_name, section_content in sections.items():
                    content_result += f"{section_name.upper()}:\n"
                    content_result += section_content[:800] + "...\n\n"
            else:
                content_result += "GENERAL CONTENT:\n"
                text_content = filing_soup.get_text()
                clean_content = ' '.join(text_content.split())
                content_result += clean_content[:1200] + "...\n"
        
        elif filing_type.upper() == "8-K":
            content_result += "CURRENT REPORT HIGHLIGHTS:\n"
            text_content = filing_soup.get_text()
            
            item_patterns = [
                r"item\s*\d+\.\d+.*?(?=item\s*\d+\.\d+|$)",
                r"item\s*\d+.*?(?=item\s*\d+|$)"
            ]
            
            items_found = []
            for pattern in item_patterns:
                matches = re.finditer(pattern, text_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    item_text = match.group(0)[:500]
                    clean_item = ' '.join(item_text.split())
                    if len(clean_item) > 50:
                        items_found.append(clean_item)
                if items_found:
                    break
            
            if items_found:
                for i, item in enumerate(items_found[:3], 1):
                    content_result += f"Item {i}: {item}...\n\n"
            else:
                clean_content = ' '.join(text_content.split())
                content_result += clean_content[:1000] + "...\n"
        
        else:
            content_result += "FILING CONTENT:\n"
            text_content = filing_soup.get_text()
            clean_content = ' '.join(text_content.split())
            content_result += clean_content[:1000] + "...\n"
        
        return content_result
        
    except Exception as e:
        return f"CONTENT EXTRACTION ERROR: {str(e)}"

def extract_10k_key_sections(soup):
    """Extract key sections from 10-K filing"""
    sections = {}
    text = soup.get_text()
    
    section_patterns = {
        "Business Overview": [
            r"item\s*1\s*[\.:\-]?\s*business",
            r"business\s*overview",
            r"description\s*of\s*business"
        ],
        "Risk Factors": [
            r"item\s*1a\s*[\.:\-]?\s*risk\s*factors",
            r"risk\s*factors"
        ],
        "Management Discussion": [
            r"item\s*7\s*[\.:\-]?\s*management",
            r"md&a",
            r"management.*discussion.*analysis"
        ],
        "Financial Statements": [
            r"item\s*8\s*[\.:\-]?\s*financial\s*statements",
            r"consolidated\s*statements",
            r"financial\s*statements"
        ]
    }
    
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start_pos = match.start()
                end_pos = min(start_pos + 2000, len(text))
                section_text = text[start_pos:end_pos]
                clean_text = ' '.join(section_text.split())
                
                if len(clean_text) > 200:
                    sections[section_name] = clean_text
                    break
            if section_name in sections:
                break
    
    return sections

if __name__ == "__main__":
    ticker = "GOOG"
    result = comprehensive_sec_analysis_tool(ticker, "10-K", include_content=True)
    print(result)