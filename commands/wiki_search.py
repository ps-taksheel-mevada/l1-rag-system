import requests
from bs4 import BeautifulSoup

def wikipedia_search(query, limit=10):
    url = "https://en.wikipedia.org/w/rest.php/v1/search/title"

    headers = {
        "sec-ch-ua-platform": '"Windows"',
        "Referer": "https://en.wikipedia.org/wiki/Data",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
        "accept": "application/json",
        "sec-ch-ua": '"Not:A-Brand";v="99", "Microsoft Edge";v="145", "Chromium";v="145"',
        "sec-ch-ua-mobile": "?0"
    }

    params = {
        "q": query,
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Request failed with status {response.status_code}",
            "details": response.text
        }


import requests
from bs4 import BeautifulSoup

def scrape_pages(data):
    url = "https://en.wikipedia.org/wiki/"
    scraped = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
    }

    for page in data["pages"]:
        if "key" in page.keys():
            res = requests.get(url + page["key"], headers=headers)

            soup = BeautifulSoup(res.content, features="html.parser")
            scraped.append({"key" : page["key"],"content" : soup.find("div", {"id": "mw-content-text"}).text})

    return scraped