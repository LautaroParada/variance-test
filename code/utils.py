import bs4 as bs
import requests
import csv

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # create the soup object, we need the lxml parser
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    # the first row is the title of each variable
    for row in table.findAll('tr')[1:]:
        # just grab the ticker name
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    print(tickers)

    return tickers

save_sp500_tickers()