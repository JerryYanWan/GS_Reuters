import re, urllib
from bs4 import BeautifulSoup

class reutersParser(object):
    def __init__(self, ticker):
        self.prefix = "http://www.reuters.com"
        self.ticker = ticker

    def __getArticleUrl__(self, date):
        r = urllib.urlopen('%s/finance/stocks/companyNews?symbol=%s&date=%s' \
                  %(self.prefix, self.ticker, date))
        soup = BeautifulSoup(r, "html.parser")
        links = soup.find_all('a', href=re.compile('/article/[^\"]+type=companyNews'))
        lobbying_all, lobbying = [], []
        for link in links:
            lobbying_all.append(self.prefix + link["href"])
        lobbying = list(set(lobbying_all))
        #print len(lobbying)
        return lobbying

    def parse(self, date):
        try:
            lobbying = self.__getArticleUrl__(date)
        except:
            lobbying = []
            pass
        documents = []
        for url in lobbying:
            #print url
            r = urllib.urlopen(url)
            soup = BeautifulSoup(r, "html.parser")
            try:
                article = soup.find_all("span", id="articleText")[0]
                headline = soup.find_all("h1", class_="article-headline")[0]
                #print article.get_text()
                try:
                    tags = soup.find_all("meta", property="og:article:tag")[0]
                    #print tags["content"]
                    documents.append((headline.get_text(), article.get_text(), tags["content"], url))
                except:
                    print "ticker: %s, date: %s, no tags" % (self.ticker, date)
                    pass
            except:
                print "ticker: %s, date: %s, no articles" % (self.ticker, date)
                pass
        return documents

