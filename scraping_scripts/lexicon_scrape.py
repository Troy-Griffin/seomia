import scrapy
from scrapy.crawler import CrawlerProcess
import json
import pandas as pd

class lexicon_spider(scrapy.Spider):
    name = "lexicon"
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    start_urls =[]
    for i, letter in enumerate(letters):
        start_urls.insert(i, f'https://www.engineering-dictionary.com/index.php?letter={letter}')
    
    def parse(self, response):
        for term in response.xpath("//body/ul[@id='letters']/li/a"):
            next_page = term.xpath('@href').extract()
            if next_page is not None:
                next_page = response.urljoin(next_page[0])
                yield scrapy.Request(next_page, callback=self.parsenext)
            
    def parsenext(self, response):
        check = response.xpath("//div[@id='definition']")
        if check != []:
            yield {"Term":check.xpath("p[1]/strong/text()").extract(), "Definition": check.xpath("p[2]/text()").extract()}

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "FEEDS": {
            "lexicon.json": {"format": "json"},
            #"items.jl": {"format": "jsonlines"},

        },
    })

    process.crawl(lexicon_spider)
    process.start() # the script will block here until the crawling is finished
   