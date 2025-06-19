print('successfully called scraper.py')
headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            }

# Getting all searching pages
import requests
responses = []
session = requests.Session()
for property in ['house','apartment']:
    for epc in ['A++','A+','A','B','C','D','E','F','G']:
        page_num = 1
        while True:
            url = 'https://www.immoweb.be/en/search/'+property+'/for-sale?countries=BE&epcScores='+epc+'&page='+str(page_num)+'&orderBy=relevance'
            r = session.get(url, headers = headers)
            if ":results='[]'" in r.text:
                print(f'{property} with epc score {epc} has {page_num} search pages')
                break
            responses.append(r)
            page_num += 1
print(f'we got {len(responses)} searching pages')

# Getting all urls from each searching page
from bs4 import BeautifulSoup
urls_of_properties = []
for r in responses:
    soup = BeautifulSoup(r.text,'lxml')
    for tag in soup.find_all('a', attrs={'class':'card__title-link','tabindex': False}):
        urls_of_properties.append(tag.get('href'))
print(f'we got {len(urls_of_properties)} property links')
print('start scraping property info from each link')

# Scraping data from each url
import csv
f = open('data/raw.csv', 'a', newline='')
writer = csv.writer(f, delimiter = ',')
writer.writerow(['Property ID',
                 'Locality name',
                 'Postal code',
                 'Price',
                 'Type of property',
                 'Subtype of property',
                 'Number of rooms',
                 'Living area',
                 'Equipped kitchen',
                 'Furnished',
                 'Open fire',
                 'Terrace',
                 'Garden',
                 'Surface of good',
                 'Number of facades',
                 'Swimming pool',
                 'State of building',
                 'EPC score',
                 'Heating type'])

def helper(func): # Helper function to handel errors
    try:
        return func()
    except:
        return None # return None if there is an error

import re
import json
for url in urls_of_properties:
    try:
        r = session.get(url, headers = headers)
        json_text = re.findall('window\\.classified = \\{.*?\\};',r.text)[0][20:-1]
        json_parser = json.loads(json_text)
        if json_parser['flags']['isLifeAnnuitySale']==True:
            continue # exclude life sales
        row = []
        row.append(helper(lambda: int(url.split('/')[-1]))) # Property ID - int
        row.append(helper(lambda: url.split('/')[-3])) # Locality name - str
        row.append(helper(lambda: int(url.split('/')[-2]))) # Postal code - int
        row.append(helper(lambda: int(json_parser['transaction']['sale']['price']))) # Price - int
        row.append(helper(lambda: json_parser['property']['type'])) # Type of property - str
        row.append(helper(lambda: json_parser['property']['subtype'])) # Subtype of property - str
        row.append(helper(lambda: int(json_parser['property']['bedroomCount']))) # Number of rooms - int
        row.append(helper(lambda: int(json_parser['property']['netHabitableSurface']))) # Living area - int
        row.append(helper(lambda: 1 if json_parser['property']['kitchen']['type']=='INSTALLED' else 0)) # Equipped kitchen - int 0/1
        row.append(helper(lambda: 1 if json_parser['transaction']['sale']['isFurnished']==True else 0)) # Furnished - int 0/1
        row.append(helper(lambda: 1 if json_parser['property']['fireplaceExists']==True else 0)) # Open fire - int 0/1
        row.append(helper(lambda: int(json_parser['property']['terraceSurface'])) if json_parser['property']['hasTerrace']==True else None) # Terrace - int/null
        row.append(helper(lambda: int(json_parser['property']['gardenSurface'])) if json_parser['property']['hasGarden']==True else None) # Garden - int/null
        row.append(helper(lambda: int(json_parser['property']['land']['surface']))) # Surface of good - int
        row.append(helper(lambda: int(json_parser['property']['building']['facadeCount']))) # Number of facades - int
        row.append(helper(lambda: 1 if json_parser['property']['hasSwimmingPool']==True else 0)) # Swimming pool - int 0/1
        row.append(helper(lambda: json_parser['property']['building']['condition'])) # State of building - str
        # Below are extra collected info
        row.append(helper(lambda: json_parser['transaction']['certificates']['epcScore'])) # EPC score - str
        row.append(helper(lambda: json_parser['property']['energy']['heatingType'])) # Heating type - str
        writer.writerow(row)
    except:
        continue # ensure the loop can continue to next url
f.close()

print('scraping finished')
