import scipy as sp
from bs4 import BeautifulSoup 
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

os.chdir("/Users/canderson/dev/amazon-registry-analysis/")

with open("in/grid.html", 'r',encoding="UTF+8") as f:
    html = f.read()

print("Processing...")
soup = BeautifulSoup(html, 'html.parser')

# Find all cards
cards = soup.find_all('div', class_='gr-card')

items = []

for card in cards:
    item = {}
    
    # Determine card type
    if 'wr-gift-fund-card' in card.get('class', []):
        item['card_type'] = 'Gift Fund'
        
        # Extract gift fund specific fields
        flag = card.find('div', class_='wr-gift-fund-card-flag')
        if flag:
            item['type'] = flag.text.strip()
        
        title = card.find('div', class_='wr-gift-fund-card__title')
        if title:
            item['title'] = title.text.strip()
        
        img = card.find('img', class_='wr-gift-fund-card__img')
        if img:
            item['image_url'] = img.get('src')
            item['image_alt'] = img.get('alt', '')
    
    elif 'registry-asin-card' in card.get('class', []):
        item['card_type'] = 'Registry Product'
        
        # Extract registry product specific fields
        title = card.find('div', class_='registry-asin-card__product-title')
        if title:
            item['title'] = title.text.strip()
        
        img = card.find('img', class_='registry-asin-card__img')
        if img:
            item['image_url'] = img.get('src')
            item['image_alt'] = img.get('alt', '')
        
        # Get link/aria-label for full product name
        link = card.find('a', class_='registry-asin-card__link')
        if link and link.get('aria-label'):
            item['full_title'] = link.get('aria-label')
    
    # Extract price (common to both types)
    price_whole = card.find('span', class_='a-price-whole')
    price_fraction = card.find('span', class_='a-price-fraction')
    if price_whole:
        price = price_whole.text.strip()
        if price_fraction:
            price += '.' + price_fraction.text.strip()
        item['price'] = price
    
    if item:
        items.append(item)

print("Done")

tab = pd.DataFrame(items)[["card_type", "title", "image_url", "image_alt", "price", "full_title"]]
tab['price'] = [float(re.sub(r",|\.\.", "", x)) for x in tab.price]
tab.to_csv("out/amazon-registry.csv", index=False)

print("Saved")