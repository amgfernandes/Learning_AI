# %%
import pandas as pd
from bs4 import BeautifulSoup
import requests
# %%
url = 'https://www.neurovasc.ch/portrait/stroke-center-stroke-units'
req = requests.get(url)
print(req) #To verify output
# %%
soup = BeautifulSoup(req.content, 'html.parser')
print(soup.prettify())
#print(soup) #To verify output (VERY LONG)
# %%
find = soup.findAll('div', class_='col-outer col-md-6')
find
# %%
for job_element in find:
    print(job_element, end="\n"*2)
# %%
spital=soup.find('div', class_='col-outer col-md-6')
spital
# %%
spital.text.strip()
# %%
spital.prettify()


# %%
