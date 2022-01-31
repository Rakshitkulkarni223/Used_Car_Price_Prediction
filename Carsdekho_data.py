#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import time
import requests


# In[2]:


header={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
}


# In[3]:


cities=['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur',
       'Kochi', 'Kolkata', 'New Delhi', 'Bangalore', 'Ahmedabad']


# In[4]:


car_details = []


# In[15]:


car_details=[]

for city in cities:
    
    options=Options()
    driver=webdriver.Chrome(executable_path=r"C:\Users\raksh\Downloads\chromedriver.exe")
    url="https://www.cardekho.com/used-cars+in+{}".format(city)
    driver.get(url)
    
    ScrollNumber = 5
    
    for i in range(1,ScrollNumber):
        print(i)
        driver.execute_script("window.scrollTo(1,50000)")
        time.sleep(1)

    file = open('cardekho.html', 'w',encoding='utf-8')
    file.write(driver.page_source)
    file.close()

    driver.close()
    
    data = open('cardekho.html','r')
    
    soup = BeautifulSoup(data, "lxml")
    
    car_list=soup.find_all("div",class_="holder hover")
    
    car_links=[]
    
    for car in car_list:
        for link in car.find_all("a",href=True):
            
            test_link=link['href']
            
            if 'https://www.cardekho.com/' not in test_link:
                test_link='https://www.cardekho.com'+test_link

            car_links.append(test_link)
    

    for testlink in car_links:

        r=requests.get(testlink,headers=header)
        soup=BeautifulSoup(r.content,"lxml")

        name=soup.find("h1").text.strip()
        
        price=soup.find("div",class_="priceSection")
        try:
            price=price.text.strip().split(' ')[1].replace(',','')
            if "Fixed" in price:
                price=price.split('F')[0]
        except:
            continue
            
        specifications=soup.find_all("div",class_="listIcons")


        spec=[]
        Name=' '.join(name.split(' ')[1:])
        Price=float(price)
        
        Location=city
    
        
        f=1
        for details in specifications:


            spec_name=details.find("div",class_="head").text.strip()
            spec_value=details.find("div",class_="fontweight500").text.strip()

            if spec_name=="Make Year":
                Year=int(spec_value)
            if spec_name=="KMs Driven":
                spec_value=int(spec_value.split(' ')[0].replace(',',''))
                Kilometers_Driven=spec_value
            if spec_name=="Fuel":
                Fuel_Type=spec_value
            if spec_name=="Transmission":
                Transmission=spec_value
            if spec_name=="No Of Owner(s)":
                Owner_Type=spec_value
            if spec_name=="Mileage":
                Mileage=spec_value
            if spec_name=="Engine":
                Engine=spec_value
            if spec_name=="Max Power" and f:
                f=0
                if "bhp" in spec_value:
                    spec_value=spec_value.split('b')
                    Power=(float(spec_value[0]))
                else:
                    Power="NA"
                
            if spec_name=="Seats":
                Seats=int(spec_value)

        spec.extend([Name,Location,Year,Kilometers_Driven,Fuel_Type,Transmission,Owner_Type,Mileage,Engine,Power,Seats,Price])
        print(spec)
        car_details.append(spec)
        print(len(car_details))


# In[14]:


import csv


columns=["Name","Location","Year","Fuel_Type","Kilometers_Driven","Owner_Type","Transmission","Mileage","Engine","Power","Seats","Price"]



with open('data/dataset.csv', 'a',encoding='utf-8',newline='') as f:
    write = csv.writer(f)
    write.writerow(columns)
    write.writerows(car_details)

print("\n\nDone!!\n\n")

