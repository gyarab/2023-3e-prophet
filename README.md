# 2023-3e-btcbot
Ročníkový projekt 2023/2024 3E Procházka, Petřík, Trávníček  

# <img src="static/img/Logo.png" alt="drawing" style="width:50px;"/>  Prophet

## Zadání projektu
Cílem tohoto projektu bylo vytvořit Python desktop aplikaci, která má přístup k online krypto burze a komunikuje s ní. V jádru aplikace bude AI, které sleduje různá reálná data z online krypto burzy. Na základě rozhodnutí tohoto AI bude aplikace hospodařit se simulovaným portfoliem. Aplikace poběží lokálně a bude mít plně funkční webové uživatelské rozhraní.

## Spuštění projektu:
1. git clone https://github.com/gyarab/2023-3e-prophet.git
2. Stáhněte si požadované knihovny (doporučuji udělat si virtuální prostředí)
3. Spusťte soubor app.py / napište do terminálnu ``` python app.py ```
4. Otevřete v prohlížeči lokální server [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Použité knihovny projektu
Pro stáhnutí všech potřebných závislostí:
```  pip3 install -r requirements.txt  ``` 

- flask
- numpy
- torch
- pandas
- matplotlib
- python-binance

## Struktura 
-  Frontend - HTML s  [Bootstrapem](https://getbootstrap.com/docs/5.0), CSS & JavaScript
-  Backend - Python
-  Framework - [Flask](https://flask.palletsprojects.com/en/3.0.x/)
