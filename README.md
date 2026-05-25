# WebLogAnalyzer

Приложение для автоматического выявления и LLM-интерпретации паттернов
активности пользователей веб-сайтов на основе анализа веб-логов.

## Старт и запуск

```bash
cd web_log_analyzer

pip install -r requirements.txt

cp .env.example .env

# Запуск
bentoml serve service:WebLogAnalyzer
```

## Структура проекта

```
web_log_analyzer/
├── app.py               
├── pipeline/
│   ├── preprocessor.py      
│   ├── clustering.py       
│   ├── pattern_mining.py   
│   ├── anomaly.py          
│   └── interpreter.py      
├── utils/
│   ├── session_model.py    
│   └── url_normalizer.py  
├── data/                    
├── tests/                 
├── requirements.txt
└── .env.example
```



## Тесты

```bash
python -m pytest tests/ -v
```


