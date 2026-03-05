to run this app you need ollama in your machine and uv python pkg manager 





use this to download ollama 



https://ollama.com/

login to ollama to from ui and then from cmd



to login from cmd use this 



```
ollama login

```



now to get both the models use this commands 

```

ollama pull mxbai-embed-large



ollama pull minimax-m2.5:cloud

```


to install uv use 

```
pip install uv
```

then to install dependencies use 

```
uv sync
```

or 
```
python -m uv sync
```

then run this to start your project 

```
uv run streamlit run main.py
```
or 
```
python -m uv run streamlit run main.py
```
 

