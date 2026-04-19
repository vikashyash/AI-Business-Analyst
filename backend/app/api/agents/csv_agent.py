import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0)

def handle_dataframe_action(file):
    try:
        df=pd.DataFrame(file)
        agent=create_pandas_dataframe_agent(llm=model,df=df,verbose=True,allow_dangerous_code=True)
        response = agent.run()
        return response
    except Exception as e:
        return str(e)
