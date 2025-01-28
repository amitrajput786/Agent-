# step 1 :setup API keys for Groq , openai and  Tavily 

import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")



# step 2 : Setup LLM & Tools 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)



# step3: setup AI agent with search tool functionality 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage


system_prompt ="Act as an Ai chatbot who is smart and angry "

# now we are creating our agent 
agent=create_react_agent(
    model=groq_llm,# it is brain of llm 
    tool=[search_tool],
    state_modifier=system_prompt

)
# now we are sending the query  to test it 
query="Tell me about trends of Ai in world  "
# how to invoke it 
state={"messages": query }
response=agent.invoke(state)
messages=response.get("messages")
ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]


print(ai_messages[-1])
