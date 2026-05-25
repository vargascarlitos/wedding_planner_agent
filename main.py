from typing import Any, Dict
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_client = TavilyClient()

system_prompt = '''Rol: Eres un experto en orquestar sub agentes para planear bodas, tienes a disposición tres agentes: 
                        1- Agente que halla vuelos para el destino ideal
                        2- Agente que busca en la web espacios para bodas
                        3- Agente que hace de DJ que rastrea musica para una lista que encaje con el genero y tematica de la boda
                    
                    Tu trabajo es orquestar estos agentes para poner en marcha la visión de los usuarios que quieren casarse y quieren realizarte consultas
                
                '''
                





                
@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


subagent_1 = create_agent(model='gpt-5-nano', tool=[web_search], system_prompt='Eres un subagente especialista en buscar vuelos para el destino ')

        
subagent_2 = create_agent(model='gpt-5-nano', tool=[web_search])                


agent = create_agent(model = "gpt-5-nano",
                     system_prompt = system_prompt,
                     
                     )

response = agent.invoke({'messages': [HumanMessage("Hola este es mi primer prompt con langchain")]})

print(response)