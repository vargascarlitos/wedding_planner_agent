import asyncio


from typing import Any, Dict
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic.v1 import tools
from tavily import TavilyClient
from langchain.agents import AgentState
from langchain.tools import ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from dotenv import load_dotenv
import pprint
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_client = TavilyClient()
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str

client_mcp = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    },
)

async def _get_mcp_tools():
    return await client_mcp.get_tools()


tools_mcp = asyncio.run(_get_mcp_tools())


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


# Sub-agente para buscar viajes en internet. 
travel_agent = create_agent(model='gpt-5-nano', 
                          tools=tools_mcp, 
                          system_prompt="""Eres agente de viajes. Busca vuelos al destino de la boda.
                                        No puedes hacer preguntas adicionales; debes encontrar las mejores opciones de 
                                        vuelo según los siguientes criterios:
                                        - Precio (el más bajo, en clase económica)
                                        - Duración (el más corto)
                                        - Fecha (la época del año que consideres más adecuada para una boda en este lugar)
                                        Para simplificar, busca solo un billete de ida.
                                        Es posible que tengas que realizar varias búsquedas para encontrar las mejores opciones.
                                        No recibirás información adicional, solo el origen y el destino. Tu tarea es analizar críticamente las mejores opciones.
                                        Si la herramienta MCP falla, devuelve resultados incorrectos o no te proporciona resultados útiles, inténtalo de nuevo.
                                        Una vez que hayas encontrado las mejores opciones, infórmale al usuario de tu lista reducida.
                                        """)

        
subagent_2 = create_agent(model='gpt-5-nano', 
                          tools=[web_search],
                          system_prompt="""
                            Eres especialista en la organización de eventos. Busca espacios en la ubicación y con la capacidad deseadas.
                            No puedes hacer más preguntas; debes encontrar las mejores opciones según los siguientes criterios:
                            - Precio (el más bajo)
                            - Capacidad (coincidencia exacta)
                            - Reseñas (las mejores)
                            Es posible que necesites realizar varias búsquedas para encontrar las mejores opciones.
                            Tienes un límite sugerido de 12 búsquedas web. Cuenta cada búsqueda que realices.
                            Después de 12 búsquedas, debes detener la búsqueda y resumir las mejores opciones que hayas encontrado hasta el momento.
                          """)  

## Tool Kit

@tool
def call_travel_agent(query: str) -> str:
    """Invoca al subagent_1 para realizar busquedas de vuelos al destino indicado"""
    response = travel_agent.invoke({"message": [HumanMessage(content=f"Buscar sitios en internet segun consulta {query}")]})
    return response["message"][-1].content


agent = create_agent(model = "gpt-5-nano",
                     tools=[call_travel_agent],
                     system_prompt = """Eres coordinadora de bodas. 
                                        Primero, reúne toda la información necesaria para actualizar el estado. 
                                        Cuando la tengas, actualiza el estado. Una vez completado y 
                                        recibido el estado, puedes delegar las tareas a tus especialistas para vuelos,
                                        lugares y listas de reproducción. Cuando recibas sus respuestas,
                                        coordina la boda perfecta para mí.
                                        """)

response = agent.invoke({'messages': [HumanMessage("Hola quiero realizar una boda,busca vuelos de Paraguay a hawai entre las fecha 12 de junio y 22 de junio de este año 2026")]})

#print(response)
#response.pretty_print()

for message in response["messages"]:
    message.pretty_print()