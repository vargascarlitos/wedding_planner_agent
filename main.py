import asyncio


from re import search
from typing import Any, Dict
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.runtime import Runtime
from pydantic.v1 import tools
from tavily import TavilyClient
from langchain.agents import AgentState
from langchain.tools import ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase 
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
import pprint
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

# Instacia el cliente para realizar busquedas en internet.
# Creo que es un MCP
tavily_client = TavilyClient()


# Aca se crea una clase que extiende de AgentState
# Es para gestionar estados en langchain con el LLM
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str
    
    
# Se instancia un cliente MCP para buscar vuelos 
client_mcp = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    },
)

# Se define una funcion asincrona para recuperar los tools del MCP
# search
# query
async def _get_mcp_tools():
    return await client_mcp.get_tools()

# se guarda los tolls del MCP
tools_mcp = asyncio.run(_get_mcp_tools())


# se crea un tool calling para realizar busquedas en internet 
# se usa el tavily client
@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error al hacer la consulta"
    


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

        
search_spaces_agent = create_agent(model='gpt-5-nano', 
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

playlist_agent = create_agent(model="gpt-5-nano",
                            tools=[query_playlist_db],
                            system_prompt="""
                                Eres un experto en listas de reproducción. Consulta la base de datos SQL y crea la 
                                lista perfecta para una boda, especificando el género musical.
                                Una vez que tengas tu lista, calcula su duración total y su costo; cada canción tiene un precio asociado.
                                Si encuentras errores al consultar la base de datos, intenta solucionarlos modificando la consulta.
                                No te rindas; sigue consultando la base de datos hasta que encuentres una lista de canciones.
                                Esta es una base de datos SQLite. Antes de escribir cualquier consulta, familiarízate con el esquema.
                            """)  

##  Coordinadores principales de los sub agents.
## Aca creamos tools pero que manejan a los sub agents segun nuestra logica de negocio
## Tambien cambiamos lo que seria el State 
## Creo ajaj

@tool
def search_travels(runtime: ToolRuntime) -> str:
    """Invoca al travel_agent para realizar busquedas de vuelos al destino indicado"""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = travel_agent.invoke({"messages": [HumanMessage(content=f"Busca vuelos de {origin} a {destination}")]})
    return response["messages"][-1].content


@tool
def search_spaces(runtime: ToolRuntime) -> str:
    """El agente del lugar elige el mejor lugar para la ubicación y la capacidad determinadas"""
    destination = runtime.state["destination"]
    capacity = runtime.state["guest_count"]
    query = f"Busca lugares en {destination} con capacidad de {capacity}"
    response = search_spaces_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def suggest_playlist(runtime: ToolRuntime) -> str:
    """El agente de listas de reproducción selecciona la lista de reproducción perfecta para el género determinado."""
    genre = runtime.state["genre"]
    response = playlist_agent.invoke({"messages": [HumanMessage(f"Busca play list de bodas con el genero {genre}")]})
    return response["messages"][-1].content

@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """Update the state when you know all of the values: origin, destination, guest_count, genre. 
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""
    return Command(update={
        "origin": origin, 
        "destination": destination, 
        "guest_count": guest_count, 
        "genre": genre, 
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
        )

## Agente orquestador
coordinator = create_agent(model = "gpt-5-nano",
                     tools=[search_travels, search_spaces, suggest_playlist, update_state],
                     state_schema=WeddingState,
                     system_prompt = """Eres coordinadora de bodas. 
                                        Primero, reúne toda la información necesaria para actualizar el estado. 
                                        Cuando la tengas, actualiza el estado. Una vez completado y 
                                        recibido el estado, puedes delegar las tareas a tus especialistas para vuelos,
                                        lugares y listas de reproducción. Cuando recibas sus respuestas,
                                        coordina la boda perfecta para mí.
                                        """)

response = coordinator.invoke({'messages': [HumanMessage(content="Soy de Londres y me gustaría una boda en París para 100 invitados, con música jazz.")]},
                              config={"tags": ["WP"], "recursion_limit": 40},)

#print(response)
#response.pretty_print()

for message in response["messages"]:
    message.pretty_print()