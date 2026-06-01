from toolregistry import ToolRegistry, Tool
from toolregistry.tool_registry import TOOL_DISCOVERY_NAME

class ToolzSearch:

    def __init__(
        self,
        registry: ToolRegistry
    ):
        """
        ToolzSearch that uses LLM to build intelligent index to do run-time querying.
        """
        self.registry = registry


    def build_index(self):
        
        for name, tool in self.registry._tools.items():
            if name == TOOL_DISCOVERY_NAME:
                continue
            
    
