import re

with open('ai-router/app/router.py', 'r') as f:
    content = f.read()

gemini_router_code = """
import os
import json

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@dataclass(frozen=True)
class GeminiLLMRouter:
    \"\"\"Routes queries using Gemini 1.5 Flash for intelligent agentic orchestration.\"\"\"

    def route(self, request: ToolRequest) -> RoutingDecision:
        if not GEMINI_AVAILABLE:
            raise RoutingError("google-generativeai or dotenv not installed.")
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        
        prompt = f\"\"\"
        You are the Agentic Controller for an ISRO remote sensing platform.
        Analyze the user's query and select the appropriate specialist tool.
        
        User Query: "{request.query}"
        Number of images provided: {len(request.images)}
        
        Allowed Tools:
        1. "single_image_vqa" - For describing or answering questions about ONE image.
        2. "run_change_detection" - For finding changes between TWO images (T1 and T2).
        3. "run_optical_sar_fusion" - For combining TWO images (one optical, one radar).
        
        Return ONLY a valid JSON object matching this schema:
        {{"selected_tool": "name_of_tool", "parameters": {{}}}}
        \"\"\"
        
        try:
            response = model.generate_content(prompt)
            decision_dict = json.loads(response.text)
            selected_tool = decision_dict["selected_tool"]
        except Exception as e:
            raise RoutingError(f"Gemini routing failed: {str(e)}")
            
        # Ensure it's valid in the registry
        DeterministicKeywordRouter._validate_candidate(request, selected_tool)
        
        return RoutingDecision(
            selected_tool=selected_tool,
            parameters=decision_dict.get("parameters", {})
        )

def route_request(request: ToolRequest) -> RoutingDecision:
    \"\"\"Route a request using Gemini if available, else fallback to deterministic.\"\"\"
    if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        return GeminiLLMRouter().route(request)
    return DeterministicKeywordRouter().route(request)
"""

# Replace the old route_request function and append the Gemini router
content = content.replace(
'''def route_request(request: ToolRequest) -> RoutingDecision:
    """Route a request with the default deterministic strategy."""
    return DeterministicKeywordRouter().route(request)''',
gemini_router_code)

with open('ai-router/app/router.py', 'w') as f:
    f.write(content)
