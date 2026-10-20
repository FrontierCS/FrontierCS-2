import argparse
import copy
import os
import sys
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.proxy import ProxyClient
from mcp import ErrorData
from mcp.types import TextContent
from .tool import check_tool_call, generate_security_policy, generate_update_security_policy, update_available_tools, update_always_allowed_tools, Tool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import threading
import httpx
import json
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import McpError
import re

mcp_server = FastMCP.as_proxy(
    ProxyClient("http://127.0.0.1:9001/mcp/"),
    name="Progent MCP Proxy"
)

rest_api = FastAPI(title="MCP API")

# Target server for proxying requests
API_PROXY_TARGET = "http://127.0.0.1"

ENABLE_SECAGENT = os.getenv("ENABLE_SECAGENT", "False").lower() == "true"

TOOL_SET = False
USER_INPUT_SET = False
USER_INPUT = None


@rest_api.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(path: str, request: Request):
    """Proxy all requests to the target server and print request details"""

    # Print request details
    # print("\n" + "="*50)
    # print(f"Proxying Request:")
    # print(f"Method: {request.method}")
    # print(f"Path: /{path}")
    # print(f"Full URL: {request.url}")
    # print(f"Headers: {dict(request.headers)}")

    # Get request body
    body = await request.body()
    if body:
        body_json = json.loads(body)
        # print(f"Body: {body_json}", file=sys.stderr)
        global USER_INPUT_SET, USER_INPUT
        if ENABLE_SECAGENT and not USER_INPUT_SET:
            messages = body_json.get("messages", [])
            try:
                content = messages[1]["content"]
                if isinstance(content, str):
                    user_input = content
                else:
                    user_input = content[0]["text"]
                print(f"User Input: {user_input}", file=sys.stderr)
                if TOOL_SET:
                    generate_security_policy(user_input)
                USER_INPUT = user_input
                USER_INPUT_SET = True
            except Exception as e:
                print(f"Error extracting user input: {e}", file=sys.stderr)

        # print(f"Body: {json.dumps(messages, indent=2)}")
    else:
        # print("Body: (empty)")
        pass

    print("="*50 + "\n")

    # Forward the request
    target_url = f"{API_PROXY_TARGET}/{path}"

    try:
        async with httpx.AsyncClient() as client:
            # Prepare headers
            headers = dict(request.headers)
            new_headers = {"Authorization": headers.get("authorization")}

            response = await client.request(
                method=request.method,
                url=target_url,
                headers=new_headers,
                json=body_json,
                params=request.query_params,
                timeout=600.0
            )

            # Print response details
            # print(f"Response Status: {response.status_code}")
            # print(f"Response Headers: {dict(response.headers)}")

            # Parse response content once
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    response_content = response.json()
                    # print(f"Response Body: {json.dumps(response_content, indent=2)}")
                except:
                    response_content = response.text
                    # print(f"Response Body: {response_content[:500]}")
            else:
                response_content = response.text
                # print(f"Response Body: {response_content[:500]}")
            # print("\n")

            # Return response with appropriate headers
            return JSONResponse(
                content=response_content if isinstance(response_content, dict) else {"data": response_content},
                status_code=response.status_code
            )

    except httpx.RequestError as e:
        # print(f"Error proxying request: {e}")
        raise HTTPException(status_code=502, detail=f"Error connecting to target server: {str(e)}")


def convert_mcp_tool_to_progent(tool) -> Tool:
    """Convert MCP tool to Progent tool format"""
    args = copy.deepcopy(tool.parameters["properties"])
    return Tool(
        name=tool.name,
        description=tool.description,
        args=args
    )


agent_expert_tool_list = {
    "general": ["get_user_info",
                "update_password",
                "update_user_info"],
    "workspace": ["read_file",
                  "send_email",
                  "delete_email",
                  "get_unread_emails",
                  "get_sent_emails",
                  "get_received_emails",
                  "get_draft_emails",
                  "search_emails",
                  "search_contacts_by_name",
                  "search_contacts_by_email",
                  "get_current_day",
                  "search_calendar_events",
                  "get_day_calendar_events",
                  "create_calendar_event",
                  "cancel_calendar_event",
                  "reschedule_calendar_event",
                  "add_calendar_event_participants",
                  "append_to_file",
                  "search_files_by_filename",
                  "create_file",
                  "delete_file",
                  "get_file_by_id",
                  "list_files",
                  "share_file",
                  "search_files"],
    "slack": ["get_channels",
              "add_user_to_channel",
              "read_channel_messages",
              "read_inbox",
              "send_direct_message",
              "send_channel_message",
              "get_users_in_channel",
              "invite_user_to_slack",
              "remove_user_from_slack"],
    "banking": ["get_iban",
                "send_money",
                "schedule_transaction",
                "update_scheduled_transaction",
                "get_balance",
                "get_most_recent_transactions",
                "get_scheduled_transactions"],
    "travel": ["get_user_information",
               "get_all_hotels_in_city",
               "get_hotels_prices",
               "get_rating_reviews_for_hotels",
               "get_hotels_address",
               "get_all_restaurants_in_city",
               "get_cuisine_type_for_restaurants",
               "get_restaurants_address",
               "get_rating_reviews_for_restaurants",
               "get_dietary_restrictions_for_all_restaurants",
               "get_contact_information_for_restaurants",
               "get_price_for_restaurants",
               "check_restaurant_opening_hours",
               "get_all_car_rental_companies_in_city",
               "get_car_types_available",
               "get_rating_reviews_for_car_rental",
               "get_car_fuel_options",
               "get_car_rental_address",
               "get_car_price_per_day",
               "reserve_hotel",
               "reserve_car_rental",
               "reserve_restaurant",
               "get_flight_information"],
    "web": ["get_webpage",
            "post_webpage"]
}

suite = os.getenv('SECAGENT_SUITE', None)

def init_policies():
    if suite == "banking":
        update_always_allowed_tools(["get_most_recent_transactions"], allow_all_no_arg_tools=True)
    elif suite == "slack":
        update_always_allowed_tools(["get_channels", "read_channel_messages", "read_inbox", "get_users_in_channel"])
    elif suite == "travel":
        update_always_allowed_tools([
            "get_user_information",
            "get_all_hotels_in_city",
            "get_hotels_prices",
            "get_rating_reviews_for_hotels",
            "get_hotels_address",
            "get_all_restaurants_in_city",
            "get_cuisine_type_for_restaurants",
            "get_restaurants_address",
            "get_rating_reviews_for_restaurants",
            "get_dietary_restrictions_for_all_restaurants",
            "get_contact_information_for_restaurants",
            "get_price_for_restaurants",
            "check_restaurant_opening_hours",
            "get_all_car_rental_companies_in_city",
            "get_car_types_available",
            "get_rating_reviews_for_car_rental",
            "get_car_fuel_options",
            "get_car_rental_address",
            "get_car_price_per_day",
            "search_calendar_events",
            "get_day_calendar_events",
            "get_flight_information"
        ])
    elif suite == "workspace":
        update_always_allowed_tools([
            "get_unread_emails",
            "get_sent_emails",
            "get_received_emails",
            "get_draft_emails",
            "search_emails",
            "search_contacts_by_name",
            "search_contacts_by_email",
            "get_current_day",
            "search_calendar_events",
            "get_day_calendar_events",
            "search_files_by_filename",
            "get_file_by_id",
            "list_files",
            "search_files"
        ])
    else:
        raise ValueError(f"Unknown suite: {suite}")



class ListingFilterMiddleware(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        print("ListingFilterMiddleware: on_list_tools called", file=sys.stderr)
        result = await call_next(context)

        global TOOL_SET
        if ENABLE_SECAGENT and not TOOL_SET:
            tool_list = []
            for tool in result:
                progent_tool = convert_mcp_tool_to_progent(tool)
                tool_list.append(progent_tool)
            update_available_tools(tool_list)
            init_policies()
            if USER_INPUT_SET:
                generate_security_policy(USER_INPUT)
            TOOL_SET = True

        headers = get_http_headers()
        agent_expert = headers.get('agent_expert', None)
        if agent_expert:
            print(f"Agent Expert: {agent_expert}", file=sys.stderr)
            expert_tools = []
            if agent_expert in ["general", "workspace", "slack", "banking", "travel", "web"]:
                for tool in result:
                    if tool.name in agent_expert_tool_list[agent_expert]:
                        expert_tools.append(tool)
                return expert_tools

        return result


class ToolCallFilterMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        print("ToolCallFilterMiddleware: on_call_tool called", file=sys.stderr)
        print(f"Tool Call: {context}", file=sys.stderr)
        tool_name, tool_kwargs = context.message.name, context.message.arguments
        check_tool_call(tool_name, tool_kwargs)

        result = await call_next(context)

        flag = os.getenv('SECAGENT_UPDATE', "True").lower() == "true"
        if ENABLE_SECAGENT and flag:
            last_tool_call = {"name": tool_name, "args": tool_kwargs}
            if isinstance(result.content[0], TextContent):
                print(f"Tool Result: {result.content[0].text}", file=sys.stderr)
                generate_update_security_policy([last_tool_call], result.content[0].text, manual_check=False)
        return result


mcp_server.add_middleware(ListingFilterMiddleware())
mcp_server.add_middleware(ToolCallFilterMiddleware())


def run_rest_api(port: int):
    """Run the REST API server"""
    print(f"Starting REST API server on port {port}")
    uvicorn.run(rest_api, host="0.0.0.0", port=port, log_level="error")


def run_mcp_proxy(port: int):
    """Run the MCP proxy server"""
    print(f"Starting MCP Proxy Server on port {port}")
    mcp_server.run(
        transport="http", host="0.0.0.0", port=port, log_level="ERROR", stateless_http=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Progent Proxy"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=9100,
        help="Port to run the API server on (default: %(default)s)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=9101,
        help="Port to run the MCP server on (default: %(default)s)",
    )
    parser.add_argument(
        "--api-proxy-target",
        type=str,
        default="https://api.openai.com",
        help="Target URL for API proxy (default: %(default)s)",
    )

    args = parser.parse_args()

    API_PROXY_TARGET = args.api_proxy_target

    rest_thread = threading.Thread(
        target=run_rest_api,
        args=(args.api_port,),
        daemon=True
    )
    rest_thread.start()

    run_mcp_proxy(args.mcp_port)
