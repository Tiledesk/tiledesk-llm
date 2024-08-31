import httpx
import json
import re

def get_graphql_answer(input, url, api_key):
    print(f"==========>  {input}")
    #print("==========>" + url)
    #print("==========>" + api_key)

    headers = {
       "content-type": "application/json",
       "X-Shopify-Access-Token": api_key}

    print(f"========= {input}")
    gql_query = clean_graphql_query(input)
    #gql_query = input
    #inputparam = json.loads(input)
    #response = httpx.post(url, json=inputparam["query"], headers=headers)

    # Print the response
    #print(response.json())

    from gql import gql, Client
    from gql.transport.httpx import HTTPXTransport #aiohttp import AIOHTTPTransport

    # Select your transport with a defined url endpoint
    #transport = AIOHTTPTransport(url=url, headers=headers)
    transport = HTTPXTransport(url=url, headers=headers)

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # Provide a GraphQL query
    print(f"========= {gql_query}")
    query = gql(gql_query)
    #print(f"========= query gql {query}")
    # Execute the query on the transport

    result = client.execute(query)
    print(f"risultato query {result}")
    return result
    #print(f"endpoint: {url}, api_key: {api_key}")




import re

def clean_graphql_query(query):
    # Remove leading and trailing whitespace
    query = query.strip()

    # Handle the case where query is in the format query='QUERY'
    match = re.match(r'^(?:const\s+)?(?:var\s+)?(?:let\s+)?(?:query\s*=\s*[\'"]*)(.*?)([\'"]*)\s*$', query, re.DOTALL)
    if match:
        query = match.group(1)

    # Case 1 and 2: Remove enclosing backticks and language specifiers
    query = re.sub(r'^```(?:graphql|query)?\s*', '', query)
    query = re.sub(r'\s*```$', '', query)

    # Remove any remaining backticks
    query = query.replace('`', '')

    # Unescape any escaped quotes
    query = query.replace('\\"', '"').replace("\\'", "'")

    # Ensure the query is properly closed
    open_braces = query.count('{')
    close_braces = query.count('}')
    if open_braces > close_braces:
        query += '}' * (open_braces - close_braces)

    return query.strip()
