from langchain_core.prompts import PromptTemplate



from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from tilellm.tools.shopify_tool import get_graphql_answer
from tilellm.shared.const import react_prompt_template
from functools import partial


def lookup(question_to_agent, chat_model, chat_history:str):
    #You are an API agent.
    template1 = """
       Given the question {question} I want you to find the answer. the first step is to create a GraphQL query 
       for Shopify Admin client that answer the question, then pass the query to tool in GraphQL format. 
       (USE ONLY GraphQL format. No comment is needed! Use the parameter in the same language as the question).
       Use this schema {schema} for GraphQL and not exceed 10 items. 
       Examples of GraphQL query:  
       - query {{ products(first: 100) {{ edges {{ node {{ id title price }} }} }} }}
       - query {{ products(first: 10) {{ edges {{ node {{ id title }} }} }} }}
       - query {{ products(first: 10, query: "price:<50") {{ edges {{ node {{ title variants(first: 1) {{ edges {{ node {{ price }} }} }} }} }} }} }}

        
       In Your Final answer, use the same language of the question, the response should interpret and summarize the key information from the query result 
       in a clear and concise manner. If there isn't product that answer the question, simply say that there isn't products.
       """
    template="""
    Follow these instructions exactly to answer the question: {question}

    1. Create a GraphQL query for Shopify Admin client that answers the question.
       - Use ONLY GraphQL format, no comments.
       - Use parameters in the same language as the question.
       - Use this schema: {schema}
       - Limit results to a maximum of 10 items.
    
    2. Query format:
       query {{
         // Your code here
       }}
    
    3. Examples of valid queries:
       - query {{ products(first: 10) {{ edges {{ node {{ id title price }} }} }} }}
       - query {{ products(first: 10, query: "price:<50") {{ edges {{ node {{ title variants(first: 1) {{ edges {{ node {{ price }} }} }} }} }} }} }}
    
    4. Present ONLY the GraphQL query, nothing else.
    
    5. After receiving the query results, provide the final answer:
       - Use the same language as the original question.
       - Interpret and summarize key information from the query results.
       - Be clear and concise.
       - If there are no products that answer the question, state this explicitly.
    
    Remember: Follow these instructions to the letter. Do not add explanations or comments that are not requested.
    """
    question = question_to_agent.question

    for tool in question_to_agent.tools:
        if 'shopify' in tool:
            shopify_tool = tool['shopify']

            # Safely access the root dictionary
            api_key = shopify_tool.root.get('api_key')
            url = shopify_tool.root.get('url')
            break  # Exit the loop once 'shopify' is found

    get_graphql_answer_with_key = partial(get_graphql_answer,  url=url, api_key=api_key)
    tools_for_agent_shopify = [
        Tool(
            name="Retrieve content from Shopify given GraphQL query",
            func=get_graphql_answer_with_key,
            description="useful when you need get the useful information from shopify"

        ),
    ]

    schema = """
    products(first: 10, query: "") {
    edges {
      node {
        title
        variants(first: 1, last: 10) {
          edges {
            node {
              price
              availableForSale
              barcode
              displayName
              id
            }
          }
        }
        bodyHtml
        descriptionHtml
        id
        productType
        tags
        totalInventory
      }
    }
  }
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "schema"], template=template
    )

    #react_prompt = hub.pull("hwchase17/react")
    react_prompt = PromptTemplate.from_template(react_prompt_template)

    agent = create_react_agent(
        llm=chat_model, tools=tools_for_agent_shopify, prompt=react_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent_shopify,
        verbose=True,
        max_iterations=4,
        early_stopping_method="force",
        handle_parsing_errors=True
    )

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(question=question, schema=schema),
               "chat_history": chat_history}
    )

    return result
