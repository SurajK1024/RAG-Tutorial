from tavily import TavilyClient
from dotenv import load_dotenv

import os

load_dotenv()

tavily_api_key = os.environ.get("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api_key)

# # 1. Searching -  used to find relevant information by the query
# search_results = tavily_client.search(
#         query="who is the johnny depp?",
#         search_depth="advanced",
#         include_domains=["linkedin.com", "github.com"]
#     )

# print("Search Results".center(50, "~"))
# print(search_results)

# # 2. Extration - used to extract specific information from the website

# extracted_data = tavily_client.extract(
#     urls=["https://www.jpmorganchase.com/about/leadership/linda-bammann"],
#     include_images=True,
#     format="markdown"
# )

# print("Extracted Data".center(50, "~"))
# print(extracted_data)
