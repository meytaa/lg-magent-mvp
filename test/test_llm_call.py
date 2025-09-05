from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # type: ignore
load_dotenv()

llm = ChatOpenAI()
print(llm.invoke("Hello, world!"))
