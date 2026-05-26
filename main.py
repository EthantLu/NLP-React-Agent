from agent.llm import HelloAgentsLLM
from agent.react import ReActAgent
from tool.tools import ToolExecutor, search

    
if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为在2026年发布了哪些手机？他们的主要卖点是什么？"
    agent.run(question)