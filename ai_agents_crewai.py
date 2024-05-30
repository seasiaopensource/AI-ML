# -*- coding: utf-8 -*-
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun


llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.6,
                             google_api_key="")

search_tool = DuckDuckGoSearchRun()
# Define agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,  #using google gemini pro API
  tools=[
        search_tool
      ])

writer = Agent(
    role='AI developer',
    goal='Develop innovative AI solutions that address real-world challenges',
    backstory="""You are an accomplished AI developer with a passion for pushing the boundaries of artificial intelligence.
    You have a deep understanding of machine learning algorithms, neural networks, and natural language processing techniques.
    Your expertise lies in designing, implementing, and testing AI systems that solve complex problems in various domains,
    including healthcare, finance, education, and robotics. Your goal is to create AI that has a positive impact on society and
    improves people's lives.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[]
)

task1 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
)

result = crew.kickoff()
