from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

# from langchain_openai import ChatOpenAI
import os

from wordweaveai.types import BookOutline


@CrewBase
class OutlineCrew:
    """Book Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # llm = ChatOpenAI(model="gpt-4o")
    deepseek_llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=1.5,
    )

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool()
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm=self.deepseek_llm,
            verbose=True,
        )

    @agent
    def outliner(self) -> Agent:
        return Agent(
            config=self.agents_config["outliner"],
            llm=self.deepseek_llm,
            verbose=True,
        )

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
        )

    @task
    def generate_outline(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline"], output_pydantic=BookOutline
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Book Outline Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
