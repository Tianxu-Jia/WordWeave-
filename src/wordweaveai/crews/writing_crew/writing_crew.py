import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

from wordweaveai.types import Chapter


@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

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
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm=self.deepseek_llm,
        )

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"],
        )

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Write Book Chapter Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
