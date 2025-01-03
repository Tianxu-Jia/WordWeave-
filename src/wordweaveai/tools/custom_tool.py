from typing import Any, Optional, Type
import json
import requests
import datetime
import os

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from duckduckgo_search import ddg


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    filename = f"search_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(filename, "w") as file:
        file.write(content)
    print(f"Results saved to {filename}")


class DuckduckGoGoDevToolSchema(BaseModel):
    """Input for DuckduckGoGoDevToolSchemaDevTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )



class DuckduckGoGoDevTool(BaseTool):
    name: str = "Search the internet via DuckDuckGoGo"
    description: str = (
        "A tool that can be used to search the internet with a search_query using DuckDuckGoGo."
    )
    args_schema: Type[BaseModel] = DuckduckGoGoDevToolSchema
    search_url: str = "https://google.serper.dev/search"
    country: Optional[str] = ""
    location: Optional[str] = ""
    locale: Optional[str] = ""
    n_results: int = 10
    save_file: bool = False

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:

        search_query = kwargs.get("search_query") or kwargs.get("query")
        save_file = kwargs.get("save_file", self.save_file)
        n_results = kwargs.get("n_results", self.n_results)

        payload = {"q": search_query, "num": n_results}

        if self.country != "":
            payload["gl"] = self.country
        if self.location != "":
            payload["location"] = self.location
        if self.locale != "":
            payload["hl"] = self.locale

        payload = json.dumps(payload)

        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "content-type": "application/json",
        }

        response = requests.request(
            "POST", self.search_url, headers=headers, data=payload
        )
        results = response.json()

        if "organic" in results:
            results = results["organic"][: self.n_results]
            string = []
            for result in results:
                try:
                    string.append(
                        "\n".join(
                            [
                                f"Title: {result['title']}",
                                f"Link: {result['link']}",
                                f"Snippet: {result['snippet']}",
                                "---",
                            ]
                        )
                    )
                except KeyError:
                    continue

            content = "\n".join(string)
            if save_file:
                _save_results_to_file(content)
            return f"\nSearch results: {content}\n"
        else:
            return results
