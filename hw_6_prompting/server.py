import logging
import os


import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from src.config import get_config
from src.langchain_model import LangChainModel
from src.pydantic_classes import Result, Query, FreeTextQuery

system_config = get_config("config.yaml")

logger = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()

model = LangChainModel(system_config)

os.environ["API_KEY"] = os.getenv("API_KEY")


@app.get("/healthcheck")
def root():
    logger.info("Got healthcheck request")
    return {"message": "Status: OK"}


@app.post("/api/ml-service/parse_queries")
def parse_queries(query: Query):
    logger.info(f"Got request: {query}")
    try:
        answer = model.parse_queries(query)
    except Exception as ex:
        logger.error(f"Something went wrong: {ex}")
        answer = Result(id=query.id, errors=[str(ex)])
    return answer.model_dump(mode="json")

@app.post("/api/ml-service/parse_free_text")
def parse_free_text(query: FreeTextQuery):
    logger.info(f"Got request: {query}")
    try:
        answer = model.parse_free_text(query)
    except Exception as ex:
        logger.error(f"Something went wrong: {ex}")
        answer = Result(id=query.id, errors=[str(ex)])
    return answer.model_dump(mode="json")


def main():
    print(system_config)
    host, port = system_config['server']['host'], system_config['server']['port']
    uvicorn.run(app, host=host, port=port, log_level='info', log_config="./log_conf.yaml")


if __name__ == "__main__":
    main()
