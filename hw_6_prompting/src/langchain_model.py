import copy
import logging
import os

from typing import Dict, List

import openai
import pandas as pd
from langchain_community.embeddings import TensorflowHubEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.model_create_request import create_request
from src.model_units_request import create_units_request
from src.parser_utils import generate_query_from_answer, response_to_dict, catalog_description_unification
from src.pydantic_classes import ResultItem, Result, ResultOrder, Query, FreeTextQuery
from src.units_parser_utils import maybe_manual_parse_units, postprocess_units_answer, \
    convert_units_to_metric

logger = logging.getLogger(__name__)


class LangChainModel:
    def __init__(self, config):
        logger.info("Starting: Loading LangChain model")
        self._faiss_vectorstore = None
        self._config = config
        self._init_client()
        self._load_and_parse_catalog()
        self._init_embedding_function()
        self._load_or_create_vectorstore()
        logger.info("Done: Loading LangChain model")

    def _init_client(self):
        client_config = self._config['client']
        self._client = openai.OpenAI(
            api_key=os.environ["API_KEY"],
            base_url=client_config['base_url'],
        )

    def _init_embedding_function(self):
        """
        Init embedding function
        :return:
        Эта функция инициализирует эмбеддинги. Эмбеддинги - это векторное представление текста.
        По началу оставьте ее как есть (
        """
        logger.info("Initializing embedding function...")
        embedding_config = self._config['embeddings']
        if embedding_config['model']['name'] == 'TensorflowHubEmbeddings':
            logger.info("Initializing TensorflowHubEmbeddings...")
            self._embedding_function = TensorflowHubEmbeddings(**embedding_config['model']['params'])
        elif embedding_config['model']['name'] == 'HuggingFaceEmbeddings':
            self._embedding_function = HuggingFaceEmbeddings(**embedding_config['model']['params'])
        else:
            raise ValueError(f"Unknown embedding model: {embedding_config['model']}")
        logger.info("Done: Initializing embedding function")

    def _load_and_parse_catalog(self):
        catalog_config = self._config['catalog']
        self._catalog = pd.read_csv(catalog_config['file'])
        self._catalog.columns = [c.strip().lower() for c in self._catalog.columns]
        for c in self._catalog.columns:
            try:
                self._catalog[c] = self._catalog[c].str.lower().str.strip()
            except AttributeError:
                pass
        description_column = catalog_config['description_column']
        self._original_catalog = copy.deepcopy(self._catalog)
        # Каталог изначально не прежначен для работы с моделью, его писали разные люди и он не стандартизирован
        # Поэтому может быть стоить привести его к какому-то стандарту
        self._catalog[description_column] = self._catalog[description_column].apply(catalog_description_unification)

    def _load_or_create_vectorstore(self):
        """
        Load or create vectorstore (FAISS DB).
        FAISS DB - это база данных, которая хранит эмбеддинги текстов и позволяет быстро искать похожие тексты
        Я уже написал для вас функцию, которая создает и загружает FAISS DB
        :return:
        """
        config = self._config['vectorstore']
        if config['create'] or not os.path.exists(self._config['vectorstore']['directory']):
            logger.info("Creating vectorstore...")
            docs = self._catalog[config['description_column']].str.lower().tolist()
            # docs = ["query: " + d for d in docs]
            self._faiss_vectorstore = FAISS.from_texts(
                texts=docs,
                embedding=self._embedding_function,
                metadatas=[{"source": 1}] * len(docs)
            )
            logger.info("Saving vectorstore to {}".format(self._config['vectorstore']['directory']))
            self._faiss_vectorstore.save_local(self._config['vectorstore']['directory'])
        logger.info("Loading vectorstore from {}".format(self._config['vectorstore']['directory']))
        self._faiss_vectorstore = FAISS.load_local(self._config['vectorstore']['directory'],
                                                   self._embedding_function,
                                                   allow_dangerous_deserialization=True)

    def _parse_units(self, text):
        # Просто чтобы показать, что можно использовать регулярки для извлечения единиц измерения
        # Но это не самый лучший способ, потому что регулярки не умеют понимать контекст
        # И даже для такой простой задачи, как извлечение единиц измерения, размер впечатляющий
        # result = maybe_manual_parse_units(text)
        # Давайте попробуем использовать модель для извлечения единиц измерения
        result = self._call_units_model(text)
        # Доработаем напильником
        result = postprocess_units_answer(result)

        # Попробуем перевести в стандартные метры и килограммы. Я уже сделал это за вас.
        # Ну если вы конечно передали мен что то разумное
        # Дополнительная задачка - перепишите эту функцию, чтобы она использовала модель
        result = convert_units_to_metric(result)
        # self._convert_units(result)
        return result

    def _call_units_model(self, text: str) -> Dict:
        """
        Call unit parser model
        @param text:
        @return:
        """
        logger.info(f"Calling units model with text: {text}")
        client_config = self._config['client']
        chat_completion = self._client.chat.completions.create(
            messages=create_units_request(text),
            **client_config['call_params']
        )
        response = chat_completion.choices[0].message.content
        response = response.split("\n")[0]
        logger.info(f"Got response: {response}")
        answer = response_to_dict(response)
        return answer

    def _call_model(self, text: str) -> Dict:
        """
        Тут мы вызываем модель, которая принимает на вход текст и возвращает словарь с полями:
        Я написал для вам вызов модели для https://www.together.ai/ если вы пользуетесь чем то
        другим, то нужно подставить свою функцию
        @param text:
        @return:
        """
        logger.info(f"Calling model with text: {text}")
        client_config = self._config['client']
        chat_completion = self._client.chat.completions.create(
            messages=create_request(text),
            **client_config['call_params']
        )
        response = chat_completion.choices[0].message.content
        response = response.split("\n")[0]
        logger.info(f"Got response: {response}")
        answer = response_to_dict(response)
        return answer

    def _postprocess_model_answer(self, model_answer: Dict) -> Dict:
        """
        @param model_answer:
        @return:
        """
        return model_answer

    def _postprocess_items(self, items: List[str], query: Dict) -> List[str]:
        """
        Postprocess items
        @param items:
        @param query:
        @return:
        """
        return items

    def preprocess_description(self, description: str) -> str:
        """
        Возможно, перед тем как отправить запрос на обработку, нужно его предобработать,
        Например, убрать лишние символы, привести к нижнему регистру и т.д.
        @param description:
        @return:
        """

        return description

    def _parse_item(self, description: str) -> ResultOrder:
        config = self._config['parse-item']  # Тут

        # возможно перед тем как отправить запрос на обработку, вам захочется помочь модели
        description = self.preprocess_description(description)

        # вызов модели, вам надо будем написать промт для модели, который она поймет
        model_parsed_answer = self._call_model(description)

        # Очень часто модель выдает ответ не совсем в том виде, в котором нам бы хотелось
        # Поэтому мы можем предобработать ответ. Например стандартизировать значения полей)
        model_parsed_answer = self._postprocess_model_answer(model_parsed_answer)

        # Теперь мы хотим собрать запрос для векторной базы данных. Она преобразует его в эмбеддинги и ищет похожие
        # строки в базе данных. Надо бы привести ответ в вид, который наиболее похож на то что храниться в БД
        queries = generate_query_from_answer(model_parsed_answer)

        catalog_items = []
        for query in queries:
            new_items = self._faiss_vectorstore._similarity_search_with_relevance_scores(query, k=config['relevant_k'])
            catalog_items.extend(new_items)
        catalog_items = [[c[0].page_content, c[1]] for c in catalog_items if c[1] > config['score_threshold']]
        catalog_items = sorted(catalog_items, key=lambda x: x[1], reverse=True)
        catalog_items = [c[0] for c in catalog_items]

        # Тут вы получили список строк из каталога, которые наиболее похожи на запрос. Но вот беда,
        # наши эмбеддинги специально не обучались для данной предметной области, Поэтоу они не понимают: что важно,
        # что не важно, что важно в данном контексте и т.д. Поэтому нам надо отфильтровать результаты
        catalog_items = self._postprocess_items(catalog_items, model_parsed_answer)
        catalog_items = catalog_items[:config['top_k']]
        order = ResultOrder(originalRequest=description)
        for cat_item in catalog_items:
            sku = ...  # Тут надо извлечь артикул из строки каталога
            original_description = ...  # Тут надо извлечь описание из строки каталога

            # Тут мы можем попробовать понять количество товара. Например, если в описании есть слово "штук"
            # или "метров" или "килограммов", то мы можем попробовать его извлечь и вернуть
            # Можно конечно решить дело регулярками, но это не наш путь ) Мы будем использовать модель
            quantity = model_parsed_answer.get("quantity", "")
            quantity = self._parse_units(quantity)
            r = ResultItem(sku=sku, description=original_description, quantity=quantity)
            order.order.append(r)
        return order

    def _parse_queries(self, document: List[str]) -> List[ResultOrder]:
        """
        Parse document
        @param document:
        @return:
        """
        logger.info(f"Parsing document...{document}")
        results = []
        for line in document:
            line = line.strip().lower()
            if len(line) == 0:
                continue
            try:
                orders_from_line = self._parse_item(line)
            except Exception as ex:
                logger.error(f"Something went wrong: {ex}")
                orders_from_line = ResultOrder(originalRequest=line)
                orders_from_line.errors.append(str(ex))
            results.append(orders_from_line)
        logger.info(f"Done: Parsing document. Result: {results}")
        return results

    def parse_queries(self, query: Query):
        """
        Parse query
        @param query: Query
        @return:
        """
        logger.info(f"Starting processing query {query}...")
        result = Result(id=query.id)
        reqs = query.requests.split("\n")
        try:
            orders = self._parse_queries(reqs)
            errors = []
        except Exception as ex:
            logger.error(f"Something went wrong: {ex}")
            orders = []
            errors = [str(ex)]
        result.processingResult.extend(orders)
        result.errors.extend(errors)
        return result

    def _parse_free_text(self, text: str) -> List[ResultOrder]:
        """
        Parse free text
        Здесь вам нужно будет реализовать логику, которая принимает на вход свободный текст и возвращает список заказов
        @param text:
        @return:
        """
        pass

    def parse_free_text(self, query: FreeTextQuery):
        """
        Parse free text
        @param query: Query
        @return:
        """
        logger.info(f"Starting processing query {query}...")
        result = Result(id=query.id)
        try:
            orders = self._free_text(query.text)
            errors = []
        except Exception as ex:
            logger.error(f"Something went wrong: {ex}")
            orders = []
            errors = [str(ex)]
        result.processingResult.extend(orders)
        result.errors.extend(errors)
        return result