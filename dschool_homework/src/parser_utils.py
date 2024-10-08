import logging
import re
from ast import literal_eval

from src.units_parser_utils import EMPTY_UNIT_ANSWER

logger = logging.getLogger(__name__)


def catalog_description_unification(text):
    """
    Unify catalog description
    """
    return text


def answer2content(result):
    try:
        return result.json()['choices'][0]['message']['content']
    except Exception as ex:
        logger.warning(f"Unable to parse an answer: {ex}")
    return None


def response_to_dict(answer):
    """
    Модель вовзращает ответ в виде строки, но нам нужно в виде словаря.
    Но мы же помним, что модель умеет возвращать .. ну например json
    надо бы извлечь из ответа json и вернуть его в виде словаря
    :param answer:
    :return:
    """
    raise NotImplementedError



def generate_query_from_answer(answer):
    """
    Generate query from answer
    :param answer:
    :return:
    """
    raise NotImplementedError
