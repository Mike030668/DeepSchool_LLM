import logging
import re

EMPTY_UNIT_ANSWER = {"mass": "", "mass_units": "", "length": "", "length_units": "", "items_count": ""}

logger = logging.getLogger(__name__)


def maybe_manual_parse_units(item_description):
    """
    This function will try to parse the item description and return a JSON object with the units
    :param item_description:
    :return:
    """
    logger.info(f"Trying to use regexp unit parser on: {item_description}")
    item_description = item_description.strip().replace(",", ".")
    try:
        items_count = str(int(item_description))
        return {"items_count": items_count, "mass": "", "mass_units": "", "length": "", "length_units": ""}
    except ValueError:
        pass
    mass_regex = r"(\d+\.?\d*)\s*(kg|kgs|kilos|kilogramos|kilogramo|k|toneladas|tons|ton|t)"
    length_regex = r"(\d+\.?\d*)\s*(metr|metros|m|mts|mt|sm|smts|mm|mms)"
    units_regex = r"(\d+)\s*(units|unidades|uds|barras|bars|piezas|pieces|ud|u|pcs|pc|pz|pza|pzs)"

    # Extract mass
    mass_match = re.search(mass_regex, item_description, re.IGNORECASE)
    mass = mass_match.group(1) if mass_match else ""
    mass_units = mass_match.group(2) if mass_match else ""

    # Extract length
    length_match = re.search(length_regex, item_description, re.IGNORECASE)
    length = length_match.group(1) if length_match else ""
    length_units = length_match.group(2) if length_match else ""
    # Extract number of units
    items_count_match = re.search(units_regex, item_description, re.IGNORECASE)
    items_count = items_count_match.group(1) if items_count_match else ""

    # Structure the JSON object
    json_object = {
        "mass": mass,
        "mass_units": mass_units,
        "length": length,
        "length_units": length_units,
        "items_count": items_count
    }
    if json_object != EMPTY_UNIT_ANSWER and json_object["items_count"] == "":
        json_object["items_count"] = 1
    return json_object


def postprocess_units_answer(answer):
    """
    Postprocess the answer
    """
    return answer


def convert_units_to_metric(answer):
    """
    Convert the units to the metric system
    :param answer:
    :return:
    """
    logger.info(f"Converting units to metric: {answer}")
    if len(answer.get("mass", "")) > 0:
        try:
            mass = float(answer.get("mass"))
            mass_units = answer.get("mass_units", "")
            if mass_units == "tons":
                mass = mass * 1000
            answer["mass"] = str(mass)
            answer["mass_units"] = "kg"
        except ValueError as ex:
            print(ex)

    if len(answer.get("length", "")) > 0:
        try:
            length = float(answer.get("length"))
            length_units = answer.get("length_units", "")
            if length_units == "sm":
                length = length / 100
            if length_units == "mm":
                length = length / 1000
            answer["length"] = str(length)
            answer["length_units"] = "m"
        except ValueError as ex:
            print(ex)
    return answer
