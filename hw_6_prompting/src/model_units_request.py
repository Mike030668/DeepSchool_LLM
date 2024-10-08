SYSTEM_CONTENT = '''I am a system that can help you with units. Write me'''


def create_units_request(text):
    text = text.strip()
    return [
        {"role": "system",
         "content": SYSTEM_CONTENT},
        # A ВОТ ТУТ МОЖНО НАКИДАТЬ ПРИМЕРОВ
        {"role": "user", "content": text}
    ]
