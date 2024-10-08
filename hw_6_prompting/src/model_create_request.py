SYSTEM_CONTENT = "I AM YOU SYSTEM PROMPT!! WRITE ME"

def create_request(text):
    return [
        {"role": "system",
         "content": SYSTEM_CONTENT},
        # A ВОТ ТУТ НАДО БЫ НАКИДАТЬ ПРИМЕРОВ
        {"role": "user", "content": text}]
