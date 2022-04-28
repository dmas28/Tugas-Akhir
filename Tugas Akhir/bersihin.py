import preprocessor as p
import re

def bersihin(text):
    text = re.sub(r'[0-9`~!@#$%\[\]^&*.,|\(\)\-_+=:;\'""?\/]*', '', p.clean(text)).strip().lower()
    text = " ".join(text.split())
    return text
