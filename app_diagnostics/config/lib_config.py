import json

CONFIG_PATH = 'app_diagnostics/config.json'

class ConfigJSON():
    def __init__(self, path):
        self.path = path

    def get_dicts(self):
        # Открываем файл для чтения
        with open(self.path, 'r') as json_file:
            return json.load(json_file)
    
def setup_config():
    configFromFile = ConfigJSON(CONFIG_PATH)
    allDicts = configFromFile.get_dicts()
    return allDicts