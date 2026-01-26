import json

class database():
    def __init__(self):
        with open("database.json", "r") as file:
            database = json.load(file)

    def get_embed(self, num):
        return database[num]["embed"]
    
    def get_url(self, num):
        return database[num]["url"]