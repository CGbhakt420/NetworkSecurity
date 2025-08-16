
# from pymongo.mongo_client import MongoClient

# uri = "mongodb+srv://sanchitkr007:ZX6kCXQBuAMtMnNN@cluster0.fxuqa4d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# # Create a new client and connect to the server
# client = MongoClient(uri)

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

import os
import sys
import json
from dotenv import load_dotenv
import certifi

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_converter(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            records = df.to_dict(orient='records')
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client = pymongo.MongoClient(MONGO_URL)
            
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__=="__main__":
    FILE_PATH = "Network_Data/phishingData.csv"
    DATABASE = "NetworkSecurity"
    COLLECTION = "phishingData"
    networkobj=NetworkDataExtract()
    records=networkobj.csv_to_json_converter(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, COLLECTION)
    print(no_of_records)
