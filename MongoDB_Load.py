import pymongo
import pandas as pd
import Constants

mongo_client = pymongo.MongoClient(Constants.MONGODB_HOST)
dblist = mongo_client.list_database_names()
if "Loans_default" in dblist:
    mongo_client.drop_database('Loans_default')

Loans_db = mongo_client["Loans_default"]
Loans_collection = Loans_db["Loans_collection"]

print("Inserting data into mongo db...")
Loans_collection.insert_many(pd.read_csv(Constants.RAW_DATASET_PATH).to_dict('records'))
print("Done...")
print("Data inserted in Loans_default.Loans_collection.records")