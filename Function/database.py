import pymongo


class database(object):

    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://127.0.0.1", 27017)
        #self.client.drop_database("GS_HKUST")
        self.db = self.client["GS_HKUST"]
        self.reuters = self.db["reuters"]
        #self.reuters.ensure_index([("uid", pymongo.ASCENDING), ("unique", True), \
        #         ("dropDups", True)])
        self.reuters.ensure_index("uid", unique = True)
        #self.connection = pymongo.Connection()
    def insert(self, items):
        try:
            self.reuters.insert(items)
            #print "insert item %s" % items
        except:
            try:
                print "omit %s, %s, %s" % (items["ticker"], items["date"], items["title"])
            except:
                pass
            pass
    def update(self, items):
        self.reuters.update(items)
    def getDate(self, date):
        cursor = self.reuters.find({"date":date})
        return cursor
    def getTags(self):
        cursor = self.reuters.find({})
        return cursor
    def getId(self, uid):
        cursor = self.reuters.find({"uid": uid})
        return cursor
