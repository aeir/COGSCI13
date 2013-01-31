#
# Compound Circumplex Classifier Algorithm (C3A)
# Note about speed -- the higher the value in mod table the longer
#
###################################################################
# DELSAR.py                                                      #
#                                                                 #
# @output CSV file   [foldername]_clustix.csv                     #
#         Save files save.* (to use as a classifier)              #
#                                                                 #
# @requires MySQLdb, gensim, cPickle                              #
# @author Eugene Yuta Bann                                        #
# @version 30/01/13                                               #
#                                                                 #
###################################################################
###################################################################
############# ```INPUT```##########################################
###################################################################

foldername = "DELSAR2_ALL_6"
limit = 1000
dimension = 36
db_host = "HOST"
db_user = "USERNAME"
db_password = "PASSWORD"
db_database = "DATABASE"

############# ALL / CONTROL REGIONS #############
w_clause = ""
#moduloTable = [199,45,101,119,846,2200,85,965,516,289,108,134]
#moduloTable = [159,5,61,79,806,2160,45,925,476,249,68,94]
#moduloTable = [169,15,71,89,816,2170,55,935,486,259,78,104]
#moduloTable = [179,25,81,99,826,2180,65,945,496,269,88,114]
#moduloTable = [189,35,91,109,836,2190,75,955,506,279,98,124]
moduloTable = [156,1,56,73,599,2152,36,915,465,237,55,80]

############# EUROPE REGION #############
#w_clause = "(timezone = 'London' OR timezone = 'Amsterdam' OR timezone = 'Athens' OR timezone = 'Edinburgh' OR timezone = 'Dublin' OR timezone = 'Berlin' OR timezone = 'Paris') AND"
#moduloTable = [26,4,9,10,99,185,8,82,41,9,7,11]

############# ASIA REGION #############
#w_clause = "(timezone = 'Kuala Lumpur' OR timezone = 'Beijing' OR timezone = 'Singapore' OR timezone = 'Jakarta' OR timezone = 'Bangkok' OR timezone = 'Hong Kong' OR timezone = 'Tokyo') AND"
#moduloTable = [11,1,4,2,29,148,2,45,14,25,1,2]

############# UNITED STATES #############
#w_clause = "(timezone = 'Eastern Time (US & Canada)' OR timezone = 'Central Time (US & Canada)' OR timezone = 'Mountain Time (US & Canada)' OR timezone = 'Pacific Time (US & Canada)') AND"
#moduloTable = [60,16,35,42,291,729,30,340,193,119,40,55]

###################################################################
###################################################################

from gensim import corpora, models, similarities
import MySQLdb, time, itertools, os, errno
from collections import Counter
import cPickle as pickle
                
try:
    os.makedirs(foldername)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

emotionKeywords = ['angry', 'ashamed', 'calm', 'depressed', 'excited', 'happy', 'interested', 'sad', 'scared', 'sleepy', 'stressed', 'surprised']
db = MySQLdb.connect(db_host, db_user, db_password, db_database)
cursor = db.cursor()

t0 = time.clock()
print "DELSAR2(" + str(limit) + ") with " + str(dimension) + " dimensions. Writing to /"+foldername

# Corpus class to load each document iteratively from database
class MyCorpus(object):
    def __iter__(self):
        for emotion in range(len(emotionKeywords)):
            try:
                sql = "SELECT text, @idx:=@idx+1 AS idx FROM `e` WHERE %s emotion = '%s' HAVING MOD(idx,%d) = 0 LIMIT %d" % (w_clause,emotionKeywords[emotion],moduloTable[emotion],limit)
                print sql
                cursor.execute("SELECT @idx:=-1;")
                cursor.execute(sql)
                results = cursor.fetchall()
                for row in results:
                    yield dictionary.doc2bow(row[0].lower().split())
            except MySQLdb.Error, e:
                print "Error %d: %s" % (e.args[0], e.args[1])

print "Creating dictionary..."
dictionary = corpora.Dictionary()
for emotion in range(len(emotionKeywords)):
    try:
        sql = "SELECT text, @idx:=@idx+1 AS idx FROM `e` WHERE %s emotion = '%s' HAVING MOD(idx,%d) = 0 LIMIT %d" % (w_clause,emotionKeywords[emotion],moduloTable[emotion],limit)
        print sql
        cursor.execute("SELECT @idx:=-1;")
        cursor.execute(sql)
        results = cursor.fetchall()
        dictionary.add_documents(row[0].lower().split() for row in results)
    except MySQLdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
# We don't use a stoplist
stoplist = set("a".split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
# We get rid of words that only occur once in the entire corpus
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
# Remove gaps in id sequence after words that were removed
dictionary.compactify()
# Print dictionary information
print dictionary
pickle.dump(dictionary, open(foldername+"/save.dict", "wb"))

print "Creating corpus object..."
# Create corpus object from database iteratively, doesn't load into memory
corpus_memory_friendly = MyCorpus()
print "Done."

print "Generating LSA Space..."     
# use a log-entropy model to weight terms
logent = models.LogEntropyModel(corpus_memory_friendly)
# initialize an LSI transformation
corpus_logent = logent[corpus_memory_friendly]
lsi = models.LsiModel(corpus_logent, id2word=dictionary, num_topics=dimension)
pickle.dump(logent, open(foldername+"/save.logent", "wb"))
pickle.dump(lsi, open(foldername+"/save.lsi", "wb"))
# create a double wrapper over the original corpus: bow->logent->fold-in-lsi
corpus_lsi = lsi[corpus_logent]
print corpus_lsi

print "Indexing LSA Space..."
index = similarities.Similarity(foldername+"/temp", corpus_lsi, num_features=dimension)     
pickle.dump(index, open(foldername+"/save.ind", "wb"))
print "Done."

print "Creating index map..."
emap = [] # word, emotion
for emotion in range(len(emotionKeywords)):
    try:
        sql = "SELECT text, @idx:=@idx+1 AS idx FROM `e` WHERE %s emotion = '%s' HAVING MOD(idx,%d) = 0 LIMIT %d" % (w_clause,emotionKeywords[emotion],moduloTable[emotion],limit)
        print sql
        cursor.execute("SELECT @idx:=-1;")
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            emap.append([row[0], emotion])
    except MySQLdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
pickle.dump(emap, open(foldername+"/save.map", "wb"))
print "Done."

print "LSA in %.2f mins." % ((time.clock() - t0)/60)
t0 = time.clock()

###############################################################

# DELSAR

print "Clustering Documents..."
t1 = time.clock()
mapEmotion = [] # index, word
queryMatch = [] # word, max cosine (ELSA)/index (DELSAR)
sequentialCount = 0 # We need this to keep track of streaming
for emotion in range(len(emotionKeywords)):
    try:
        sql = "SELECT text, @idx:=@idx+1 AS idx FROM `e` WHERE %s emotion = '%s' HAVING MOD(idx,%d) = 0 LIMIT %d" % (w_clause,emotionKeywords[emotion],moduloTable[emotion],limit)
        print sql
        cursor.execute("SELECT @idx:=-1;")
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            # For each document, a is actual term, b is most similar term using LSA
            a = emotionKeywords[emotion]
            # Delete the emotion keyword from the document
            tweet = row[0].replace(emotionKeywords[emotion], "")
            # Convert the document to logent LSA space
            vec_bow = dictionary.doc2bow(tweet.lower().split())
            query_lsi = lsi[logent[vec_bow]]
            # Compute document similarity
            sims = index[query_lsi]
            sims = sorted(enumerate(sims), key=lambda item: item[0])
            # Delete the current document from the array
            del sims[sequentialCount]
            # DELSAR just wants to know what emotion is the most similar document
            b = sims.index(max(sims, key=lambda x: x[1]))
            queryMatch.append([a,b]) 
            mapEmotion.append(emotionKeywords[emotion]) 
            sequentialCount += 1
            # The next line for debugging
            # print str((limit*len(emotionKeywords))-sequentialCount)
    except MySQLdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])      
clusters = []
for a in range(len(emotionKeywords)):
    for vec in queryMatch:
        if vec[0] == emotionKeywords[a]:
              clusters.append((vec[0],mapEmotion[vec[1]]))
pickle.dump(clusters, open(foldername+"/save.clust", "wb"))
print "Clustering in %.2f mins." % ((time.clock() - t0)/60)

print "Saving Clustering Matrix as CSV..."
e_vecs = []
c = Counter(clusters)    
for i in range(len(emotionKeywords)):
    e_vector = []
    for j in range(len(emotionKeywords)):
        for vec in c:
            if vec[1] == emotionKeywords[j]:
                if vec[0] == emotionKeywords[i]:
                    e_vector.append(c[vec])    
    e_vecs.append(e_vector)
vlines = ","
for z in range(len(emotionKeywords)):
    vlines += emotionKeywords[z] + ","
vlines += "\n"
for x in range(len(e_vecs[0])):
    vlines += emotionKeywords[x] + ","
    for y in range(len(e_vecs)):
        try:
            vlines += str(e_vecs[y][x])
        except IndexError:
            vlines += str(0)
        if y != (len(e_vecs)-1):
            vlines += ","
    vlines += "\n"
filename = foldername+"_clustix.csv"                
try:
    f = open(foldername+"/"+filename, "w")
    try:
        f.writelines(vlines)
    finally:
        f.close()
except IOError:
    pass
print "Done."

db.close()

######### END #########

