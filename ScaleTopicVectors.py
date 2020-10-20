import numpy as np
from cognite.processpool import ProcessPool
from gensim.models.keyedvectors import KeyedVectors
import sys
import numba
from emstore.create import populate_batch_buffer_leveldb
from emstore import Emstore
import threading
import time
import memcache

class TopicVectorComputationWorker:

    def run(self, dimensions, embeddingType, topicfilename):
        # print('Init Topic Vectors')
        t0 = time.time()
        fname = topicfilename
        with open(fname) as f:
            topics = f.readlines()
        topicVectorBatch = []
        topicBatch = []
        i = 0
        value = None
        global content
        content1 = [x.strip() for x in topics]
        mc = memcache.Client(['127.0.0.1:11211'], debug=0)
        for y in content1:
            sentence_2 = y
            i = i + 1
            try:
                k = y.replace(" ", "").replace(",", "").replace("'", "").replace("\n", "")
                value = mc.get(embeddingType + k)
                if value is not None:
                    # print(k + ": Vector Exist")
                    flag = 1
                else:
                    t00 = time.time()
                    topic_vector = avg_feature_vector(sentence_2.split(), modeldatatwitter, num_features=dimensions,
                                                      index2word_set=set(modeldatatwitter.wv.index2word))
                    t01 = time.time()
                    print("Vector Generation Time", t01 - t00)
                    topicBatch.append(k)
                    topicVectorBatch.append(topic_vector)
                    mc.set(embeddingType + k, topic_vector)

                    if len(topicVectorBatch) == 100:
                        print('Topic Vector Batch sent to Emstore')
                        populate_batch_buffer_leveldb(topicBatch, topicVectorBatch, '~/topicfullcorpus')
                        topicBatch.clear()
                        topicVectorBatch.clear()
                        t1 = time.time()
                        print("100 Vector Batch Serialisation Time", t1 - t0)
                        t0 = time.time()

            except Exception as e:
                print(e)
                pass

        return


modeldatawiki = None
modeldatatwitter = None
modeldataconceptnet = None
modeldatatopics = None

lock = threading.Lock()


def loadModelWikipedia():
    global modeldatawiki
    if modeldatawiki is None:
        # Load wikipedia word vectors
        modeldatawiki = KeyedVectors.load_word2vec_format("/root/wiki.en.vec")
    return modeldatawiki


def loadModelTwitter():
    # Connect to databse
    global modeldatatwitter
    if modeldatatwitter is None:
        # Load wikipedia word vectors
        modeldatatwitter = KeyedVectors.load_word2vec_format("/root/twitterembeddings/word2vec_twitter_tokens.bin",
                                                             binary='True', unicode_errors='ignore')
    return modeldatatwitter


def loadModelConceptNet():
    # Connect to databse
    global modeldataconceptnet
    if modeldataconceptnet is None:
        # Load wikipedia word vectors
        modeldataconceptnet = KeyedVectors.load_word2vec_format("/root/commonsenseembeddings/numberbatch-en-19.08.txt")
    return modeldataconceptnet


def loadModelTopics():
    global modeldatatopics
    if modeldatatopics is None:
        # Load wikipedia word vectors
        modeldatatopics = Emstore('/root/topicDataLevelDB')
    return modeldatatopics


def scipy_cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    from scipy.spatial.distance import cosine
    return 1 - (cosine(v1, v2))


@numba.jit(target='cpu', nopython=True)
def cosine_similarity(u, v):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio


def avg_feature_vector(words, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # list containing names of words in the vocabulary
    # index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def generateSegmentsVectorFile(model, dimensions, embeddingtype, filename):
    with open(filename) as f:
        segments = f.readlines()
    segmentVectorBatch = []
    segmentBatch = []
    i = 0
    value = None
    global content
    content1 = [x.strip() for x in segments]
    import memcache
    mc = memcache.Client(['127.0.0.1:11211'], debug=0)
    for y in content1:
        sentence_2 = y
        i = i + 1
        k = y.replace(" ", "").replace(",", "").replace("'", "").replace("\n", "")
        try:
            value = mc.get(embeddingtype + k)
            if value is not None:
                # print(k + ": Vector Exist")
                flag = 1
            else:

                segment_vector = avg_feature_vector(sentence_2.split(), model, num_features=dimensions,
                                                    index2word_set=set(model.wv.index2word))
                segmentBatch.append(k)
                segmentVectorBatch.append(segment_vector)

                mc.set(embeddingtype + k, segment_vector)

                if len(segmentVectorBatch) == 100:
                    print('Segment Vector Batch sent to Emstore')
                    populate_batch_buffer_leveldb(segmentBatch, segmentVectorBatch, '~/segment1')
                    segmentBatch.clear()
                    segmentVectorBatch.clear()

        except Exception as e:
            print(e)
            pass

    return


modeldatatwitter = loadModelTwitter()
modeldatatopics = loadModelTopics()

scriptindex = sys.argv[1]

if scriptindex == '1':
    print('Generate/Serialise Topic Vectors')
    pool = ProcessPool(TopicVectorComputationWorker, 8)
    print('Process Pool Initialised')
    futures = [pool.submit_job(400, "twitter", "topicDataFile0" + str(i)) for i in
               [0, 1, 2, 3, 4, 5, 6, 7]]
    topicVectors = [f.result for f in futures]
    pool.join()
else:
    print('Generate/Serialise Segment Vectors')
    generateSegmentsVectorFile(modeldatatwitter, 400, "twitter", "segmenttwittervectors.txt")
